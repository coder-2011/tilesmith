import builtins
import base64
import contextlib
import fcntl
import hashlib
import http.server
import inspect
import io
import json
import os
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WARMUP = 25
BENCH_ITERATIONS = 100
BENCH_SAMPLES = 7
MAX_SOURCE_BYTES = 64 * 1024 * 1024
OPTIMIZE_LOCK = threading.Lock()
REMOTE_ROOT = "/tmp/tilesmith"
REMOTE_COMMAND_TIMEOUT_MS = 20 * 60 * 1000
UPLOAD_CHUNK_BYTES = 384 * 1024  # Divisible by 3 so base64 chunks concatenate.


@dataclass(frozen=True)
class _ComputeConfig:
    """Hold exported ComputeSDK credentials and one sandbox lifetime."""

    api_key: str = field(repr=False)
    gateway_url: str
    sandbox_timeout: int
    template: str
    token_id: str = field(repr=False)
    token_secret: str = field(repr=False)


@dataclass(frozen=True)
class _Settings:
    agent_root: Path
    cache_dir: Path
    chip: str
    compute: _ComputeConfig | None
    config: str
    iterations: int
    min_improvement: float
    model: str
    sandbox: str
    skills: tuple[str, ...]
    timeout: str


# compile returns a normal torch.compile callable while routing Inductor graphs through Tilesmith.
def compile(model, *, example_inputs, sandbox=None, chip=None):
    if sandbox not in (None, "computesdk"):
        raise ValueError("sandbox must be None or 'computesdk'")
    if sandbox is None and chip is not None:
        raise ValueError("chip is only valid when sandbox is set")
    if sandbox == "computesdk" and (not isinstance(chip, str) or not chip.strip()):
        raise ValueError("sandbox='computesdk' requires a non-empty chip")
    settings = _read_settings(sandbox, chip or "")

    import torch
    from torch._inductor.compile_fx import compile_fx

    if not callable(model):
        raise TypeError("model must be callable")
    if not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be a tuple of positional model arguments")
    leaves = torch.utils._pytree.tree_leaves(example_inputs)
    cuda_inputs = [
        value
        for value in leaves
        if torch.is_tensor(value) and value.device.type == "cuda"
    ]
    if not cuda_inputs:
        raise ValueError("example_inputs must contain at least one CUDA tensor")
    if sandbox == "computesdk":
        device = cuda_inputs[0].device
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device_name = torch.cuda.get_device_name(index)
        if not _chip_matches(chip, device_name):
            raise ValueError(
                f"chip={chip!r} does not match local execution device {device_name!r}"
            )
    if "inner_compile" not in inspect.signature(compile_fx).parameters:
        raise RuntimeError(
            "this PyTorch version does not expose Inductor's inner_compile hook"
        )

    return torch.compile(model, backend=_Backend(settings))


# _read_settings keeps the public API small while making expensive agent choices explicit in the environment.
def _read_settings(sandbox, chip):
    iterations = int(os.environ.get("TILESMITH_ITERATIONS", "4"))
    min_improvement = float(os.environ.get("TILESMITH_MIN_IMPROVEMENT", "0.01"))
    if iterations < 0:
        raise ValueError("TILESMITH_ITERATIONS must be non-negative")
    if min_improvement < 0 or min_improvement >= 1:
        raise ValueError("TILESMITH_MIN_IMPROVEMENT must be in [0, 1)")

    cache_default = Path.home() / ".cache" / "tilesmith"
    return _Settings(
        agent_root=Path(os.environ.get("TILESMITH_AGENT_ROOT", ROOT))
        .expanduser()
        .resolve(),
        cache_dir=Path(os.environ.get("TILESMITH_CACHE_DIR", cache_default))
        .expanduser()
        .resolve(),
        chip=chip,
        compute=_read_compute_config() if sandbox == "computesdk" else None,
        config=os.environ.get("CLIPROXY_CONFIG", ""),
        iterations=iterations,
        min_improvement=min_improvement,
        model=os.environ.get("TILESMITH_MODEL", "gpt-5.5"),
        sandbox=sandbox or "local",
        skills=tuple(_default_skills()),
        timeout=os.environ.get("TILESMITH_TIMEOUT", "20m"),
    )


# _read_compute_config validates every gateway input before provisioning a billable sandbox.
def _read_compute_config():
    try:
        from computesdk import DEFAULT_GATEWAY_URL
    except ImportError as error:
        raise RuntimeError(
            "sandbox='computesdk' requires the httpx Python package"
        ) from error

    names = (
        "COMPUTESDK_API_KEY",
        "COMPUTESDK_TEMPLATE",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    )
    values = {name: os.environ.get(name, "") for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("sandbox='computesdk' requires exported " + ", ".join(missing))
    sandbox_timeout = int(os.environ.get("TILESMITH_SANDBOX_TIMEOUT", "3600"))
    if sandbox_timeout <= 60:
        raise ValueError("TILESMITH_SANDBOX_TIMEOUT must be greater than 60")
    return _ComputeConfig(
        api_key=values["COMPUTESDK_API_KEY"],
        gateway_url=os.environ.get("COMPUTESDK_TRIBUTARY_URL", DEFAULT_GATEWAY_URL),
        sandbox_timeout=sandbox_timeout,
        template=values["COMPUTESDK_TEMPLATE"],
        token_id=values["MODAL_TOKEN_ID"],
        token_secret=values["MODAL_TOKEN_SECRET"],
    )


# _chip_matches accepts a provider selector such as H100 against its full device name.
def _chip_matches(requested, observed):
    return requested.rstrip("!").lower() in observed.lower()


# _default_skills gives the kernel agent the local API and CUDA guidance when those files exist.
def _default_skills():
    skill_root = Path.home() / ".codex" / "skills"
    cuda_root = skill_root / "high-performance-cuda"
    paths = [
        skill_root / "check-normal" / "SKILL.md",
        cuda_root / "SKILL.md",
        cuda_root / "references" / "core-principles.md",
        cuda_root / "references" / "autotuning-and-benchmarking.md",
        cuda_root / "references" / "numerical-correctness.md",
        cuda_root / "references" / "triton-and-codegen.md",
    ]
    return [str(path) for path in paths if path.is_file()]


class _Backend:
    # __init__ retains immutable settings across forward and lazy backward compilation.
    def __init__(self, settings):
        self.settings = settings

    # __call__ delegates Dynamo's graph to the same compile_fx entrypoint as the built-in Inductor backend.
    def __call__(self, graph_module, example_inputs):
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(
            graph_module,
            example_inputs,
            inner_compile=self._inner_compile,
            config_patches={"triton.cudagraphs": False},
        )

    # _inner_compile adds a lazy optimizer after Inductor has produced and loaded one generated Python wrapper.
    def _inner_compile(self, graph_module, example_inputs, **kwargs):
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.output_code import (
            CompiledFxGraph,
            CompiledFxGraphConstantsWithGm,
        )

        compiled = compile_fx_inner(graph_module, example_inputs, **kwargs)
        if not isinstance(compiled, CompiledFxGraph):
            return compiled
        if "triton_" not in compiled.source_code:
            return compiled
        if compiled.mutated_input_idxs:
            warnings.warn(
                "Tilesmith left an input-mutating graph on ordinary Inductor",
                stacklevel=2,
            )
            return compiled
        if not _is_raw_generated_call(compiled):
            warnings.warn(
                "Tilesmith left a wrapped Inductor graph unchanged", stacklevel=2
            )
            return compiled

        constants = CompiledFxGraphConstantsWithGm(graph_module).unwrap(compiled)
        compiled.current_callable = _LazyKernel(
            compiled.current_callable,
            compiled.source_code,
            constants,
            self.settings,
        )
        return compiled


# _is_raw_generated_call prevents replacing a callable after Inductor has attached semantic runtime wrappers.
def _is_raw_generated_call(compiled):
    from torch._inductor.codecache import get_path

    function = compiled.current_callable
    code = getattr(function, "__code__", None)
    if code is None or function.__name__ != "call":
        return False
    generated_path = Path(get_path(compiled.cache_key, "py")[2]).resolve()
    return Path(code.co_filename).resolve() == generated_path


class _LazyKernel:
    # __init__ defers optimization until real runtime tensors are available for validation and timing.
    def __init__(self, baseline, source, constants, settings):
        self.baseline = baseline
        self.source = _canonical_source(source)
        self.constants = constants
        self.settings = settings
        self.installed = None
        self.lock = threading.Lock()

    # __call__ installs exactly one validated artifact, then preserves Inductor's boxed calling convention.
    def __call__(self, inputs):
        if self.installed is None:
            with self.lock:
                if self.installed is None:
                    self.installed = self._install(inputs)
        return self.installed(inputs)

    # _install reuses a validated cache entry or runs the complete agent loop against these runtime inputs.
    def _install(self, inputs):
        import torch

        cuda_inputs = [
            value
            for value in inputs
            if torch.is_tensor(value) and value.device.type == "cuda"
        ]
        if not cuda_inputs:
            return self.baseline
        device = cuda_inputs[0].device
        if self.settings.compute is not None:
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            device_name = torch.cuda.get_device_name(index)
            if not _chip_matches(self.settings.chip, device_name):
                raise RuntimeError(
                    f"sandbox chip {self.settings.chip!r} does not match "
                    f"execution device {device_name!r}"
                )
        cache_key = _cache_key(self.source, self.settings, device)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

        with _cache_lock(self.settings.cache_dir / f"{cache_key}.lock"):
            cached = _read_cache(self.settings.cache_dir, cache_key)
            if cached is None:
                with OPTIMIZE_LOCK, _preserve_rng():
                    if self.settings.compute is not None:
                        with _ComputeEvaluator(
                            self.source, self.constants, inputs, self.settings
                        ) as evaluator:
                            optimized = _run_agent(
                                self.source, self.settings, evaluator
                            )
                    else:
                        evaluator = _Evaluator(
                            self.baseline, self.constants, inputs, device
                        )
                        optimized = _run_agent(self.source, self.settings, evaluator)
                cached = _write_cache(self.settings.cache_dir, cache_key, optimized)
        if cached.read_text(encoding="utf-8") == self.source:
            return self.baseline
        return _load_call(cached, self.constants)


# _canonical_source removes PyTorch's incrementing, non-executable AOT ID comment.
def _canonical_source(source):
    first_line, separator, remainder = source.partition("\n")
    if first_line.startswith("# AOT ID:"):
        return remainder if separator else ""
    return source


# _cache_key binds an accepted source file to its graph, agent policy, software, and NVIDIA target.
def _cache_key(source, settings, device):
    import torch
    import triton

    index = device.index if device.index is not None else torch.cuda.current_device()
    identity = {
        "bench_iterations": BENCH_ITERATIONS,
        "bench_samples": BENCH_SAMPLES,
        "capability": torch.cuda.get_device_capability(index),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(index),
        "iterations": settings.iterations,
        "min_improvement": settings.min_improvement,
        "model": settings.model,
        "sandbox": settings.sandbox,
        "sandbox_chip": settings.chip,
        "sandbox_gateway": (
            settings.compute.gateway_url if settings.compute is not None else ""
        ),
        "sandbox_template": (
            settings.compute.template if settings.compute is not None else ""
        ),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "warmup": WARMUP,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    digest.update(source.encode())
    for path in [
        Path(__file__),
        ROOT / "computesdk.py",
        ROOT / "sandbox_worker.py",
        settings.agent_root / "main.go",
        *map(Path, settings.skills),
    ]:
        if path.is_file():
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# _cache_lock serializes generation so concurrent first calls never observe a partial artifact.
@contextlib.contextmanager
def _cache_lock(path):
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# _preserve_rng prevents trial executions from consuming randomness meant for the user's first model call.
@contextlib.contextmanager
def _preserve_rng():
    import torch

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


# _read_cache accepts only a manifest whose key and source digest match the artifact on disk.
def _read_cache(cache_dir, cache_key):
    artifact = cache_dir / f"{cache_key}.py"
    manifest = cache_dir / f"{cache_key}.json"
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        source = artifact.read_bytes()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if metadata != {
        "cache_key": cache_key,
        "sha256": hashlib.sha256(source).hexdigest(),
    }:
        return None
    return artifact


# _write_cache syntax-checks the final candidate and atomically publishes source before its manifest.
def _write_cache(cache_dir, cache_key, source):
    builtins.compile(source, f"tilesmith-cache-{cache_key}.py", "exec")
    artifact = cache_dir / f"{cache_key}.py"
    manifest = cache_dir / f"{cache_key}.json"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    with tempfile.NamedTemporaryFile(
        "w", dir=cache_dir, suffix=".py", delete=False
    ) as source_file:
        source_file.write(source)
        source_temp = Path(source_file.name)
    with tempfile.NamedTemporaryFile(
        "w", dir=cache_dir, suffix=".json", delete=False
    ) as manifest_file:
        json.dump(
            {"cache_key": cache_key, "sha256": source_hash},
            manifest_file,
            sort_keys=True,
        )
        manifest_temp = Path(manifest_file.name)
    os.replace(source_temp, artifact)
    os.replace(manifest_temp, manifest)
    return artifact


# _load_call imports one accepted source file with the constants Inductor attached to the original module.
def _load_call(path, constants):
    from torch._inductor.codecache import PyCodeCache, code_hash

    source = path.read_text(encoding="utf-8")
    module = PyCodeCache.load_by_key_path(code_hash(source), str(path), attrs=constants)
    return module.call


# _cpu_copy makes remote payloads device-independent while preserving tensor layouts.
def _cpu_copy(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu(memory_format=torch.preserve_format)
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    return value


class _Evaluator:
    # __init__ snapshots real inputs before either boxed callable can clear its argument list.
    def __init__(self, baseline, constants, inputs, device):
        import torch

        self.baseline = baseline
        self.constants = constants
        self.device = device
        self.inputs = tuple(inputs)
        self.directory = tempfile.TemporaryDirectory(prefix="tilesmith-evaluator-")
        self.lock = threading.Lock()
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        capability = torch.cuda.get_device_capability(index)
        self.chip = (
            f"{torch.cuda.get_device_name(index)} (sm_{capability[0]}{capability[1]})"
        )

    # evaluate rejects incorrect source before returning an interleaved CUDA-event comparison.
    def evaluate(self, source):
        import torch
        from torch._inductor.codecache import code_hash

        if len(source.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("candidate source exceeds 64 MiB")
        candidate_path = Path(self.directory.name) / f"{code_hash(source)}.py"
        if not candidate_path.exists():
            candidate_path.write_text(source, encoding="utf-8")
        with self.lock, torch.cuda.device(self.device), torch.no_grad():
            candidate = _load_call(candidate_path, self.constants)
            for variant in ("normal", "zero", "high"):
                cpu_state = torch.random.get_rng_state()
                cuda_states = torch.cuda.get_rng_state_all()
                expected = self.baseline(list(self._inputs(variant)))
                torch.random.set_rng_state(cpu_state)
                torch.cuda.set_rng_state_all(cuda_states)
                actual = candidate(list(self._inputs(variant)))
                torch.testing.assert_close(actual, expected, equal_nan=True)
            torch.cuda.synchronize(self.device)
            report = self._benchmark(candidate)
            report["chip"] = self.chip
            report["reference"] = "torch_inductor_generated"
            return report

    # _inputs clones strides and optionally changes values without changing compiled shapes or dtypes.
    def _inputs(self, variant):
        import torch

        values = []
        for value in self.inputs:
            if not torch.is_tensor(value):
                values.append(value)
                continue
            cloned = value.detach().clone(memory_format=torch.preserve_format)
            if variant == "zero" and (
                cloned.is_floating_point() or cloned.is_complex()
            ):
                cloned.zero_()
            if variant == "high" and (
                cloned.is_floating_point() or cloned.is_complex()
            ):
                cloned.mul_(4)
            values.append(cloned)
        return tuple(values)

    # _benchmark alternates baseline/candidate order and returns paired samples plus their median ratio.
    def _benchmark(self, function):
        import torch

        inputs = self._inputs("normal")
        functions = (("torch_inductor", self.baseline), ("candidate_triton", function))
        for _ in range(WARMUP):
            for _, measured in functions:
                measured(list(inputs))
        torch.cuda.synchronize(self.device)

        samples = {name: [] for name, _ in functions}
        for sample in range(BENCH_SAMPLES):
            order = functions if sample % 2 == 0 else tuple(reversed(functions))
            for name, measured in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(BENCH_ITERATIONS):
                    measured(list(inputs))
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / BENCH_ITERATIONS)

        ratios = [
            candidate / baseline
            for baseline, candidate in zip(
                samples["torch_inductor"], samples["candidate_triton"]
            )
        ]
        return {
            "benchmark_iterations": BENCH_ITERATIONS,
            "benchmark_samples": BENCH_SAMPLES,
            "candidate_ms": statistics.median(samples["candidate_triton"]),
            "candidate_samples_ms": samples["candidate_triton"],
            "score": statistics.median(ratios),
            "torch_inductor_ms": statistics.median(samples["torch_inductor"]),
            "torch_inductor_samples_ms": samples["torch_inductor"],
            "warmup": WARMUP,
        }


class _ComputeSandbox:
    """Own one reusable ComputeSDK sandbox for an entire graph optimization track."""

    # __init__ retains only the placement request and optional host runtime versions.
    def __init__(self, config, chip, versions=None):
        self.config = config
        self.requested_chip = chip
        self.versions = versions or {}
        self.context = None
        self.sandbox = None
        self.chip = chip
        # Reserve the final minute of the sandbox lifetime for teardown.
        self.command_timeout_ms = min(
            REMOTE_COMMAND_TIMEOUT_MS,
            (config.sandbox_timeout - 60) * 1000,
        )

    # __enter__ provisions once, uploads the worker, and verifies the selected GPU.
    def __enter__(self):
        from computesdk import ephemeral_sandbox

        self.context = ephemeral_sandbox(
            api_key=self.config.api_key,
            token_id=self.config.token_id,
            token_secret=self.config.token_secret,
            gateway_url=self.config.gateway_url,
            timeout_ms=self.config.sandbox_timeout * 1000,
            template_id=self.config.template,
            gpu=self.requested_chip,
        )
        self.sandbox = self.context.__enter__()
        try:
            created = self.sandbox.run(
                f"mkdir -p {REMOTE_ROOT}/cache",
                timeout_ms=60_000,
            )
            _require_remote_success(created, "sandbox setup")
            self.write_text(
                "worker.py",
                (ROOT / "sandbox_worker.py").read_text(encoding="utf-8"),
            )
            request = {
                "chip": self.requested_chip,
                "iterations": BENCH_ITERATIONS,
                "samples": BENCH_SAMPLES,
                "warmup": WARMUP,
                **self.versions,
            }
            self.write_text("request.json", json.dumps(request, sort_keys=True))
            metadata = self.run("probe", timeout_ms=60_000)
            capability = str(metadata["compute_capability"]).replace(".", "")
            self.chip = f"{metadata['gpu']} (sm_{capability})"
            return self
        except BaseException:
            self.context.__exit__(*sys.exc_info())
            raise

    # __exit__ destroys the sandbox even when an agent or benchmark call fails.
    def __exit__(self, error_type, error, traceback):
        return self.context.__exit__(error_type, error, traceback)

    # write_text transfers one source or configuration file into the worker root.
    def write_text(self, name, content):
        self.sandbox.write_text(f"{REMOTE_ROOT}/{name}", content)

    # write_bytes transfers one binary payload as bounded base64 chunks.
    def write_bytes(self, name, content):
        chunk_names = []
        for index, offset in enumerate(range(0, len(content), UPLOAD_CHUNK_BYTES)):
            chunk = content[offset : offset + UPLOAD_CHUNK_BYTES]
            chunk_name = f"{name}-{index:05d}.b64"
            encoded = base64.b64encode(chunk).decode("ascii")
            self.write_text(chunk_name, encoded)
            chunk_names.append(f"{REMOTE_ROOT}/{chunk_name}")
        if not chunk_names:
            raise ValueError("remote binary payload must not be empty")
        source_files = " ".join(map(shlex.quote, chunk_names))
        destination = shlex.quote(f"{REMOTE_ROOT}/{name}")
        result = self.sandbox.run(
            f"cat {source_files} | base64 -d > {destination}",
            timeout_ms=self.command_timeout_ms,
        )
        _require_remote_success(result, f"upload {name}")

    # run executes one worker action and returns its final JSON object.
    def run(self, action, timeout_ms=None):
        if timeout_ms is None:
            timeout_ms = self.command_timeout_ms
        command = shlex.join(["python3", f"{REMOTE_ROOT}/worker.py", action])
        result = self.sandbox.run(
            command,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_ms=timeout_ms,
        )
        _require_remote_success(result, f"remote {action}")
        for line in reversed(result.stdout.splitlines()):
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(report, dict):
                report["provider"] = self.sandbox.provider
                return report
        raise RuntimeError(f"remote {action} returned no JSON report")


# _require_remote_success preserves bounded worker diagnostics without exposing credentials.
def _require_remote_success(result, operation):
    if result.exit_code == 0:
        return
    details = (result.stderr or result.stdout).strip()[-4_000:]
    raise RuntimeError(
        f"{operation} failed with exit code {result.exit_code}: {details}"
    )


class _ComputeEvaluator:
    """Evaluate generated candidates in one warm ComputeSDK sandbox."""

    # __init__ snapshots source, constants, and runtime inputs before upload.
    def __init__(self, baseline_source, constants, inputs, settings):
        import torch
        import triton

        self.baseline_source = baseline_source
        self.worker = _ComputeSandbox(
            settings.compute,
            settings.chip,
            {"torch": str(torch.__version__), "triton": str(triton.__version__)},
        )
        payload = {
            "constants": _cpu_copy(constants),
            "inputs": _cpu_copy(tuple(inputs)),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self.payload = buffer.getvalue()
        self.chip = settings.chip

    # __enter__ provisions the track and uploads inputs shared by every candidate.
    def __enter__(self):
        self.worker.__enter__()
        try:
            self.worker.write_text("baseline.py", self.baseline_source)
            self.worker.write_bytes("payload.pt", self.payload)
            self.chip = self.worker.chip
            return self
        except BaseException:
            self.worker.__exit__(*sys.exc_info())
            raise

    # __exit__ tears down the remote track after the agent selects its artifact.
    def __exit__(self, error_type, error, traceback):
        return self.worker.__exit__(error_type, error, traceback)

    # evaluate uploads only the changing candidate and invokes the trusted worker.
    def evaluate(self, source):
        self.worker.write_text("candidate.py", source)
        return self.worker.run("compare")


# _evaluator_server exposes the in-process CUDA evaluator only to a token-authenticated localhost command.
@contextlib.contextmanager
def _evaluator_server(evaluator):
    token = secrets.token_urlsafe(32)

    class Handler(http.server.BaseHTTPRequestHandler):
        # do_POST evaluates raw source and returns structured evidence followed by its score.
        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_SOURCE_BYTES:
                self.send_error(413)
                return
            try:
                result = evaluator.evaluate(self.rfile.read(length).decode())
                report = json.dumps(result, sort_keys=True)
                response = f"{report}\n{result['score']:.9g}\n".encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except Exception as error:
                self.send_error(400, str(error))

        # log_message suppresses access-log noise while the Go agent already reports each trial.
        def log_message(self, format, *args):
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", token
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


# _run_agent lets the existing Go agent converse, search, evaluate, and atomically choose the final source.
def _run_agent(source, settings, evaluator):
    if shutil.which("go") is None:
        raise RuntimeError("go is required to run the Tilesmith agent")
    if not (settings.agent_root / "go.mod").is_file():
        raise RuntimeError(f"TILESMITH_AGENT_ROOT has no go.mod: {settings.agent_root}")
    if settings.iterations > 0 and not settings.config:
        raise RuntimeError(
            "CLIPROXY_CONFIG is required when TILESMITH_ITERATIONS is greater than zero"
        )

    with tempfile.TemporaryDirectory(
        prefix="tilesmith-agent-", dir=settings.cache_dir
    ) as directory:
        kernel_path = Path(directory) / "kernel.py"
        kernel_path.write_text(source, encoding="utf-8")
        with _evaluator_server(evaluator) as (url, token):
            evaluate = shlex.join(
                [sys.executable, str(Path(__file__)), "_evaluate", url, token]
            )
            command = [
                "go",
                "run",
                ".",
                "-kernel",
                str(kernel_path),
                "-evaluate",
                evaluate,
                "-chip",
                evaluator.chip,
                "-iterations",
                str(settings.iterations),
                "-min-improvement",
                str(settings.min_improvement),
                "-timeout",
                settings.timeout,
            ]
            if settings.config:
                command.extend(["-config", settings.config])
            if settings.model:
                command.extend(["-model", settings.model])
            for skill in settings.skills:
                command.extend(["-skill", skill])
            process = subprocess.run(
                command,
                cwd=settings.agent_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        print(process.stdout, end="")
        if process.returncode != 0:
            raise RuntimeError(
                f"Tilesmith agent exited with status {process.returncode}"
            )
        return kernel_path.read_text(encoding="utf-8")


# _evaluate_cli forwards the candidate selected by the Go subprocess to its parent CUDA process.
def _evaluate_cli(url, token):
    kernel_path = os.environ.get("TILESMITH_KERNEL")
    if not kernel_path:
        raise RuntimeError("TILESMITH_KERNEL is required")
    source = Path(kernel_path).read_bytes()
    request = urllib.request.Request(
        url,
        data=source,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/x-python"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        sys.stdout.write(response.read().decode())


# _main reserves a private subprocess mode for the Go evaluator command.
def _main():
    if len(sys.argv) == 4 and sys.argv[1] == "_evaluate":
        _evaluate_cli(sys.argv[2], sys.argv[3])
        return
    raise SystemExit(
        "tilesmith.py is imported as a library; only _evaluate is a CLI command"
    )


if __name__ == "__main__":
    _main()
