import builtins
import contextlib
import fcntl
import hashlib
import http.server
import inspect
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
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WARMUP = 25
BENCH_ITERATIONS = 100
BENCH_SAMPLES = 7
MAX_SOURCE_BYTES = 64 * 1024 * 1024
OPTIMIZE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Settings:
    agent_root: Path
    cache_dir: Path
    config: str
    iterations: int
    min_improvement: float
    model: str
    skills: tuple[str, ...]
    timeout: str


# compile returns a normal torch.compile callable while routing Inductor graphs through Tilesmith.
def compile(model, *, example_inputs):
    import torch
    from torch._inductor.compile_fx import compile_fx

    if not callable(model):
        raise TypeError("model must be callable")
    if not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be a tuple of positional model arguments")
    leaves = torch.utils._pytree.tree_leaves(example_inputs)
    if not any(
        torch.is_tensor(value) and value.device.type == "cuda" for value in leaves
    ):
        raise ValueError("example_inputs must contain at least one CUDA tensor")
    if "inner_compile" not in inspect.signature(compile_fx).parameters:
        raise RuntimeError(
            "this PyTorch version does not expose Inductor's inner_compile hook"
        )

    settings = _read_settings()
    return torch.compile(model, backend=_Backend(settings))


# _read_settings keeps the public API small while making expensive agent choices explicit in the environment.
def _read_settings():
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
        config=os.environ.get("CLIPROXY_CONFIG", ""),
        iterations=iterations,
        min_improvement=min_improvement,
        model=os.environ.get("TILESMITH_MODEL", "gpt-5.4"),
        skills=tuple(_default_skills()),
        timeout=os.environ.get("TILESMITH_TIMEOUT", "20m"),
    )


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
        self.source = source
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
        cache_key = _cache_key(self.source, self.settings, device)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

        with _cache_lock(self.settings.cache_dir / f"{cache_key}.lock"):
            cached = _read_cache(self.settings.cache_dir, cache_key)
            if cached is None:
                with OPTIMIZE_LOCK, _preserve_rng():
                    evaluator = _Evaluator(
                        self.baseline, self.constants, inputs, device
                    )
                    optimized = _run_agent(self.source, self.settings, evaluator)
                cached = _write_cache(self.settings.cache_dir, cache_key, optimized)
        if cached.read_text(encoding="utf-8") == self.source:
            return self.baseline
        return _load_call(cached, self.constants)


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
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "warmup": WARMUP,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    digest.update(source.encode())
    for path in [
        Path(__file__),
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


class _Evaluator:
    # __init__ snapshots real inputs before either boxed callable can clear its argument list.
    def __init__(self, baseline, constants, inputs, device):
        self.baseline = baseline
        self.constants = constants
        self.device = device
        self.inputs = tuple(inputs)
        self.directory = tempfile.TemporaryDirectory(prefix="tilesmith-evaluator-")
        self.lock = threading.Lock()

    # evaluate rejects incorrect source before reporting its median CUDA-event runtime in milliseconds.
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
            return self._benchmark(candidate)

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

    # _benchmark excludes compilation and warmup, then returns the median of independent CUDA-event samples.
    def _benchmark(self, function):
        import torch

        inputs = self._inputs("normal")
        for _ in range(WARMUP):
            function(list(inputs))
        torch.cuda.synchronize(self.device)

        samples = []
        for _ in range(BENCH_SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(BENCH_ITERATIONS):
                function(list(inputs))
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / BENCH_ITERATIONS)
        return statistics.median(samples)


# _evaluator_server exposes the in-process CUDA evaluator only to a token-authenticated localhost command.
@contextlib.contextmanager
def _evaluator_server(evaluator):
    token = secrets.token_urlsafe(32)

    class Handler(http.server.BaseHTTPRequestHandler):
        # do_POST evaluates raw candidate source and returns one lower-is-better score.
        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_SOURCE_BYTES:
                self.send_error(413)
                return
            try:
                score = evaluator.evaluate(self.rfile.read(length).decode())
                response = f"{score:.9g}\n".encode()
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
