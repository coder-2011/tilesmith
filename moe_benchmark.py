"""Run a public Tilesmith MoE comparison on one requested NVIDIA GPU."""

import argparse
import asyncio
import json
import math
import os
import shlex
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/tmp/tilesmith-moe"


# _require_success turns one remote process failure into a bounded host error.
def _require_success(result, operation):
    """Raise with bounded diagnostics when a remote command fails."""
    if result.exit_code == 0:
        return
    details = (result.stderr or result.stdout).strip()[-4_000:]
    raise RuntimeError(
        f"{operation} failed with exit code {result.exit_code}: {details}"
    )


# _agent_environment forwards only the model credential needed by the remote process.
def _agent_environment(model):
    """Return the secret-safe environment used by the remote Tilesmith agent."""
    from tilesmith import _model_connection, _wait_for_model

    _wait_for_model(model)
    api_key, base_url = _model_connection()
    return {
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url or "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": REMOTE_ROOT,
        "TILESMITH_CACHE_DIR": f"{REMOTE_ROOT}/cache",
        "TILESMITH_MODEL": model,
        "TILESMITH_TIMEOUT": "1200",
    }


# _skill_files maps the local optimization guidance into Tilesmith's remote lookup paths.
def _skill_files():
    """Return available trusted skill files and their expected remote paths."""
    from tilesmith import _default_skills

    home = Path.home()
    remote_home = Path("/root")
    files = {}
    for configured in _default_skills():
        source = Path(configured)
        relative = source.relative_to(home)
        destination = remote_home / relative
        files[source] = str(destination)
    return files


# _upload_runtime transfers only the files the remote public API actually imports.
async def _upload_runtime(sandbox):
    """Upload Tilesmith, the benchmark, and trusted agent context."""
    skill_files = _skill_files()
    directories = {
        str(Path(destination).parent) for destination in skill_files.values()
    }
    directories.add(REMOTE_ROOT)
    directories.add(f"{REMOTE_ROOT}/tilesmith/sandboxes")
    command = "mkdir -p " + " ".join(sorted(directories))
    _require_success(
        await sandbox.run_command(command, timeout_ms=60_000),
        "remote directory setup",
    )
    package_root = ROOT / "src" / "tilesmith"
    remote_package = f"{REMOTE_ROOT}/tilesmith"
    for source, destination in (
        (ROOT / "moe_benchmark.py", f"{REMOTE_ROOT}/moe_benchmark.py"),
        (ROOT / "requirements.txt", f"{REMOTE_ROOT}/requirements.txt"),
        (package_root / "__init__.py", f"{remote_package}/__init__.py"),
        (package_root / "measurement.py", f"{remote_package}/measurement.py"),
        (package_root / "sandbox_worker.py", f"{remote_package}/sandbox_worker.py"),
        (
            package_root / "sandboxes" / "__init__.py",
            f"{remote_package}/sandboxes/__init__.py",
        ),
        (
            package_root / "sandboxes" / "_computesdk_bridge.cjs",
            f"{remote_package}/sandboxes/_computesdk_bridge.cjs",
        ),
        (
            package_root / "sandboxes" / "computesdk.py",
            f"{remote_package}/sandboxes/computesdk.py",
        ),
    ):
        await sandbox.write_file(destination, source.read_text(encoding="utf-8"))
    for source, destination in skill_files.items():
        await sandbox.write_file(destination, source.read_text(encoding="utf-8"))


# _run_worker streams a background process through ordinary sandbox files.
async def _run_worker(sandbox, environment):
    """Run the GPU worker while polling complete stdout and stderr traces."""
    stdout_path = f"{REMOTE_ROOT}/worker.stdout"
    stderr_path = f"{REMOTE_ROOT}/worker.stderr"
    exit_path = f"{REMOTE_ROOT}/worker.exit"
    worker = shlex.join(["python3", f"{REMOTE_ROOT}/moe_benchmark.py", "--worker"])
    script = (
        f"{worker} > {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}; "
        f"printf '%s' $? > {shlex.quote(exit_path)}"
    )
    command = f"nohup {shlex.join(['sh', '-lc', script])} > /dev/null 2>&1 &"
    launcher = await sandbox.run_command(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        timeout_ms=60_000,
    )
    _require_success(launcher, "remote MoE worker launch")

    offsets = {stdout_path: 0, stderr_path: 0}
    streams = {stdout_path: sys.stdout, stderr_path: sys.stderr}
    deadline = asyncio.get_running_loop().time() + 2_400
    while True:
        # File polling avoids ComputeSDK's daemon streaming path and retains full logs.
        for path, stream in streams.items():
            if not await sandbox.exists(path):
                continue
            content = await sandbox.read_file(path)
            offset = offsets[path]
            if len(content) > offset:
                print(content[offset:], end="", file=stream, flush=True)
                offsets[path] = len(content)
        if await sandbox.exists(exit_path):
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("remote MoE benchmark exceeded 2400 seconds")
        await asyncio.sleep(2)

    exit_code = int((await sandbox.read_file(exit_path)).strip())
    if exit_code != 0:
        stderr = await sandbox.read_file(stderr_path) if await sandbox.exists(stderr_path) else ""
        raise RuntimeError(
            f"remote MoE benchmark failed with exit code {exit_code}: {stderr[-4_000:]}"
        )


# _run_remote owns the paid GPU lifecycle and streams progress to the caller.
async def _run_remote(model, chip):
    """Provision one requested GPU, run the worker, and always tear it down."""
    from tilesmith.measurement import chip_matches
    from tilesmith.sandboxes.computesdk import ComputeSDKClient, ProviderSpec
    from tilesmith import _read_compute_config

    config = _read_compute_config(chip)
    agent_environment = _agent_environment(model)
    provider = ProviderSpec(
        package=config.provider_package,
        factory=config.provider_factory,
        version=config.provider_version,
        config=config.provider_config,
    )
    options = {
        "gpu": config.chip,
        "timeout": config.sandbox_timeout * 1_000,
    }
    if config.template is not None:
        options["templateId"] = config.template
    async with ComputeSDKClient([provider]) as client:
        readiness_timeout = config.readiness_timeout_ms
        async with client.ephemeral_sandbox(
            options,
            provider=config.provider,
            request_timeout=readiness_timeout / 1_000,
        ) as sandbox:
            print(
                f"waiting for a command-ready {config.chip} sandbox...",
                flush=True,
            )
            await sandbox.wait_until_ready(readiness_timeout)
            await _upload_runtime(sandbox)
            capability = await sandbox.run_command(
                "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader && df -h /",
                timeout_ms=60_000,
            )
            _require_success(capability, "GPU capability check")
            if not chip_matches(config.chip, capability.stdout):
                raise RuntimeError(
                    f"requested {config.chip}, sandbox reported {capability.stdout.strip()}"
                )
            print(capability.stdout, end="", flush=True)
            print("installing the remote PyTorch and agent runtime...", flush=True)
            install = await sandbox.run_command(
                f"uv pip install --system torch numpy -r {REMOTE_ROOT}/requirements.txt",
                timeout_ms=1_200_000,
            )
            _require_success(install, "remote dependency install")
            print(install.stdout, end="", flush=True)
            print(install.stderr, end="", file=sys.stderr, flush=True)
            print("running eager, torch.compile, and tilesmith.compile...", flush=True)
            await _run_worker(sandbox, agent_environment)


# _worker builds and compares eager, Inductor, and Tilesmith on the active CUDA GPU.
def _worker():
    """Execute the deterministic MoE comparison on the provisioned GPU."""
    import copy

    import torch
    from torch import nn
    from torch.nn import functional as functional

    import tilesmith

    class Expert(nn.Module):
        """Apply one gated feed-forward expert to every token."""

        # __init__ creates the two projection matrices owned by one expert.
        def __init__(self, width, hidden):
            super().__init__()
            self.up = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)

        # forward applies the expert's GELU feed-forward transformation.
        def forward(self, inputs):
            return self.down(functional.gelu(self.up(inputs)))

    class TopTwoMoE(nn.Module):
        """Route each token to its two highest-probability dense experts."""

        # __init__ creates the router and independently parameterized experts.
        def __init__(self, width, hidden, expert_count):
            super().__init__()
            self.router = nn.Linear(width, expert_count, bias=False)
            self.experts = nn.ModuleList(
                Expert(width, hidden) for _ in range(expert_count)
            )
            self.expert_count = expert_count

        # forward mixes the top-two expert outputs with normalized router weights.
        def forward(self, inputs):
            router = functional.softmax(self.router(inputs).float(), dim=-1)
            values, indices = torch.topk(router, k=2, dim=-1)
            selected = functional.one_hot(indices, self.expert_count)
            weights = (selected * values.unsqueeze(-1)).sum(dim=-2)
            expert_outputs = torch.stack(
                [expert(inputs) for expert in self.experts],
                dim=-2,
            )
            return (expert_outputs * weights.to(inputs.dtype).unsqueeze(-1)).sum(
                dim=-2
            )

    class MoETransformer(nn.Module):
        """Combine causal self-attention and top-two MoE token mixing."""

        # __init__ creates one compact inference-only Transformer block.
        def __init__(self, vocabulary, width, heads, expert_count, expert_hidden):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, width)
            self.attention_norm = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width, bias=False)
            self.attention_out = nn.Linear(width, width, bias=False)
            self.moe_norm = nn.LayerNorm(width)
            self.moe = TopTwoMoE(width, expert_hidden, expert_count)
            self.output_norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, vocabulary, bias=False)
            self.heads = heads

        # forward returns deterministic token logits for the entire sequence.
        def forward(self, token_ids):
            hidden = self.embedding(token_ids)
            batch, sequence, width = hidden.shape
            head_width = width // self.heads
            qkv = self.qkv(self.attention_norm(hidden))
            qkv = qkv.view(batch, sequence, 3, self.heads, head_width)
            query, key, value = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores.float() / math.sqrt(head_width)
            causal = torch.ones(
                sequence,
                sequence,
                device=hidden.device,
                dtype=torch.bool,
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            probabilities = functional.softmax(scores, dim=-1).to(hidden.dtype)
            attended = torch.matmul(probabilities, value)
            attended = attended.transpose(1, 2).reshape(batch, sequence, width)
            hidden = hidden + self.attention_out(attended)
            hidden = hidden + self.moe(self.moe_norm(hidden))
            return self.lm_head(self.output_norm(hidden))

    # benchmark interleaves variants so cache and clock drift affect them symmetrically.
    def benchmark(functions, inputs, warmup=25, iterations=100, samples=7):
        """Return median CUDA-event latency and all interleaved samples."""
        with torch.no_grad():
            for _ in range(warmup):
                for function in functions.values():
                    function(inputs)
            torch.cuda.synchronize()
            timings = {name: [] for name in functions}
            items = tuple(functions.items())
            for sample in range(samples):
                order = items if sample % 2 == 0 else tuple(reversed(items))
                for name, function in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        function(inputs)
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) / iterations)
        return {
            name: {
                "median_ms": statistics.median(samples_ms),
                "samples_ms": samples_ms,
            }
            for name, samples_ms in timings.items()
        }

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda:0")
    model = MoETransformer(
        vocabulary=2048,
        width=256,
        heads=8,
        expert_count=4,
        expert_hidden=512,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    tokens = torch.randint(0, 2048, (8, 128), device=device)

    eager = copy.deepcopy(model)
    inductor = torch.compile(copy.deepcopy(model))
    optimized = tilesmith.compile(
        copy.deepcopy(model),
        example_inputs=(tokens,),
        loops=1,
    )
    with torch.no_grad():
        eager_output = eager(tokens)
        inductor_output = inductor(tokens)
        optimized_output = optimized(tokens)
    torch.testing.assert_close(
        inductor_output,
        eager_output,
        atol=1.5e-1,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        optimized_output,
        eager_output,
        atol=1.5e-1,
        rtol=2e-2,
    )
    timings = benchmark(
        {
            "eager_pytorch": eager,
            "torch_compile": inductor,
            "tilesmith_compile": optimized,
        },
        tokens,
    )
    inductor_ms = timings["torch_compile"]["median_ms"]
    tilesmith_ms = timings["tilesmith_compile"]["median_ms"]
    result = {
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "input_shape": list(tokens.shape),
        "max_abs_error_torch_compile": float(
            (inductor_output - eager_output).abs().max()
        ),
        "max_abs_error_tilesmith_compile": float(
            (optimized_output - eager_output).abs().max()
        ),
        "output_shape": list(eager_output.shape),
        "torch": torch.__version__,
        "timings": timings,
    }
    print("TILESMITH_MOE_RESULT=" + json.dumps(result, sort_keys=True))
    if tilesmith_ms >= inductor_ms:
        raise RuntimeError(
            f"Tilesmith did not beat torch.compile: {tilesmith_ms:.6f} ms "
            f">= {inductor_ms:.6f} ms"
        )


# _parse_args distinguishes the local provisioner from its uploaded worker mode.
def _parse_args():
    """Parse the explicit paid-run and internal worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--model",
        default=os.environ.get("TILESMITH_E2E_MODEL", "openai/gpt-5.5"),
        help="OpenAI-compatible model used by the Tilesmith agent",
    )
    parser.add_argument(
        "--chip",
        default=os.environ.get("TILESMITH_E2E_CHIP", "L4"),
        help="ComputeSDK GPU requested for the end-to-end proof",
    )
    return parser.parse_args()


# main runs either the local paid provisioner or the remote GPU worker.
def main():
    """Dispatch the selected side of the GPU benchmark."""
    arguments = _parse_args()
    if arguments.worker:
        _worker()
    else:
        asyncio.run(_run_remote(arguments.model, arguments.chip))


if __name__ == "__main__":
    main()
