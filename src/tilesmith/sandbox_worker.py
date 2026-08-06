"""Run Tilesmith correctness checks and CUDA benchmarks inside a sandbox."""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .measurement import (
        assert_candidate_matches,
        assert_tensor_aliases_match,
        assert_tensor_graph_matches,
        benchmark_pair,
        chip_matches,
        clone_tensor_graph,
    )
except ImportError:
    # The GPU sandbox runs this copied source as worker.py beside measurement.py.
    from measurement import (
        assert_candidate_matches,
        assert_tensor_aliases_match,
        assert_tensor_graph_matches,
        benchmark_pair,
        chip_matches,
        clone_tensor_graph,
    )


ROOT = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")


# _load_generated attaches serialized Inductor constants to one generated wrapper.
def _load_generated(torch, path, constants):
    from torch._inductor.codecache import PyCodeCache, code_hash

    source = path.read_text(encoding="utf-8")
    module = PyCodeCache.load_by_key_path(code_hash(source), str(path), attrs=constants)
    return module.call


# _to_cuda restores recursively serialized inputs and constants on the sandbox GPU.
def _to_cuda(torch, value):
    """Move one serialized tensor graph to CUDA while preserving storage aliases."""
    return clone_tensor_graph(torch, value, map_location="cuda")


# _metadata records the exact software and NVIDIA device used for each result.
def _metadata(torch, triton):
    capability = torch.cuda.get_device_capability()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "cache_policy": "warm",
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "cuda": str(torch.version.cuda),
        "driver": driver,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
    }


# _request reads the host-owned benchmark policy and expected runtime versions.
def _request():
    value = json.loads((ROOT / "request.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("request.json must contain an object")
    return value


# _validate_environment rejects a sandbox that cannot execute the generated source faithfully.
def _validate_environment(metadata, request):
    requested_chip = str(request["chip"])
    if not chip_matches(requested_chip, metadata["gpu"]):
        raise RuntimeError(f"requested {requested_chip}, sandbox provisioned {metadata['gpu']}")
    for package in ("torch", "triton"):
        expected = request.get(package)
        if expected is not None and str(expected) != metadata[package]:
            raise RuntimeError(f"{package} version mismatch: host {expected}, sandbox {metadata[package]}")


# _probe reports capabilities before the agent receives the sandbox as its target.
def _probe(torch, triton):
    request = _request()
    metadata = _metadata(torch, triton)
    _validate_environment(metadata, request)
    return metadata


# _compare validates and benchmarks one real Inductor replacement candidate.
def _compare(torch, triton):
    request = _request()
    metadata = _metadata(torch, triton)
    _validate_environment(metadata, request)
    serialized = torch.load(ROOT / "payload.pt", map_location="cpu", weights_only=False)
    constants = _to_cuda(torch, serialized["constants"])
    inputs = _to_cuda(torch, serialized["inputs"])
    baseline_constants = clone_tensor_graph(torch, constants)
    candidate_constants = clone_tensor_graph(torch, constants)
    constant_snapshot = clone_tensor_graph(torch, candidate_constants)
    baseline = _load_generated(torch, ROOT / "baseline.py", baseline_constants)
    candidate = _load_generated(
        torch,
        ROOT / "candidate.py",
        candidate_constants,
    )
    assert_candidate_matches(
        torch,
        baseline,
        candidate,
        inputs,
        reference_constants=baseline_constants,
        candidate_constants=candidate_constants,
    )
    torch.cuda.synchronize()

    metadata.update(
        benchmark_pair(
            torch,
            baseline,
            candidate,
            inputs,
            "torch_inductor",
            int(request["warmup"]),
            int(request["iterations"]),
            int(request["samples"]),
        )
    )
    assert_candidate_matches(
        torch,
        baseline,
        candidate,
        inputs,
        reference_constants=baseline_constants,
        candidate_constants=candidate_constants,
    )
    assert_tensor_graph_matches(
        torch,
        candidate_constants,
        constant_snapshot,
        "candidate constants",
    )
    assert_tensor_aliases_match(
        torch,
        candidate_constants,
        constant_snapshot,
        "candidate constants",
    )
    metadata["reference"] = "torch_inductor_generated"
    return metadata


# _run_generated invokes one generated candidate for an agent-selected measurement tool.
def _run_generated(torch, triton):
    serialized = torch.load(ROOT / "payload.pt", map_location="cpu", weights_only=False)
    constants = _to_cuda(torch, serialized["constants"])
    inputs = _to_cuda(torch, serialized["inputs"])
    candidate = _load_generated(torch, ROOT / "candidate.py", constants)
    with torch.no_grad():
        candidate(list(inputs))
    torch.cuda.synchronize()
    return {"ran": True}


# _write_result atomically replaces stale evidence only after encoding succeeds.
def _write_result(result):
    """Publish one complete JSON result without exposing a partial destination."""
    encoded = json.dumps(result, sort_keys=True)
    temporary = ROOT / f".result-{os.getpid()}.json"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, ROOT / "result.json")
    return len(encoded)


# main dispatches the single requested worker action and emits one JSON report.
def main():
    """Run exactly one known worker action and atomically publish its result."""
    import torch
    import triton

    actions = {
        "probe": _probe,
        "compare": _compare,
        "run-generated": _run_generated,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {'|'.join(actions)}")
    result = actions[sys.argv[1]](torch, triton)
    # Large traces travel through the file endpoint instead of truncation-prone command output.
    result_bytes = _write_result(result)
    print(json.dumps({"result_bytes": result_bytes}, sort_keys=True))


if __name__ == "__main__":
    main()
