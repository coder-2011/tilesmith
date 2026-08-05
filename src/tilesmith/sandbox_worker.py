"""Run Tilesmith correctness checks and CUDA benchmarks inside a sandbox."""

import importlib.util
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


# _load_module imports one self-contained Python file under a unique module name.
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# _load_generated attaches serialized Inductor constants to one generated wrapper.
def _load_generated(torch, path, constants):
    from torch._inductor.codecache import PyCodeCache, code_hash

    source = path.read_text(encoding="utf-8")
    module = PyCodeCache.load_by_key_path(code_hash(source), str(path), attrs=constants)
    return module.call


# _clone_inputs keeps reference and candidate executions from sharing mutable tensors.
def _clone_inputs(torch, inputs):
    """Clone one input graph without losing repeated references or tensor views."""
    return clone_tensor_graph(torch, tuple(inputs))


# _to_cuda restores recursively serialized inputs and constants on the sandbox GPU.
def _to_cuda(torch, value):
    """Move one serialized tensor graph to CUDA while preserving storage aliases."""
    return clone_tensor_graph(torch, value, map_location="cuda")


# _outputs normalizes a single tensor and a tuple of tensors for comparison.
def _outputs(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


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


# _check compares the controlled generated wrapper against eager PyTorch.
def _check(torch, workload, kernel):
    if hasattr(workload, "make_cases"):  # noqa: SIM108 - keep the optional workload protocol readable.
        cases = workload.make_cases("cuda")
    else:
        cases = [workload.make_inputs("cuda")]

    atol = getattr(workload, "ATOL", 1e-5)
    rtol = getattr(workload, "RTOL", 1e-4)
    case_count = 0
    with torch.no_grad():
        for inputs in cases:
            expected = workload.model(*_clone_inputs(torch, inputs))
            actual = kernel.call(list(_clone_inputs(torch, inputs)))
            torch.testing.assert_close(
                _outputs(actual),
                _outputs(expected),
                atol=atol,
                check_stride=True,
                rtol=rtol,
                equal_nan=True,
            )
            case_count += 1
    torch.cuda.synchronize()
    return case_count


# _extract captures and validates TorchInductor's generated wrapper for the paid fixture.
def _extract(torch, workload):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    torch._dynamo.reset()
    config.fx_graph_cache = False
    sources = []

    # inner_compile records the exact wrapper that the returned callable executes.
    def inner_compile(graph_module, example_inputs, **kwargs):
        compiled = compile_fx_inner(graph_module, example_inputs, **kwargs)
        sources.append(compiled.source_code)
        return compiled

    # backend preserves Inductor's outer compilation while inserting the capture hook.
    def backend(graph_module, example_inputs):
        return compile_fx(
            graph_module,
            example_inputs,
            inner_compile=inner_compile,
            config_patches={"triton.cudagraphs": False},
        )

    inputs = workload.make_inputs("cuda")
    with torch.no_grad():
        expected = workload.model(*_clone_inputs(torch, inputs))
        compiled = torch.compile(workload.model, backend=backend, fullgraph=True)
        actual = compiled(*_clone_inputs(torch, inputs))
        torch.testing.assert_close(
            _outputs(actual),
            _outputs(expected),
            atol=getattr(workload, "ATOL", 1e-5),
            rtol=getattr(workload, "RTOL", 1e-4),
            equal_nan=True,
        )
    torch.cuda.synchronize()

    if len(sources) != 1:
        raise RuntimeError(f"expected one generated wrapper, found {len(sources)}")
    first_line, separator, remainder = sources[0].partition("\n")
    if first_line.startswith("# AOT ID:"):
        return remainder if separator else ""
    return sources[0]


# _benchmark_fixture compares eager PyTorch with the controlled generated wrapper.
def _benchmark_fixture(torch, workload, kernel, request):
    inputs = workload.make_inputs("cuda")

    # reference adapts the eager positional convention to the paired timer.
    def reference(values):
        return workload.model(*values)

    return benchmark_pair(
        torch,
        reference,
        kernel.call,
        inputs,
        "pytorch",
        int(request["warmup"]),
        int(request["iterations"]),
        int(request["samples"]),
    )


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
    assert_candidate_matches(torch, baseline, candidate, inputs)
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
    assert_candidate_matches(torch, baseline, candidate, inputs)
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


# _extract_fixture prepares the controlled obvious-improvement benchmark source.
def _extract_fixture(torch, triton):
    request = _request()
    metadata = _metadata(torch, triton)
    _validate_environment(metadata, request)
    workload = _load_module("tilesmith_workload", ROOT / "workload.py")
    metadata["source"] = _extract(torch, workload)
    metadata["shape"] = list(workload.SHAPE)
    metadata["dtype"] = workload.DTYPE
    return metadata


# _evaluate_fixture checks and times one controlled benchmark candidate.
def _evaluate_fixture(torch, triton):
    request = _request()
    metadata = _metadata(torch, triton)
    _validate_environment(metadata, request)
    workload = _load_module("tilesmith_workload", ROOT / "workload.py")
    candidate = _load_module("tilesmith_candidate", ROOT / "candidate.py")
    metadata["cases"] = _check(torch, workload, candidate)

    metadata.update(_benchmark_fixture(torch, workload, candidate, request))
    _check(torch, workload, candidate)
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


# _run_fixture invokes one fixture candidate for an agent-selected measurement tool.
def _run_fixture(torch, triton):
    workload = _load_module("tilesmith_workload", ROOT / "workload.py")
    candidate = _load_module("tilesmith_candidate", ROOT / "candidate.py")
    with torch.no_grad():
        candidate.call(list(workload.make_inputs("cuda")))
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
        "extract-fixture": _extract_fixture,
        "evaluate-fixture": _evaluate_fixture,
        "run-generated": _run_generated,
        "run-fixture": _run_fixture,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {'|'.join(actions)}")
    result = actions[sys.argv[1]](torch, triton)
    # Large traces travel through the file endpoint instead of truncation-prone command output.
    result_bytes = _write_result(result)
    print(json.dumps({"result_bytes": result_bytes}, sort_keys=True))


if __name__ == "__main__":
    main()
