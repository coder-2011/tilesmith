"""Run Tilesmith correctness checks and CUDA benchmarks inside a sandbox."""

import importlib.util
import gzip
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


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
    return tuple(
        value.clone(memory_format=torch.preserve_format)
        if torch.is_tensor(value)
        else value
        for value in inputs
    )


# _to_cuda restores recursively serialized inputs and constants on the sandbox GPU.
def _to_cuda(torch, value):
    if torch.is_tensor(value):
        return value.to("cuda", memory_format=torch.preserve_format)
    if isinstance(value, tuple):
        return tuple(_to_cuda(torch, item) for item in value)
    if isinstance(value, list):
        return [_to_cuda(torch, item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cuda(torch, item) for key, item in value.items()}
    return value


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
        "ncu": shutil.which("ncu"),
        "nvidia_smi_q_xml": subprocess.check_output(
            ["nvidia-smi", "-q", "-x"],
            text=True,
        ),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
    }


# _profile_exports returns the complete Kineto trace plus readable aggregate and memory views.
def _profile_exports(profiler, prefix):
    trace_path = ROOT / f"{prefix}-trace.json"
    memory_path = ROOT / f"{prefix}-memory.raw.json.gz"
    cpu_stacks_path = ROOT / f"{prefix}-cpu-stacks.txt"
    device_stacks_path = ROOT / f"{prefix}-device-stacks.txt"
    profiler.export_chrome_trace(str(trace_path))

    try:
        averages = profiler.key_averages(
            group_by_input_shape=True,
            group_by_stack_n=8,
            group_by_overload_name=True,
        )
    except TypeError:
        # Older supported PyTorch builds do not expose overload grouping.
        averages = profiler.key_averages(
            group_by_input_shape=True,
            group_by_stack_n=8,
        )
    tables = {
        "self_cpu_time_total": averages.table(
            sort_by="self_cpu_time_total",
            row_limit=-1,
        )
    }
    for metric in ("self_device_time_total", "self_cuda_time_total"):
        try:
            tables[metric] = averages.table(sort_by=metric, row_limit=-1)
            break
        except (AttributeError, KeyError, RuntimeError):
            continue

    exports = {
        "chrome_trace": json.loads(trace_path.read_text(encoding="utf-8")),
        "events": len(profiler.events()),
        "key_averages": tables,
    }
    try:
        profiler.export_memory_timeline(str(memory_path))
        with gzip.open(memory_path, "rt", encoding="utf-8") as memory_file:
            exports["memory_timeline"] = json.load(memory_file)
    except Exception as error:
        exports["memory_timeline_error"] = str(error)

    for name, path, metric in (
        ("cpu_stacks", cpu_stacks_path, "self_cpu_time_total"),
        ("device_stacks", device_stacks_path, "self_cuda_time_total"),
    ):
        try:
            profiler.export_stacks(str(path), metric)
            exports[name] = path.read_text(encoding="utf-8")
        except Exception as error:
            exports[f"{name}_error"] = str(error)
    return exports


# _profile_cuda captures every PyTorch/Kineto trace and allocator view available in-process.
def _profile_cuda(torch, function, inputs, prefix):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.reset_peak_memory_stats()
    memory_before = dict(torch.cuda.memory_stats())
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("tilesmith_candidate"):
            function(list(inputs))
        torch.cuda.synchronize()

    try:
        memory_snapshot = torch.cuda.memory_snapshot()
    except Exception as error:
        memory_snapshot = {"error": str(error)}

    return {
        "activities": ["CPU", "CUDA"],
        "allocator": {
            "memory_snapshot": memory_snapshot,
            "stats_after": dict(torch.cuda.memory_stats()),
            "stats_before": memory_before,
        },
        "config": {
            "profile_memory": True,
            "record_shapes": True,
            "with_flops": True,
            "with_modules": True,
            "with_stack": True,
        },
        "kineto": _profile_exports(profiler, prefix),
        "peak_memory_allocated": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved": torch.cuda.max_memory_reserved(),
    }


# _request reads the host-owned benchmark policy and expected runtime versions.
def _request():
    value = json.loads((ROOT / "request.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("request.json must contain an object")
    return value


# _validate_environment rejects a sandbox that cannot execute the generated source faithfully.
def _validate_environment(metadata, request):
    requested_chip = str(request["chip"]).rstrip("!")
    if requested_chip.lower() not in metadata["gpu"].lower():
        raise RuntimeError(
            f"requested {requested_chip}, sandbox provisioned {metadata['gpu']}"
        )
    for package in ("torch", "triton"):
        expected = request.get(package)
        if expected is not None and str(expected) != metadata[package]:
            raise RuntimeError(
                f"{package} version mismatch: host {expected}, "
                f"sandbox {metadata[package]}"
            )


# _check compares the controlled generated wrapper against eager PyTorch.
def _check(torch, workload, kernel):
    if hasattr(workload, "make_cases"):
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


# _generated_inputs changes values without changing compiled shapes, dtypes, or strides.
def _generated_inputs(torch, inputs, variant):
    values = []
    for value in inputs:
        if not torch.is_tensor(value):
            values.append(value)
            continue
        cloned = value.detach().clone(memory_format=torch.preserve_format)
        if variant == "zero" and (cloned.is_floating_point() or cloned.is_complex()):
            cloned.zero_()
        if variant == "high" and (cloned.is_floating_point() or cloned.is_complex()):
            cloned.mul_(4)
        values.append(cloned)
    return tuple(values)


# _check_generated compares candidate source with PyTorch's unmodified Inductor wrapper.
def _check_generated(torch, baseline, candidate, inputs):
    with torch.no_grad():
        for variant in ("normal", "zero", "high"):
            cpu_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all()
            expected = baseline(list(_generated_inputs(torch, inputs, variant)))
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)
            actual = candidate(list(_generated_inputs(torch, inputs, variant)))
            torch.testing.assert_close(
                actual,
                expected,
                equal_nan=True,
                check_stride=True,
            )
    torch.cuda.synchronize()


# _benchmark_pair alternates execution order and reports paired CUDA-event ratios.
def _benchmark_pair(
    torch,
    reference,
    candidate,
    inputs,
    reference_name,
    warmup,
    iterations,
    sample_count,
):
    functions = ((reference_name, reference), ("candidate_triton", candidate))
    with torch.no_grad():
        for _ in range(warmup):
            for _, measured in functions:
                measured(inputs)
        torch.cuda.synchronize()

        samples = {name: [] for name, _ in functions}
        for sample in range(sample_count):
            order = functions if sample % 2 == 0 else tuple(reversed(functions))
            for name, measured in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    measured(inputs)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / iterations)

    ratios = [
        candidate_ms / reference_ms
        for reference_ms, candidate_ms in zip(
            samples[reference_name], samples["candidate_triton"]
        )
    ]
    return {
        "benchmark_iterations": iterations,
        "benchmark_samples": sample_count,
        "candidate_ms": statistics.median(samples["candidate_triton"]),
        "candidate_samples_ms": samples["candidate_triton"],
        f"{reference_name}_ms": statistics.median(samples[reference_name]),
        f"{reference_name}_samples_ms": samples[reference_name],
        "score": statistics.median(ratios),
        "warmup": warmup,
    }


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

    # candidate adapts Inductor's boxed convention to the paired timer.
    def candidate(values):
        return kernel.call(list(values))

    return _benchmark_pair(
        torch,
        reference,
        candidate,
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
    baseline = _load_generated(torch, ROOT / "baseline.py", constants)
    candidate = _load_generated(torch, ROOT / "candidate.py", constants)
    _check_generated(torch, baseline, candidate, inputs)

    # reference and measured preserve the boxed generated-wrapper convention.
    def reference(values):
        return baseline(list(values))

    def measured(values):
        return candidate(list(values))

    metadata.update(
        _benchmark_pair(
            torch,
            reference,
            measured,
            inputs,
            "torch_inductor",
            int(request["warmup"]),
            int(request["iterations"]),
            int(request["samples"]),
        )
    )
    metadata["profiling"] = _profile_cuda(
        torch,
        measured,
        inputs,
        "generated-candidate",
    )
    metadata["profiling_progress"] = [
        "correctness_complete",
        "timing_complete",
        "torch_profile_complete",
    ]
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
    inputs = workload.make_inputs("cuda")

    # measured adapts the generated fixture's boxed convention for profiling.
    def measured(values):
        return candidate.call(list(values))

    metadata.update(_benchmark_fixture(torch, workload, candidate, request))
    metadata["profiling"] = _profile_cuda(
        torch,
        measured,
        inputs,
        "fixture-candidate",
    )
    metadata["profiling_progress"] = [
        "correctness_complete",
        "timing_complete",
        "torch_profile_complete",
    ]
    return metadata


# _profile_generated runs one warmed generated candidate for an external Nsight Compute launch.
def _profile_generated(torch, triton):
    serialized = torch.load(ROOT / "payload.pt", map_location="cpu", weights_only=False)
    constants = _to_cuda(torch, serialized["constants"])
    inputs = _to_cuda(torch, serialized["inputs"])
    candidate = _load_generated(torch, ROOT / "candidate.py", constants)
    with torch.no_grad():
        candidate(list(inputs))
    torch.cuda.synchronize()
    return {"profiled": True}


# _profile_fixture runs one warmed fixture candidate for an external Nsight Compute launch.
def _profile_fixture(torch, triton):
    workload = _load_module("tilesmith_workload", ROOT / "workload.py")
    candidate = _load_module("tilesmith_candidate", ROOT / "candidate.py")
    with torch.no_grad():
        candidate.call(list(workload.make_inputs("cuda")))
    torch.cuda.synchronize()
    return {"profiled": True}


# main dispatches the single requested worker action and emits one JSON report.
def main():
    import torch
    import triton

    actions = {
        "probe": _probe,
        "compare": _compare,
        "extract-fixture": _extract_fixture,
        "evaluate-fixture": _evaluate_fixture,
        "profile-generated": _profile_generated,
        "profile-fixture": _profile_fixture,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {'|'.join(actions)}")
    result = actions[sys.argv[1]](torch, triton)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
