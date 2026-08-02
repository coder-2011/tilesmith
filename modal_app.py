import importlib.util
import io
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import modal


app = modal.App("tilesmith")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.13.0")
cache = modal.Volume.from_name("tilesmith-h100-cache", create_if_missing=True)


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
        value.clone(memory_format=torch.preserve_format) if torch.is_tensor(value) else value
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


# _outputs normalizes a single tensor and a tuple of tensors to the same comparison shape.
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
        "clock_policy": "Modal default",
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "cuda": str(torch.version.cuda),
        "driver": driver,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
    }


# _check compares the generated wrapper against eager PyTorch on every workload case.
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
            expected_inputs = _clone_inputs(torch, inputs)
            candidate_inputs = _clone_inputs(torch, inputs)
            expected = workload.model(*expected_inputs)
            actual = kernel.call(list(candidate_inputs))
            torch.testing.assert_close(
                _outputs(actual),
                _outputs(expected),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            case_count += 1
    torch.cuda.synchronize()
    return case_count


# _generated_inputs creates value variants without changing compiled shapes, dtypes, or strides.
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
            torch.testing.assert_close(actual, expected, equal_nan=True)
    torch.cuda.synchronize()


# _benchmark_pair alternates execution order and reports a median of paired CUDA-event ratios.
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


# _extract captures TorchInductor's generated wrapper through its inner compile hook.
def _extract(torch, workload):
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    # Both graph caches can bypass the backend or its inner source-capture hook.
    torch._dynamo.reset()
    config.fx_graph_cache = False
    sources = []

    # inner_compile records the exact wrapper that the returned callable executes.
    def inner_compile(graph_module, example_inputs, **kwargs):
        compiled = compile_fx_inner(graph_module, example_inputs, **kwargs)
        sources.append(compiled.source_code)
        return compiled

    # backend preserves Inductor's normal outer compilation while inserting the capture hook.
    def backend(graph_module, example_inputs):
        return compile_fx(
            graph_module,
            example_inputs,
            inner_compile=inner_compile,
            config_patches={"triton.cudagraphs": False},
        )

    inputs = workload.make_inputs("cuda")
    expected_inputs = _clone_inputs(torch, inputs)
    compiled_inputs = _clone_inputs(torch, inputs)
    with torch.no_grad():
        expected = workload.model(*expected_inputs)
        compiled = torch.compile(workload.model, backend=backend, fullgraph=True)
        actual = compiled(*compiled_inputs)
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
    # PyTorch increments this comment even when the executable wrapper is identical.
    if first_line.startswith("# AOT ID:"):
        return remainder if separator else ""
    return sources[0]


# _benchmark compares eager PyTorch with the custom generated wrapper under identical timing.
def _benchmark(torch, workload, kernel):
    inputs = workload.make_inputs("cuda")
    warmup = getattr(workload, "WARMUP", 25)
    iterations = getattr(workload, "BENCH_ITERATIONS", 100)
    sample_count = getattr(workload, "BENCH_SAMPLES", 7)

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
        warmup,
        iterations,
        sample_count,
    )


# run_gpu extracts, checks, or benchmarks one generated wrapper on an NVIDIA H100.
@app.function(image=image, gpu="H100!", timeout=1200, volumes={"/cache": cache})
def run_gpu(action, workload_source, kernel_source):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/cache/torchinductor"

    import torch
    import triton

    with tempfile.TemporaryDirectory(prefix="tilesmith-") as directory:
        workload_path = Path(directory) / "workload.py"
        workload_path.write_text(workload_source, encoding="utf-8")
        workload = _load_module("tilesmith_workload", workload_path)
        metadata = _metadata(torch, triton)
        if hasattr(workload, "SHAPE"):
            metadata["shape"] = list(workload.SHAPE)
        if hasattr(workload, "DTYPE"):
            metadata["dtype"] = workload.DTYPE

        if action == "extract":
            metadata["source"] = _extract(torch, workload)
            cache.commit()
            return metadata

        kernel_path = Path(directory) / "candidate.py"
        kernel_path.write_text(kernel_source, encoding="utf-8")
        kernel = _load_module("tilesmith_candidate", kernel_path)
        metadata["cases"] = _check(torch, workload, kernel)
        if action in {"bench", "evaluate"}:
            metadata.update(_benchmark(torch, workload, kernel))
        cache.commit()
        return metadata


# compare_generated benchmarks unmodified PyTorch-Inductor and candidate wrappers on an exact H100.
@app.function(image=image, gpu="H100!", timeout=1200, volumes={"/cache": cache})
def compare_generated(baseline_source, candidate_source, payload, expected_chip):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/cache/torchinductor"

    import torch
    import triton

    metadata = _metadata(torch, triton)
    if expected_chip != "H100" or "H100" not in metadata["gpu"]:
        raise RuntimeError(
            f"requested {expected_chip}, sandbox provisioned {metadata['gpu']}"
        )

    with tempfile.TemporaryDirectory(prefix="tilesmith-compare-") as directory:
        baseline_path = Path(directory) / "baseline.py"
        candidate_path = Path(directory) / "candidate.py"
        baseline_path.write_text(baseline_source, encoding="utf-8")
        candidate_path.write_text(candidate_source, encoding="utf-8")
        serialized = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=False
        )
        cuda_constants = _to_cuda(torch, serialized["constants"])
        cuda_inputs = _to_cuda(torch, serialized["inputs"])
        baseline = _load_generated(torch, baseline_path, cuda_constants)
        candidate = _load_generated(torch, candidate_path, cuda_constants)
        _check_generated(torch, baseline, candidate, cuda_inputs)

        # reference adapts the unmodified boxed wrapper to the paired timer.
        def reference(values):
            return baseline(list(values))

        # measured adapts the candidate boxed wrapper to the paired timer.
        def measured(values):
            return candidate(list(values))

        metadata.update(
            _benchmark_pair(
                torch,
                reference,
                measured,
                cuda_inputs,
                "torch_inductor",
                25,
                100,
                7,
            )
        )
        metadata["reference"] = "torch_inductor_generated"
        cache.commit()
        return metadata


# main transfers local source to Modal so remote code never depends on local absolute paths.
@app.local_entrypoint()
def main(
    action: str = "extract", workload: str = "workload.py", kernel: str = "generated.py"
):
    if action not in {"extract", "check", "bench", "evaluate"}:
        raise ValueError("action must be extract, check, bench, or evaluate")

    workload_source = Path(workload).read_text(encoding="utf-8")
    kernel_path = Path(os.environ.get("TILESMITH_KERNEL", kernel))
    kernel_source = ""
    if action != "extract":
        kernel_source = kernel_path.read_text(encoding="utf-8")

    result = run_gpu.remote(action, workload_source, kernel_source)
    if action == "extract":
        source = result.pop("source")
        kernel_path.write_text(source, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if action in {"bench", "evaluate"}:
        print(f"{result['score']:.9g}")
