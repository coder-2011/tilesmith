import importlib.util
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import modal


app = modal.App("tilesmith")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.13.0")
cache = modal.Volume.from_name("tilesmith-torchinductor-cache", create_if_missing=True)


# _load_module imports one self-contained Python file under a unique module name.
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# _clone_inputs keeps reference and candidate executions from sharing mutable tensors.
def _clone_inputs(torch, inputs):
    return tuple(
        value.clone(memory_format=torch.preserve_format) if torch.is_tensor(value) else value
        for value in inputs
    )


# _outputs normalizes a single tensor and a tuple of tensors to the same comparison shape.
def _outputs(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


# _metadata records the exact software and NVIDIA device used for each result.
def _metadata(torch, triton):
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
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


# _extract compiles the workload once and reads TorchInductor's emitted wrapper.
def _extract(torch, workload, debug_dir):
    from torch._inductor import config

    # FX graph cache hits skip debug artifacts; kernel caches remain enabled.
    config.fx_graph_cache = False
    inputs = workload.make_inputs("cuda")
    expected_inputs = _clone_inputs(torch, inputs)
    compiled_inputs = _clone_inputs(torch, inputs)
    with torch.no_grad():
        expected = workload.model(*expected_inputs)
        compiled = torch.compile(workload.model, fullgraph=True)
        actual = compiled(*compiled_inputs)
        torch.testing.assert_close(
            _outputs(actual),
            _outputs(expected),
            atol=getattr(workload, "ATOL", 1e-5),
            rtol=getattr(workload, "RTOL", 1e-4),
            equal_nan=True,
        )
    torch.cuda.synchronize()

    generated = list(Path(debug_dir).rglob("output_code.py"))
    if len(generated) != 1:
        raise RuntimeError(f"expected one output_code.py, found {len(generated)}")
    return generated[0].read_text(encoding="utf-8")


# _benchmark measures only warmed GPU execution with CUDA events.
def _benchmark(torch, workload, kernel):
    inputs = workload.make_inputs("cuda")
    warmup = getattr(workload, "WARMUP", 25)
    iterations = getattr(workload, "BENCH_ITERATIONS", 100)
    sample_count = getattr(workload, "BENCH_SAMPLES", 7)

    with torch.no_grad():
        for _ in range(warmup):
            kernel.call(list(inputs))
        torch.cuda.synchronize()

        samples = []
        for _ in range(sample_count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                kernel.call(list(inputs))
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), samples


# run_gpu extracts, checks, or benchmarks one generated wrapper on an NVIDIA L4.
@app.function(image=image, gpu="L4", timeout=1200, volumes={"/cache": cache})
def run_gpu(action, workload_source, kernel_source):
    debug_dir = tempfile.mkdtemp(prefix="tilesmith-debug-")
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = debug_dir
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/cache/torchinductor"

    import torch
    import triton

    with tempfile.TemporaryDirectory(prefix="tilesmith-") as directory:
        workload_path = Path(directory) / "workload.py"
        workload_path.write_text(workload_source, encoding="utf-8")
        workload = _load_module("tilesmith_workload", workload_path)
        metadata = _metadata(torch, triton)

        if action == "extract":
            metadata["source"] = _extract(torch, workload, debug_dir)
            cache.commit()
            return metadata

        kernel_path = Path(directory) / "candidate.py"
        kernel_path.write_text(kernel_source, encoding="utf-8")
        kernel = _load_module("tilesmith_candidate", kernel_path)
        metadata["cases"] = _check(torch, workload, kernel)
        if action == "bench":
            milliseconds, samples = _benchmark(torch, workload, kernel)
            metadata["milliseconds"] = milliseconds
            metadata["samples_ms"] = samples
        cache.commit()
        return metadata


# main transfers local source to Modal so remote code never depends on local absolute paths.
@app.local_entrypoint()
def main(action: str = "extract", workload: str = "workload.py", kernel: str = "generated.py"):
    if action not in {"extract", "check", "bench"}:
        raise ValueError("action must be extract, check, or bench")

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
    if action == "bench":
        print(f"{result['milliseconds']:.9g}")
