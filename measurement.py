"""Own Tilesmith's shared CUDA correctness, timing, and profiling policy."""

import gzip
import json
import shutil
import statistics
import subprocess
from pathlib import Path


# variant_inputs clones one compiled input shape and applies the requested correctness values.
def variant_inputs(torch, inputs, variant):
    values = []
    for value in inputs:
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


# benchmark_pair returns fixed-policy paired CUDA-event samples and their median ratio.
def benchmark_pair(
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
                measured(list(inputs))
        torch.cuda.synchronize()

        samples = {name: [] for name, _ in functions}
        for sample in range(sample_count):
            order = functions if sample % 2 == 0 else tuple(reversed(functions))
            for name, measured in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    measured(list(inputs))
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


# profile_cuda captures the complete in-process Kineto and CUDA allocator packet.
def profile_cuda(torch, function, inputs, directory, prefix):
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

    profiling = {
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
        "kineto": _profile_exports(profiler, directory, prefix),
        "peak_memory_allocated": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved": torch.cuda.max_memory_reserved(),
    }
    if shutil.which("nvidia-smi") is not None:
        try:
            profiling["nvidia_smi_q_xml"] = subprocess.check_output(
                ["nvidia-smi", "-q", "-x"],
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            profiling["nvidia_smi_error"] = str(error)
    return profiling


# _profile_exports turns one Kineto recording into raw and readable report forms.
def _profile_exports(profiler, directory, prefix):
    directory = Path(directory)
    trace_path = directory / f"{prefix}-trace.json"
    memory_path = directory / f"{prefix}-memory.raw.json.gz"
    cpu_stacks_path = directory / f"{prefix}-cpu-stacks.txt"
    device_stacks_path = directory / f"{prefix}-device-stacks.txt"
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
