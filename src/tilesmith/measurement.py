"""Own Tilesmith's shared CUDA correctness and timing policy."""

import re
import statistics


# chip_matches compares provider selectors and device names by exact GPU model token.
def chip_matches(requested, observed):
    pattern = r"[a-z]+\d+[a-z]*"
    requested_tokens = set(re.findall(pattern, requested.rstrip("!").lower()))
    observed_tokens = set(re.findall(pattern, observed.lower()))
    return bool(requested_tokens & observed_tokens)


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


# assert_candidate_matches applies Tilesmith's shared numerical gate to one generated candidate.
def assert_candidate_matches(torch, reference, candidate, inputs):
    with torch.no_grad():
        for variant in ("normal", "zero", "high"):
            cpu_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all()
            expected_inputs = variant_inputs(torch, inputs, variant)
            expected = reference(list(expected_inputs))
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)
            actual_inputs = variant_inputs(torch, inputs, variant)
            actual = candidate(list(actual_inputs))
            torch.testing.assert_close(
                actual,
                expected,
                equal_nan=True,
                check_stride=True,
            )


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
        # Generated wrappers clear boxed inputs, so every invocation needs a fresh list.
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
            samples[reference_name], samples["candidate_triton"], strict=True
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
