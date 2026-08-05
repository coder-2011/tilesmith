"""Own Tilesmith's shared CUDA correctness and timing policy."""

import io
import math
import re
import statistics


# chip_matches compares provider selectors and device names by exact GPU model token.
def chip_matches(requested, observed):
    """Return whether two GPU descriptions share an exact model token."""
    requested = requested.rstrip("!").lower()
    observed = observed.lower()
    compact_pattern = r"[a-z]+\d+[a-z]*"
    requested_compact = set(re.findall(compact_pattern, requested))
    observed_compact = set(re.findall(compact_pattern, observed))
    if requested_compact & observed_compact:
        return True
    pattern = r"[a-z]+|\d+[a-z]*"
    ignored = {"blackwell", "geforce", "gpu", "nvidia"}
    requested_tokens = [token for token in re.findall(pattern, requested) if token not in ignored]
    observed_tokens = [token for token in re.findall(pattern, observed) if token not in ignored]
    requested_numbers = [index for index, token in enumerate(requested_tokens) if token[0].isdigit()]
    if not requested_numbers:
        return False
    model_end = requested_numbers[0] + 1
    model = requested_tokens[:model_end]
    width = len(model)
    return any(observed_tokens[index : index + width] == model for index in range(len(observed_tokens) - width + 1))


# clone_tensor_graph copies one object graph while retaining tensor storage aliases and views.
def clone_tensor_graph(torch, value, map_location=None):
    """Clone tensors as one serialization graph, optionally moving every storage."""
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location=map_location, weights_only=False)


# _tensor_leaves returns tensor occurrences in deterministic container traversal order.
def _tensor_leaves(torch, value):
    """Collect tensor leaves without deduplicating repeated references."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in _tensor_leaves(torch, child)]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in _tensor_leaves(torch, child)]
    return []


# _tensor_versions captures observable in-place mutation without retaining tensors.
def _tensor_versions(torch, value):
    """Return each tensor's version counter, or None when it has no counter."""
    versions = []
    for tensor in _tensor_leaves(torch, value):
        try:
            versions.append(tensor._version)
        except RuntimeError:
            versions.append(None)
    return tuple(versions)


# _version_changes normalizes counters so independently cloned graphs can be compared.
def _version_changes(before, after):
    """Return the increment made to every available tensor version counter."""
    return tuple(None if start is None or end is None else end - start for start, end in zip(before, after, strict=True))


# _alias_signature describes storage sharing and layout without retaining raw addresses.
def _alias_signature(torch, value):
    """Return a comparable storage-group signature for every tensor occurrence."""
    groups = {}
    signature = []
    for tensor in _tensor_leaves(torch, value):
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage._cdata)
        group = groups.setdefault(storage_key, len(groups))
        signature.append(
            (
                group,
                tuple(tensor.size()),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
            )
        )
    return tuple(signature)


# assert_tensor_graph_matches checks values, structure, layout, and storage relationships.
def assert_tensor_graph_matches(torch, actual, expected, label="tensor graph"):
    """Reject observable tensor-state or alias differences between two object graphs."""
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.is_tensor(expected):
            raise AssertionError(f"{label} changed tensor structure")
        if type(actual) is not type(expected):
            raise AssertionError(f"{label} changed tensor type")
        _assert_quantization_matches(torch, actual, expected, label)
        torch.testing.assert_close(actual, expected, equal_nan=True, check_stride=True)
        if actual.storage_offset() != expected.storage_offset():
            raise AssertionError(f"{label} changed a tensor storage offset")
        if actual.is_conj() != expected.is_conj():
            raise AssertionError(f"{label} changed lazy conjugation state")
        _assert_signed_zero_matches(torch, actual, expected, label)
        return
    if type(actual) is not type(expected):
        raise AssertionError(f"{label} changed container type")
    if isinstance(actual, dict):
        if tuple(actual) != tuple(expected):
            raise AssertionError(f"{label} changed mapping keys or order")
        for key in actual:
            assert_tensor_graph_matches(torch, actual[key], expected[key], label)
    elif isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{label} changed container length")
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_tensor_graph_matches(
                torch,
                actual_item,
                expected_item,
                label,
            )
    elif actual != expected:
        raise AssertionError(f"{label} changed a non-tensor value")


# _assert_quantization_matches preserves the encoding metadata hidden by dequantized values.
def _assert_quantization_matches(torch, actual, expected, label):
    """Reject changes to a quantized tensor's scheme, scale, zero point, or axis."""
    if actual.is_quantized != expected.is_quantized:
        raise AssertionError(f"{label} changed tensor quantization")
    if not actual.is_quantized:
        return
    qscheme = actual.qscheme()
    if qscheme != expected.qscheme():
        raise AssertionError(f"{label} changed tensor quantization scheme")
    if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        if actual.q_scale() != expected.q_scale() or actual.q_zero_point() != expected.q_zero_point():
            raise AssertionError(f"{label} changed tensor quantization parameters")
        return
    if actual.q_per_channel_axis() != expected.q_per_channel_axis():
        raise AssertionError(f"{label} changed tensor quantization axis")
    if not torch.equal(actual.q_per_channel_scales(), expected.q_per_channel_scales()):
        raise AssertionError(f"{label} changed tensor quantization scales")
    if not torch.equal(actual.q_per_channel_zero_points(), expected.q_per_channel_zero_points()):
        raise AssertionError(f"{label} changed tensor quantization zero points")


# _assert_signed_zero_matches preserves the IEEE zero sign ignored by numerical closeness.
def _assert_signed_zero_matches(torch, actual, expected, label):
    """Reject sign-bit changes where both tensor components are exactly zero."""
    if actual.is_complex():
        components = ((actual.real, expected.real), (actual.imag, expected.imag))
    elif actual.is_floating_point():
        components = ((actual, expected),)
    else:
        return
    for actual_part, expected_part in components:
        both_zero = (actual_part == 0) & (expected_part == 0)
        actual_signs = torch.signbit(actual_part) & both_zero
        expected_signs = torch.signbit(expected_part) & both_zero
        if not torch.equal(actual_signs, expected_signs):
            raise AssertionError(f"{label} changed IEEE signed-zero semantics")


# assert_tensor_aliases_match compares alias groups after values and layouts are checked.
def assert_tensor_aliases_match(torch, actual, expected, label="tensor graph"):
    """Reject changes to repeated references, views, or shared tensor storage."""
    if _alias_signature(torch, actual) != _alias_signature(torch, expected):
        raise AssertionError(f"{label} changed tensor alias relationships")


# variant_inputs clones one compiled input graph and applies the requested correctness values.
def variant_inputs(torch, inputs, variant):
    """Return independent inputs with the original storage and view relationships."""
    values = clone_tensor_graph(torch, tuple(inputs))
    if variant == "normal":
        return values
    cloned_tensors = _tensor_leaves(torch, values)
    visited_storages = set()
    for cloned in cloned_tensors:
        storage = cloned.untyped_storage()
        storage_key = (str(cloned.device), storage._cdata)
        if storage_key in visited_storages:
            continue
        visited_storages.add(storage_key)
        element_count = storage.nbytes() // cloned.element_size()
        physical = cloned.new_empty(0).set_(storage, 0, (element_count,), (1,))
        if variant == "zero":
            physical.zero_()
        elif variant == "high" and (physical.is_floating_point() or physical.is_complex()):
            physical.mul_(4)
        elif variant == "high" and physical.dtype == torch.bool:
            physical.fill_(True)
    return values


# _torch_process_state snapshots stable PyTorch policies a generated callable can mutate.
def _torch_process_state(torch):
    """Return caller-visible PyTorch process settings protected during evaluation."""
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "default_dtype": torch.get_default_dtype(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }


# _restore_torch_process_state prevents validation trials from leaking policy changes.
def _restore_torch_process_state(torch, state):
    """Restore a process-state snapshot after one reference or candidate call."""
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.enabled = state["cudnn_enabled"]
    torch.set_default_dtype(state["default_dtype"])
    torch.use_deterministic_algorithms(
        state["deterministic_algorithms"],
        warn_only=state["deterministic_warn_only"],
    )
    torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = state["cuda_matmul_allow_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = state["fp16_reduced_precision_reduction"]


# assert_candidate_matches applies Tilesmith's shared numerical gate to one generated candidate.
def assert_candidate_matches(
    torch,
    reference,
    candidate,
    inputs,
    *,
    reference_constants=(),
    candidate_constants=(),
):
    """Compare values, mutation, aliases, and RNG transitions on shared variants."""
    with torch.no_grad():
        for variant in ("normal", "zero", "high"):
            cpu_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all()
            initial_process_state = _torch_process_state(torch)
            try:
                expected_inputs = variant_inputs(torch, inputs, variant)
                expected_version_start = _tensor_versions(torch, (expected_inputs, reference_constants))
                expected = reference(list(expected_inputs))
                expected_versions = _version_changes(
                    expected_version_start,
                    _tensor_versions(torch, (expected_inputs, reference_constants)),
                )
                expected_cpu_state = torch.random.get_rng_state()
                expected_cuda_states = torch.cuda.get_rng_state_all()
                expected_process_state = _torch_process_state(torch)

                torch.random.set_rng_state(cpu_state)
                torch.cuda.set_rng_state_all(cuda_states)
                _restore_torch_process_state(torch, initial_process_state)
                actual_inputs = variant_inputs(torch, inputs, variant)
                actual_version_start = _tensor_versions(torch, (actual_inputs, candidate_constants))
                actual = candidate(list(actual_inputs))
                actual_versions = _version_changes(
                    actual_version_start,
                    _tensor_versions(torch, (actual_inputs, candidate_constants)),
                )
                actual_cpu_state = torch.random.get_rng_state()
                actual_cuda_states = torch.cuda.get_rng_state_all()
                actual_process_state = _torch_process_state(torch)
            finally:
                _restore_torch_process_state(torch, initial_process_state)
            assert_tensor_graph_matches(
                torch,
                (actual_inputs, candidate_constants, actual),
                (expected_inputs, reference_constants, expected),
                "callable ABI",
            )
            assert_tensor_aliases_match(
                torch,
                (actual_inputs, candidate_constants, actual),
                (expected_inputs, reference_constants, expected),
                "callable ABI",
            )
            assert_tensor_graph_matches(
                torch,
                (actual_cpu_state, actual_cuda_states),
                (expected_cpu_state, expected_cuda_states),
                "RNG transition",
            )
            assert_tensor_graph_matches(
                torch,
                actual_process_state,
                expected_process_state,
                "PyTorch process-state transition",
            )
            if actual_versions != expected_versions:
                raise AssertionError("callable ABI changed tensor version-counter transitions")


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
    """Return finite positive paired CUDA-event samples and their median ratio."""
    functions = ((reference_name, reference), ("candidate_triton", candidate))
    input_snapshot = clone_tensor_graph(torch, inputs)
    process_state = _torch_process_state(torch)
    try:
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
    finally:
        # Timed candidate calls must not leak process-wide policy into the application.
        _restore_torch_process_state(torch, process_state)

    assert_tensor_graph_matches(torch, inputs, input_snapshot, "timing inputs")
    assert_tensor_aliases_match(torch, inputs, input_snapshot, "timing inputs")
    for name, values in samples.items():
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{name} returned non-finite or non-positive timing")

    ratios = [candidate_ms / reference_ms for reference_ms, candidate_ms in zip(samples[reference_name], samples["candidate_triton"], strict=True)]
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
