# Tilesmith

Tilesmith extracts TorchInductor's generated Triton wrapper, asks an LLM for
better source, and installs a candidate only after eager-PyTorch correctness and
NVIDIA GPU timing both pass.

## Run

Authenticate Modal, then extract `generated.py` from the included RMSNorm
workload on an NVIDIA L4:

```sh
modal setup
modal run -q modal_app.py --action extract
```

Point Tilesmith at CLIProxyAPI and run one optimization attempt:

```sh
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.4

go run . \
  -kernel generated.py \
  -check 'modal run -q modal_app.py --action check' \
  -bench 'modal run -q modal_app.py --action bench' \
  -iterations 1 \
  -trials 1 \
  -timeout 20m
```

The Python commands receive each candidate in `TILESMITH_KERNEL`. The harness
copies that source and the local workload into Modal, checks normal, zero, and
high-magnitude inputs against eager PyTorch, warms the GPU, and reports the
median of seven CUDA-event samples. A named Modal Volume persists the
source-hashed TorchInductor/Triton cache between runs.

`workload.py` is also the customization interface. Replace `model()` and
`make_inputs()` with a self-contained workload; optionally provide
`make_cases()`, `ATOL`, `RTOL`, and the benchmark constants shown in the example.

## Using the result

The installed file is the callable Python actually benchmarks. On an NVIDIA
machine, load it before the training loop and call its generated interface:

```python
import runpy

kernel = runpy.run_path("generated.py")
(output,) = kernel["call"]([input_tensor])
```

Replacing `generated.py` does not mutate a `torch.compile` callable that is
already loaded in memory. Restart that process or load the optimized artifact
before use. Tilesmith atomically replaces the file only after a confirmed
speedup; otherwise the original remains unchanged.
