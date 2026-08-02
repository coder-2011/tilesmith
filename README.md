# Tilesmith

Tilesmith extracts TorchInductor's generated Triton wrapper, asks an LLM for
better source, and installs a candidate only after reference correctness and
NVIDIA GPU timing both pass.

## PyTorch API

Run from this checkout on an NVIDIA machine with PyTorch 2.13, Go, and
CLIProxyAPI configured:

```sh
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.4
```

```python
import tilesmith

optimized_model = tilesmith.compile(model, example_inputs=(x,))
output = optimized_model(x)
```

`tilesmith.compile` returns a normal `torch.compile` callable, so PyTorch keeps
the model's output structure and autograd integration. Like `torch.compile`, it
is lazy: `compile()` validates the example inputs without executing the model,
and the first real call compiles each eligible Inductor graph and runs the
agent. Forward and lazy backward graphs use the same integration.

For every graph, Tilesmith checks normal, zero, and high-magnitude inputs
against the original generated callable before timing warmed CUDA execution.
It uses 25 warmups and the median of seven 100-iteration CUDA-event samples. An
optimized callable is installed only after the agent repeats its final
correctness and speedup confirmation.

Validated artifacts are cached in `~/.cache/tilesmith`. The key includes the
generated source, PyTorch/Triton/CUDA versions, NVIDIA device, agent settings,
and loaded skill contents. Input-mutating graphs, non-Triton graphs, and graphs
with extra Inductor runtime wrappers safely remain on ordinary Inductor.

## Agent benchmark

Configure a Modal profile and CLIProxyAPI, then run the explicit paid benchmark:

```sh
modal token set --profile tilesmith-h100
export MODAL_PROFILE=tilesmith-h100
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.4

python3 benchmark.py run
```

This deploys one scale-to-zero H100 evaluator, extracts the included RMSNorm
through TorchInductor at `4096 × 4096` fp16, and creates a correct smoke-test
baseline with four redundant launches. The agent makes four conversational model
calls, retaining every proposed source and evaluator result. Its first turn must
use CLIProxyAPI's hosted web search, and later turns can search whenever useful.
When present, the benchmark also loads the local `check-normal` and
`high-performance-cuda` skills plus their relevant benchmarking, correctness,
and Triton references. It exits nonzero unless the full loop searches the web
and confirms at least a 25% improvement. This controlled fixture proves the
agent can find an obvious win; its speedup is not a claim about unmodified
TorchInductor output.

Ordinary `go test ./...` never invokes Modal. Only `benchmark.py run` provisions
the paid GPU.

## Manual use

Deploy the evaluator and extract `generated.py` from your workload:

```sh
modal deploy modal_app.py
modal run -q modal_app.py --action extract

go run . \
  -kernel generated.py \
  -evaluate 'python3 benchmark.py evaluate --profile tilesmith-h100 --workload workload.py' \
  -skill "$HOME/.codex/skills/check-normal/SKILL.md" \
  -skill "$HOME/.codex/skills/high-performance-cuda" \
  -iterations 4 \
  -timeout 20m
```

The evaluator receives each candidate in `TILESMITH_KERNEL`. It copies that
source and the local workload into Modal, checks normal, zero, and
high-magnitude inputs against eager PyTorch, warms the H100, and reports the
median of seven CUDA-event samples. A named Modal Volume persists the
source-hashed TorchInductor/Triton cache between runs. Credentials stay in the
Modal profile and are never written into this repository.

`-skill` accepts either a file or a directory containing `SKILL.md` and can be
repeated. Skill files are trusted developer context, so only pass guidance you
control. Up to 8 MiB is accepted. Use `-web-search=false` only for a model or
proxy route that cannot provide the hosted tool.

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
