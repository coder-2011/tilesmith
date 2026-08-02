# Tilesmith

Tilesmith extracts TorchInductor's generated Triton wrapper, asks an LLM for
better source, and installs a candidate only after reference correctness and
NVIDIA GPU timing both pass.

## PyTorch API

Run from this checkout on an NVIDIA machine with PyTorch 2.13, Go, and
CLIProxyAPI configured:

```sh
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.5
```

```python
import tilesmith

optimized_model = tilesmith.compile(model, example_inputs=(x,))
output = optimized_model(x)
```

To keep execution and the agent local but send every candidate benchmark to an
exact H100 sandbox, deploy the evaluator once and select it at compile time:

```sh
MODAL_PROFILE=tilesmith-h100 modal deploy modal_app.py
```

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    sandbox="modal",
    chip="H100",
)
output = optimized_model(x)
```

Only benchmarking is remote. The returned callable still executes in the
calling Python process, so its CUDA inputs must also be on an H100. Modal is
currently the only sandbox and H100 is the only remote chip; Tilesmith rejects
an omitted, unsupported, or execution-mismatched chip before starting PyTorch
compilation.

`tilesmith.compile` returns a normal `torch.compile` callable, so PyTorch keeps
the model's output structure and autograd integration. Like `torch.compile`, it
is lazy: `compile()` validates the example inputs without executing the model,
and the first real call compiles each eligible Inductor graph and runs the
agent. Forward and lazy backward graphs use the same integration.

The agent receives `benchmark_candidate(source)` as a strict function tool in
addition to hosted web search. Tilesmith owns loading, correctness, warmup, and
timing; the model never writes benchmark code. A proposal is eligible only if
its exact normalized source hash matches a successful tool call. The generated
Inductor `call(args)` function is the enforced ABI, so no marker decorator is
needed.

Each graph agent pins the current best source, keeps only its five newest
proposal/evaluation turns, and may make up to four paid benchmark calls per
attempt. The exact evaluator chip is included in its trusted context.

For every graph, Tilesmith checks normal, zero, and high-magnitude inputs
against the original PyTorch-Inductor callable before timing warmed CUDA
execution. It uses 25 warmups, alternates baseline/candidate order, and reports
the median of seven paired 100-iteration CUDA-event ratios plus both raw sample
sets. An optimized callable is installed only after a fresh correctness and
speedup confirmation.

Validated artifacts are cached in `~/.cache/tilesmith`. The key includes the
generated source, PyTorch/Triton/CUDA versions, local NVIDIA device, sandbox and
chip, agent settings, and loaded skill contents. Repeating the same compile
track loads that exact accepted source; PyTorch/Triton's normal source cache
then avoids recompiling it. A changed graph or environment starts a new track.
Input-mutating graphs, non-Triton graphs, and graphs with extra Inductor runtime
wrappers safely remain on ordinary Inductor.

## Agent benchmark

Configure a Modal profile and CLIProxyAPI, then run the explicit paid benchmark:

```sh
modal token set --profile tilesmith-h100
export MODAL_PROFILE=tilesmith-h100
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.5

python3 benchmark.py run
```

This deploys one scale-to-zero exact H100 evaluator, extracts the included
RMSNorm through TorchInductor at `4096 × 4096` fp16, and creates a correct
smoke-test baseline with four redundant launches. The agent makes four outer
optimization attempts and can benchmark several candidates inside each model
tool loop. Its first turn must use CLIProxyAPI's hosted web search, and later
turns can search whenever useful.
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
high-magnitude inputs against eager PyTorch, then alternates eager and candidate
timing on the H100. A named Modal Volume persists the source-hashed
TorchInductor/Triton cache between runs. Credentials stay in the Modal profile
and are never written into this repository.

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
