# Tilesmith

Tilesmith extracts TorchInductor's generated Triton wrapper, asks an LLM for
better source, and installs a candidate only after reference correctness and
NVIDIA GPU timing both pass.

## PyTorch API

Run from this checkout on an NVIDIA machine with PyTorch 2.13, Go, `httpx`,
and CLIProxyAPI configured:

```sh
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.5
```

```python
import tilesmith

optimized_model = tilesmith.compile(model, example_inputs=(x,))
output = optimized_model(x)
```

To keep execution and the agent local but send every candidate benchmark to a
ComputeSDK sandbox, export the gateway configuration and select the chip at
compile time:

```sh
export COMPUTESDK_API_KEY=...
export COMPUTESDK_TEMPLATE=...
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    sandbox="computesdk",
    chip="H100",
)
output = optimized_model(x)
```

Only benchmarking is remote. The returned callable still executes in the
calling Python process, so its CUDA inputs must match `chip`. Tilesmith rejects
an omitted or execution-mismatched chip before starting PyTorch compilation.
`COMPUTESDK_TEMPLATE` must provide that GPU plus Python and host-matching
PyTorch/Triton versions.

Tilesmith creates one ephemeral sandbox for each uncached graph optimization
track, keeps it warm across all of that agent's benchmark calls, and destroys
it afterward. `COMPUTESDK_TRIBUTARY_URL` can override the gateway and
`TILESMITH_SANDBOX_TIMEOUT` controls its maximum lifetime in seconds. The
pinned ComputeSDK gateway protocol currently routes through Modal and therefore
requires exported Modal provider tokens, but Tilesmith does not import, deploy,
or configure Modal itself.

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
chip, ComputeSDK gateway/template, agent settings, and loaded skill contents.
Repeating the same compile track loads that exact accepted source; PyTorch/
Triton's normal source cache then avoids recompiling it. A changed graph or
environment starts a new track.
Input-mutating graphs, non-Triton graphs, and graphs with extra Inductor runtime
wrappers safely remain on ordinary Inductor.

## Agent benchmark

Export the ComputeSDK variables above, configure CLIProxyAPI, then run the
explicit paid benchmark:

```sh
export CLIPROXY_CONFIG=/path/to/cliproxy-config.yaml
export TILESMITH_MODEL=gpt-5.5

python3 benchmark.py run
```

This provisions one ephemeral H100 sandbox, extracts the included RMSNorm
through TorchInductor at `4096 × 4096` fp16, and creates a correct smoke-test
baseline with four redundant launches. The same sandbox handles every agent
benchmark call so compilation caches stay warm. The agent makes four outer
optimization attempts and can benchmark several candidates inside each model
tool loop. Its first turn must use CLIProxyAPI's hosted web search, and later
turns can search whenever useful.
When present, the benchmark also loads the local `check-normal` and
`high-performance-cuda` skills plus their relevant benchmarking, correctness,
and Triton references. It exits nonzero unless the full loop searches the web
and confirms at least a 25% improvement. This controlled fixture proves the
agent can find an obvious win; its speedup is not a claim about unmodified
TorchInductor output.

Ordinary `go test ./...` never invokes ComputeSDK. Only `benchmark.py run` or an
uncached `compile(..., sandbox="computesdk", chip=...)` provisions a paid GPU.
Credential values are never written into this repository or passed to the
agent.
