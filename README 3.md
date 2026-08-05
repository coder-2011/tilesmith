# Tilesmith

Tilesmith is a small addition to `torch.compile`, not an alternative compiler.
PyTorch runs Dynamo, AOTAutograd, and TorchInductor normally; only after Inductor
finishes an eligible raw Triton wrapper does Tilesmith ask a LangGraph Deep Agent
to improve it. A candidate is installed only after reference correctness and
NVIDIA GPU timing both pass.

## PyTorch API

Install the agent dependencies in an environment that already has CUDA,
PyTorch, and Triton:

```sh
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...
export TILESMITH_MODEL=gpt-5.5
```

`OPENROUTER_API_KEY` is also accepted and selects OpenRouter's compatible API
automatically.

`OPENAI_BASE_URL` may point at any OpenAI Responses-compatible gateway,
including an independently running CLIProxyAPI server:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8317/v1
export OPENAI_API_KEY=your-proxy-key
```

Use Tilesmith like `torch.compile`:

```python
import tilesmith

optimized_model = tilesmith.compile(model, example_inputs=(x,), loops=1)
output = optimized_model(x)
```

`tilesmith.compile` returns a normal `torch.compile` callable, so PyTorch keeps
the model's output structure, autograd integration, compiler configuration, and
runtime wrappers. Compilation remains lazy: after PyTorch finishes each eligible
forward or backward Inductor graph, the first real call asks Tilesmith to improve
that artifact. Dynamic-shape, mixed-device, multi-GPU, input-mutating,
non-Triton, and wrapped graphs safely stay on ordinary Inductor.
Tilesmith bypasses only AOTAutograd's higher-level artifact cache because its
warm hits skip the generated-source hook; Inductor's FX/code caches remain
enabled. Agent or sandbox failures warn and fall back to ordinary Inductor.

## Remote benchmarks

The agent loop always runs in the calling Python process. To run candidate
correctness and timing in a ComputeSDK sandbox instead, configure a direct
provider and name the execution chip explicitly. The reviewed Modal default
needs only its provider credentials:

```sh
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    sandbox="computesdk",
    chip="H100!",
)
output = optimized_model(x)
```

Only benchmarking is remote; the returned callable still runs locally, so its
CUDA device must match `chip`. Tilesmith keeps one ephemeral sandbox warm for
the whole graph optimization track, then destroys it.
Remote optimization currently accepts only local `cuda:0` graphs because a
one-GPU sandbox exposes its target as `cuda:0`.
`TILESMITH_SANDBOX_TIMEOUT` controls the maximum sandbox lifetime in seconds.
Modal uses a pinned build-capable Python/uv image by default. Tilesmith waits
for real command execution, verifies Python, uv, and a C compiler, then installs
and checks the host's exact PyTorch and Triton versions. `COMPUTESDK_TEMPLATE`
overrides that image when a custom environment is needed.
Tilesmith loads the pinned ComputeSDK and provider npm packages through its
bundled Node bridge; it does not use a ComputeSDK API key or import a provider
Python SDK. Other providers can be selected with `COMPUTESDK_PROVIDER`,
`COMPUTESDK_PROVIDER_PACKAGE`, `COMPUTESDK_PROVIDER_FACTORY`,
`COMPUTESDK_PROVIDER_VERSION`, and a JSON object in
`COMPUTESDK_PROVIDER_CONFIG`.

## Agent graph

Each uncached Triton graph gets a Deep Agents `CompiledStateGraph` with file
tools, local bash, context management, a general-purpose subagent, and hosted
web search. A ComputeSDK run also gets `gpu_shell(command, timeout_seconds)`, an
unrestricted shell in the same warm GPU sandbox used for final validation. The
graph works in five persistent rolling files at
`~/.cache/tilesmith/tracks/<graph-key>/attempts/attempt_00.py` through
`attempt_04.py`. Each file starts with the current PyTorch FX graph as comments,
followed by the executable Inductor/Triton wrapper. The next attempt is copied
from the last one; after `attempt_04.py`, the oldest contents are discarded and
the remaining four shift down.

Bash receives only a minimal environment without API credentials and works in
that graph directory. Deep Agents' `LocalShellBackend` is still host shell
access, not an operating-system security sandbox; use Tilesmith only with
models and workloads you trust. `gpu_shell` is different: it executes arbitrary
commands inside the ephemeral GPU sandbox. The current attempt is mirrored to
`/tmp/tilesmith/workbench/candidate.py` before every command, remote edits to
that file are copied back afterward, and other remote files persist for the
graph. The host keeps separate authoritative harness files for its acceptance
gate. The agent can run Python, install packages, write experiments, invoke
Triton or CUDA, inspect PTX/SASS, and use `ncu`, `nsys`, or any tools present in
the template.

The graph chooses its own optimization and profiling workflow. `loops`, which
defaults to one, is the number of candidates accepted by the host benchmark
gate. When an agent turn ends, the parent LangGraph automatically checks and
times the current attempt. A rejected or duplicate candidate does not consume a
loop; its report is returned on the next turn. Both the parent graph and each
Deep Agent round have a fixed 70-step safety ceiling. Timing repetitions remain
internal and are not user-configurable.

The host restores the trusted FX comment header and rejects incorrect source
before timing. Complete host reports are retained under the graph's `profiles/`
directory. Profiling is deliberately separate: the remote workspace includes
the trusted wrapper, serialized inputs, and a neutral one-run candidate target.
The agent chooses and runs any profiler or experiment through `gpu_shell`; for
example, it can wrap that target with `ncu` or `nsys`. Artifacts persist for the
graph under `artifacts/`. The host selects the fastest successfully measured
candidate regardless of the agent's final text and runs a fresh
correctness-and-timing confirmation before installation.

Correctness covers normal, zero, and high-magnitude inputs against the original
TorchInductor callable. Timing uses 25 warmups, alternating baseline/candidate
order, and the median of seven paired 100-iteration CUDA-event ratios with all
raw samples retained. `TILESMITH_MIN_IMPROVEMENT`, default `0.01`, sets the
required fractional confirmed speedup.

Validated artifacts are cached in `~/.cache/tilesmith`. The key includes the
generated source, PyTorch/Triton/CUDA and Deep Agents versions, NVIDIA device,
ComputeSDK configuration and chip, agent settings, and loaded skill contents.
Repeating the same compile track loads that exact accepted source;
PyTorch and Triton's normal cache then avoids rebuilding it.

## GPU agent benchmark

After exporting the LLM and ComputeSDK variables above, run the explicit paid
fixture:

```sh
python3 benchmark.py run --chip L4 --loops 1
```

It provisions one requested GPU, extracts the included `4096 × 4096` fp16 RMSNorm, and
creates a correct baseline with four redundant Triton launches. It exits
nonzero unless the graph is multi-turn, uses its open GPU shell, removes the
controlled waste, and freshly confirms at least a 25% speedup. When
available, the graph also receives the complete local `high-performance-triton`
and `KernelWiki` skills alongside `check-normal` and `high-performance-cuda`
guidance. Every paid run uses a fresh persistent track so a prior fixture win
cannot satisfy a later test.

For a public-API model comparison, run the explicit MoE harness:

```sh
python3 moe_benchmark.py --chip L4
```

It provisions one requested Modal GPU, installs PyTorch and Tilesmith's requirements,
and runs the same deterministic BF16 top-two MoE Transformer as eager PyTorch,
ordinary `torch.compile`, and `tilesmith.compile`. It requires output parity
before reporting interleaved CUDA-event latency samples. This outer harness
runs the public API inside the requested GPU because its example inputs must be real
CUDA tensors; normal `sandbox="computesdk"` mode still keeps the agent in its
calling process.

Ordinary local tests never provision ComputeSDK. Only `benchmark.py run` or an
uncached `compile(..., sandbox="computesdk", chip=...)` call, plus the explicit
`moe_benchmark.py` harness, creates a paid GPU. Provider credentials are never
written to the repository or passed into the sandbox. The MoE harness accepts
`OPENAI_API_KEY` or `OPENROUTER_API_KEY` and forwards only that model API
credential to the remote Tilesmith process.
