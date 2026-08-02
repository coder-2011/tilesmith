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

## Remote benchmarks

The agent loop always runs in the calling Python process. To run candidate
correctness and timing in a ComputeSDK sandbox instead, export the gateway
configuration and name the execution chip explicitly:

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

Only benchmarking is remote; the returned callable still runs locally, so its
CUDA device must match `chip`. `COMPUTESDK_TEMPLATE` must provide that GPU and
host-compatible PyTorch and Triton versions. Tilesmith keeps one ephemeral
sandbox warm for the whole graph optimization track, then destroys it.
`COMPUTESDK_TRIBUTARY_URL` overrides the gateway and
`TILESMITH_SANDBOX_TIMEOUT` controls the maximum sandbox lifetime in seconds.
The pinned gateway protocol currently routes through Modal provider tokens,
but Tilesmith does not import or configure Modal itself.

## Agent graph

Each uncached Triton graph gets a Deep Agents `CompiledStateGraph` with file
tools, local bash, context management, a general-purpose subagent, hosted web
search, and the real `benchmark_candidate(path)` tool. The graph works in five
persistent rolling files at
`~/.cache/tilesmith/tracks/<graph-key>/attempts/attempt_00.py` through
`attempt_04.py`. Each file starts with the current PyTorch FX graph as comments,
followed by the executable Inductor/Triton wrapper. The next attempt is copied
from the last one; after `attempt_04.py`, the oldest contents are discarded and
the remaining four shift down.

Bash receives only a minimal environment without API credentials and is
instructed to work in that graph directory. Deep Agents' `LocalShellBackend` is
still host shell access, not an operating-system security sandbox; use
Tilesmith only with models and workloads you trust. Candidate execution remains
exclusive to the benchmark tool.

The graph chooses its own optimization workflow. The host restricts only the
expensive boundary: `loops` is the number of benchmark/profile cycles and
defaults to one. A parent LangGraph advances only after
`benchmark_candidate(path)` measures one new candidate; duplicate source does
not consume a loop. Both the parent graph and each Deep Agent round have a
fixed 70-step safety ceiling. Timing repetitions are intentionally internal,
not user-configurable. The model context retains the five newest benchmark
payloads without discarding other tool work.

The tool reads the named attempt, restores its trusted FX comment header,
rejects incorrect source before timing, and returns loop progress, every raw
timing sample, a complete PyTorch/Kineto Chrome trace, operator summaries,
stacks, the raw memory timeline, allocator statistics/snapshot, GPU metadata,
and the exact evaluator chip. ComputeSDK evaluation also runs Nsight Compute's
full metric set when `ncu` is installed, returning its raw CSV or the complete
availability/failure result. Every packet is also retained under the graph's
`profiles/` directory. The host selects the fastest successfully measured
candidate regardless of the agent's final text and runs a fresh confirmation
before installation.

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

## H100 agent benchmark

After exporting the LLM and ComputeSDK variables above, run the explicit paid
fixture:

```sh
python3 benchmark.py run --loops 1
```

It provisions one H100, extracts the included `4096 × 4096` fp16 RMSNorm, and
creates a correct baseline with four redundant Triton launches. It exits
nonzero unless the graph is multi-turn, calls the real benchmark tool, removes
the controlled waste, and freshly confirms at least a 25% speedup. When
available, the graph also receives the complete local `high-performance-triton`
and `KernelWiki` skills alongside `check-normal` and `high-performance-cuda`
guidance.

Ordinary local tests never provision ComputeSDK. Only `benchmark.py run` or an
uncached `compile(..., sandbox="computesdk", chip=...)` call creates a paid GPU.
Credential values are never written to the repository or passed to the agent.
