# Tilesmith

Tilesmith is a small optimization layer for PyTorch. PyTorch still performs
graph capture, AOTAutograd, and TorchInductor compilation. After Inductor emits
an eligible CUDA wrapper, Tilesmith gives that complete generated implementation
to an Agent, measures its candidates against the original wrapper, and
installs only a numerically correct, independently confirmed speedup.

```python
import tilesmith

optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    agent_model="gpt-5.5",
)
output = optimized_model(x)

for result in tilesmith.benchmarks(optimized_model):
    print(result.torch_inductor_ms, result.tilesmith_ms, result.speedup)

for diagnostic in tilesmith.diagnostics(optimized_model):
    print(diagnostic.kind, diagnostic.message)
```

`optimized_model` remains a normal `torch.compile` callable. Tilesmith does not
replace PyTorch or ask the model to rewrite the user's module. Benchmark results
and fallback diagnostics appear after the first call because optimization is lazy.

## Install

Start with an environment that already has a matching CUDA build of PyTorch and
Triton, then install Tilesmith and its agent runtime:

```sh
python3 -m pip install tilesmith-ai
cliproxyapi -codex-login
```

All model traffic goes through a private CLIProxyAPI process. Its upstreams may
be OAuth accounts or API providers from the CLIProxyAPI YAML. Tilesmith detects
accounts at `~/.cli-proxy-api` and Homebrew's CLIProxyAPI config; set
`TILESMITH_CLIPROXYAPI_EXECUTABLE`, `TILESMITH_CLIPROXYAPI_AUTH_DIR`, or
`TILESMITH_CLIPROXYAPI_CONFIG` to override those paths. Tilesmith copies the
config for each run, preserving provider and routing settings while replacing
only its private loopback server settings.

Provider keys can instead be exported normally. Tilesmith converts them into
the temporary CLIProxyAPI config; it does not create a direct provider path:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

Supported conventional names are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `XAI_API_KEY`, `VERTEX_API_KEY`, and
`OPENROUTER_API_KEY`. For any OpenAI-compatible endpoint, export
`TILESMITH_API_KEY` together with `TILESMITH_API_BASE_URL`.

## Compile from PyTorch

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    agent_model="gpt-5.5",
    agency="medium",
    loops=1,
    max_model_calls=None,
)
```

`example_inputs` must be a tuple with at least one CUDA tensor. `agent_model`
selects the optimizer model and overrides `TILESMITH_MODEL`; choose one listed
by CLIProxyAPI. Compilation is lazy: PyTorch finishes
compiling first, and the first real invocation supplies the concrete tensors
used for Tilesmith's correctness and timing checks.

When `example_inputs`, `agent_model`/`TILESMITH_MODEL`, or CLIProxyAPI is
unavailable, `tilesmith.compile(...)` returns ordinary
`torch.compile(...)`. `tilesmith.benchmarks(...)` then returns an empty tuple.

`agency` controls how broadly the agent may alter generated code. `"low"`
preserves the existing wrapper and kernel boundaries while allowing localized
tuning such as coalescing, tile sizes, launch parameters, and vectorization.
The default `"medium"` may restructure wrapper logic, intermediate buffers,
and kernel fusion or decomposition while preserving the high-level algorithm.
`"high"` may also redesign the algorithm and generated implementation. All
three levels preserve the generated callable ABI and pass the same host-owned
correctness, timing, and fresh-confirmation gates.

`loops` counts accepted optimization cycles, not agent turns or timing
repetitions. It defaults to `1`. A slower, incorrect, duplicate, or marginal
candidate is rejected and the same loop continues.

`max_model_calls` is an optional exact budget on LangChain model invocations
shared by the main agent and its subagents. It defaults to `None`; there is no
default 70-step limit. Provider transport retries remain inside one invocation.
Shell, tool, benchmark, and graph-transition calls do not consume this budget.
For example, `max_model_calls=70` permits 70 invocations and blocks the 71st.
If the budget expires without a validated win, Tilesmith keeps ordinary
Inductor rather than installing an unconfirmed candidate.

Dynamic-shape, mixed-device, multi-GPU, input-mutating, and already wrapped
graphs currently stay on ordinary Inductor. A raw single-GPU CUDA wrapper stays
eligible whether it embeds Triton or generated CUDA templates, calls external
libraries, or combines them. Inductor's FX and generated-code caches remain
enabled.

## What the agent does

Each uncached graph gets an agent.

Each `tilesmith.compile()` call receives a unique run ID. When an eligible graph
is first invoked, it keeps five rolling source files within that run:

```text
~/.cache/tilesmith/runs/<run-id>/<cache-key>/attempts/
  attempt_00.py
  attempt_01.py
  attempt_02.py
  attempt_03.py
  attempt_04.py
```

`<cache-key>` is the 64-character SHA-256 identity of one generated graph and
its optimization environment. A single compile run may contain several such
graphs, such as forward, backward, or graph-break regions.

Every attempt starts with the captured PyTorch FX program as comments, followed
by one canonical Inductor-generated Python wrapper for that graph. The wrapper
contains its `call(args)` entry point, allocations, streams, launches, embedded
Triton or generated-template source when Inductor emits it, and external-library
call sites. Library implementations, weights, PTX, cubins, and shared objects
remain runtime dependencies or cache artifacts rather than being copied into
the attempt. Forward, backward, graph-break, and shape-specialized graphs each
receive their own track instead of being merged into one model-sized file.

A new turn copies the most recently evaluated source. After `attempt_04.py`,
the oldest contents are discarded and the other attempts shift down. Complete
host reports are retained beside the attempts under `profiles/`. Set
`TILESMITH_DEBUG=1` to also retain per-turn structured agent traces. An exact
artifact-cache hit does not rerun the agent; its fresh run track contains
`cache.json`, which identifies the reused artifact.

An agent final answer ends only that agent turn. The outer LangGraph does not
finish a requested loop until the host:

1. compares the candidate with the original Inductor wrapper on normal, zero,
   and high-magnitude inputs, including output strides;
2. measures alternating paired CUDA-event samples;
3. observes at least `TILESMITH_MIN_IMPROVEMENT`, which defaults to `0.02`;
4. repeats correctness and timing in a fresh confirmation; and
5. sees the confirmation clear the same speedup threshold.
