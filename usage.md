# Usage

Tilesmith wraps `torch.compile`. Give it a model and representative CUDA
inputs, then call the returned model normally.

## Install

Start with an environment that already has compatible CUDA builds of PyTorch
and Triton:

```sh
python3 -m pip install tilesmith-ai
export OPENAI_API_KEY=...
```

`OPENROUTER_API_KEY` also works. Set `OPENAI_BASE_URL` to use an
OpenAI-compatible gateway, and `TILESMITH_MODEL` to choose a model. The default
model is `gpt-5.5`.

## Compile a model

```python
import torch
import tilesmith

model = torch.nn.Linear(1024, 1024).cuda().eval()
x = torch.randn(32, 1024, device="cuda")

optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
)

y = optimized_model(x)
```

`example_inputs` must be a tuple containing at least one CUDA tensor. Use
representative shapes, dtypes, and devices: Tilesmith uses the first real call
to check correctness and measure candidates.

Optimization is lazy. `tilesmith.compile()` returns a normal
`torch.compile` callable, and PyTorch compiles it when you first invoke it.
Tilesmith then works on eligible Inductor-generated CUDA wrappers. It does not
rewrite the original model.

## Control the optimization

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    agency="medium",
    loops=1,
    max_model_calls=None,
    fullgraph=False,
    dynamic=None,
    mode=None,
    options=None,
)
```

- `agency="low"` limits changes to local kernel tuning.
- `agency="medium"` is the default. It may change wrapper logic, buffers, and
  kernel fusion while preserving the high-level algorithm.
- `agency="high"` allows broader algorithm and implementation changes while
  preserving the generated callable's behavior.
- `loops` is the number of confirmed improvements to accept. It defaults to
  `1`; rejected candidates do not consume a loop.
- `max_model_calls` sets an exact optional limit on model invocations. The
  default is `None`, meaning no limit.
- `fullgraph`, `dynamic`, `mode`, and `options` pass through to PyTorch.

Every accepted candidate must match the original wrapper and clear the minimum
speedup twice. Incorrect, slower, marginal, or interrupted attempts fall back
to ordinary TorchInductor.

Tilesmith currently optimizes raw single-GPU CUDA wrappers. Dynamic-shape,
mixed-device, multi-GPU, input-mutating, and already wrapped graphs stay on
ordinary Inductor.

## Use a remote GPU sandbox

Set `sandbox="computesdk"` to let the agent test kernels on a ComputeSDK GPU:

```sh
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

```python
optimized_model = tilesmith.compile(
    model,
    example_inputs=(x,),
    sandbox="computesdk",
    chip="L4",
)
```

Modal is the default ComputeSDK provider, but it does not have a separate
Tilesmith execution path. To use another official provider, select it and set
that provider's normal credentials. For example:

```sh
export COMPUTESDK_PROVIDER=e2b
export E2B_API_KEY=...
```

Tilesmith selects the matching pinned provider package. Custom providers can
override `COMPUTESDK_PROVIDER_PACKAGE`, `COMPUTESDK_PROVIDER_FACTORY`,
`COMPUTESDK_PROVIDER_VERSION`, and `COMPUTESDK_PROVIDER_CONFIG`.
`COMPUTESDK_TEMPLATE` overrides the provider's default template.

The accepted callable still runs in your local process. Your local CUDA device
must therefore match `chip`, and remote evaluation currently supports
`cuda:0`. One sandbox stays warm for the graph's optimization run and is
destroyed afterward.

## Environment settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `TILESMITH_MODEL` | `gpt-5.5` | Agent model |
| `TILESMITH_MIN_IMPROVEMENT` | `0.02` | Required fractional speedup |
| `TILESMITH_TIMEOUT` | `1200` | Optimization timeout in seconds |
| `TILESMITH_SANDBOX_TIMEOUT` | `3600` | ComputeSDK sandbox lifetime in seconds |
| `TILESMITH_CACHE_DIR` | `~/.cache/tilesmith` | Cache and run evidence directory |
| `TILESMITH_DEBUG` | unset | Set to `1` to retain complete per-turn agent traces |

## Cache and failures

Accepted implementations are stored under
`~/.cache/tilesmith/artifacts/`. A later compile with the same generated graph
and optimization settings reuses the artifact without rerunning the agent or
starting a sandbox. Run evidence is kept under `~/.cache/tilesmith/runs/`.

Set `TILESMITH_DEBUG=1` to save complete per-turn agent traces as
`profiles/trace_00.json`, `trace_01.json`, and so on beneath that run directory.
Traces can include generated source and tool arguments, so treat the cache
directory as private. Cache hits do not create traces because no agent runs.

If the agent, provider, validation, or timing step fails, Tilesmith warns and
keeps the ordinary Inductor implementation. It never installs an unconfirmed
candidate.

## Validate a checkout

The local test suite does not provision a paid GPU sandbox:

```sh
uv run python -m unittest discover -s tests -v
```

The end-to-end benchmark does provision the requested ComputeSDK GPU and may
incur provider charges:

```sh
python3 moe_benchmark.py run --chip L4 --model openai/gpt-5.5
```
