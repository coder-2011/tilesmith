# Tilesmith

Tilesmith asks an LLM to improve a generated Triton Python file, but installs a
candidate only after it passes a user-owned correctness check and produces a
confirmed benchmark improvement.

The first version is intentionally a post-processing command, not a new
TorchInductor backend. Point it at the generated file after the normal backend
has emitted Triton:

```sh
go run . \
  -kernel generated.py \
  -check 'python verify.py' \
  -bench 'python bench.py' \
  -config /path/to/cliproxy-config.yaml \
  -model your-model
```

Both Python commands receive the candidate path in `TILESMITH_KERNEL`.
`-check` must exit nonzero for any incorrect candidate. `-bench` should warm up,
use GPU timing, and print one finite positive lower-is-better score on its last
line. Tilesmith runs it three times and compares medians by default.

Use `-iterations 0` to verify the Python commands and measure only the baseline.
`CLIPROXY_CONFIG`, `TILESMITH_MODEL`, and `TILESMITH_PROXY_KEY` provide the
corresponding flag/config defaults.
