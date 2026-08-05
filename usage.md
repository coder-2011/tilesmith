# Usage

Tilesmith wraps `torch.compile`. Give it a model and representative CUDA inputs,
then use the returned model normally.

Start with an environment that already has CUDA PyTorch and Triton:

```sh
python3 -m pip install tilesmith-ai
export OPENAI_API_KEY=...
```

```python
import torch
import tilesmith

model = torch.nn.Linear(1024, 1024).cuda().eval()
x = torch.randn(32, 1024, device="cuda")

optimized_model = tilesmith.compile(model, example_inputs=(x,))
y = optimized_model(x)
```

Optimization happens lazily on the first call. If Tilesmith cannot confirm a
correct speedup, it keeps the normal TorchInductor result.

The main optional settings are `agency="low" | "medium" | "high"`, `loops`,
and `max_model_calls`. Their defaults are `"medium"`, `1`, and no model-call
limit.
