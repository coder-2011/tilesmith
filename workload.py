"""Define the fixed RMS-normalization workload used by the GPU benchmark."""

import torch

ATOL = 2e-3
RTOL = 2e-3
SHAPE = (4096, 4096)
DTYPE = "float16"


# model is the eager PyTorch reference that TorchInductor will lower to Triton.
def model(x):
    """Return the eager RMS-normalization reference output."""
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + 1e-5)
    return normalized.to(x.dtype)


# make_inputs returns the fixed-shape input used for extraction and timing.
def make_inputs(device):
    """Create the deterministic fixed-shape benchmark input."""
    torch.manual_seed(0)
    x = torch.randn(SHAPE, device=device, dtype=getattr(torch, DTYPE))
    return (x,)


# make_cases adds zero and high-magnitude inputs without changing the compiled shape.
def make_cases(device):
    """Create normal, zero, and high-magnitude correctness cases."""
    (x,) = make_inputs(device)
    return [(x,), (torch.zeros_like(x),), (x * 32,)]
