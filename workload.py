import torch


ATOL = 2e-3
RTOL = 2e-3
WARMUP = 25
BENCH_ITERATIONS = 100
BENCH_SAMPLES = 7


# model is the eager PyTorch reference that TorchInductor will lower to Triton.
def model(x):
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + 1e-5)
    return normalized.to(x.dtype)


# make_inputs returns the fixed-shape input used for extraction and timing.
def make_inputs(device):
    torch.manual_seed(0)
    x = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    return (x,)


# make_cases adds zero and high-magnitude inputs without changing the compiled shape.
def make_cases(device):
    (x,) = make_inputs(device)
    return [(x,), (torch.zeros_like(x),), (x * 32,)]
