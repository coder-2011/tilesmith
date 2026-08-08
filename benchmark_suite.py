"""Benchmark Tilesmith on MoE, CNN, and explicit recurrent inference workloads."""

import copy
import dataclasses
import html
import json
import math
import statistics
from pathlib import Path


# _benchmark interleaves models to limit clock and cache-order bias in CUDA timings.
def _benchmark(torch, functions, inputs, warmup=25, iterations=100, samples=7):
    """Return median CUDA-event latency and paired samples for each callable."""
    with torch.inference_mode():
        for _ in range(warmup):
            for function in functions.values():
                function(*inputs)
        torch.cuda.synchronize()
        timings = {name: [] for name in functions}
        items = tuple(functions.items())
        for sample in range(samples):
            order = items if sample % 2 == 0 else tuple(reversed(items))
            for name, function in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    function(*inputs)
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) / iterations)
    if any(not math.isfinite(value) or value <= 0 for values in timings.values() for value in values):
        raise RuntimeError("benchmark returned a non-finite or non-positive timing")
    return {
        name: {"median_ms": statistics.median(values), "samples_ms": values}
        for name, values in timings.items()
    }


# _workloads builds the exact fixed-shape inference models used by every loop count.
def _workloads(torch):
    """Return named model factories and representative CUDA input factories."""
    nn = torch.nn
    functional = torch.nn.functional

    class Expert(nn.Module):
        """Apply one gated feed-forward expert to every input token."""

        def __init__(self, width, hidden):
            """Create the two projections owned by this expert."""
            super().__init__()
            self.up = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)

        def forward(self, inputs):
            """Transform tokens through the gated feed-forward expert."""
            return self.down(functional.gelu(self.up(inputs)))

    class TopTwoMoE(nn.Module):
        """Mix the two highest-probability dense expert outputs per token."""

        def __init__(self, width, hidden, expert_count):
            """Create the router and independently parameterized experts."""
            super().__init__()
            self.router = nn.Linear(width, expert_count, bias=False)
            self.experts = nn.ModuleList(Expert(width, hidden) for _ in range(expert_count))
            self.expert_count = expert_count

        def forward(self, inputs):
            """Route each token to two experts and combine their weighted outputs."""
            routing = functional.softmax(self.router(inputs).float(), dim=-1)
            values, indices = torch.topk(routing, k=2, dim=-1)
            selected = functional.one_hot(indices, self.expert_count)
            weights = (selected * values.unsqueeze(-1)).sum(dim=-2)
            outputs = torch.stack([expert(inputs) for expert in self.experts], dim=-2)
            return (outputs * weights.to(inputs.dtype).unsqueeze(-1)).sum(dim=-2)

    class MoETransformer(nn.Module):
        """Run a compact causal-attention Transformer block with top-two MoE."""

        def __init__(self):
            """Create the fixed 256-wide inference-only Transformer workload."""
            super().__init__()
            width = 256
            self.embedding = nn.Embedding(2_048, width)
            self.attention_norm = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width, bias=False)
            self.attention_out = nn.Linear(width, width, bias=False)
            self.moe_norm = nn.LayerNorm(width)
            self.moe = TopTwoMoE(width, 512, 4)
            self.output_norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, 2_048, bias=False)
            self.heads = 8

        def forward(self, token_ids):
            """Return causal next-token logits for the complete token sequence."""
            hidden = self.embedding(token_ids)
            batch, sequence, width = hidden.shape
            head_width = width // self.heads
            qkv = self.qkv(self.attention_norm(hidden))
            qkv = qkv.view(batch, sequence, 3, self.heads, head_width)
            query, key, value = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-1, -2)).float() / math.sqrt(head_width)
            causal = torch.ones(sequence, sequence, device=hidden.device, dtype=torch.bool).tril()
            probabilities = functional.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1).to(hidden.dtype)
            attended = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, sequence, width)
            hidden = hidden + self.attention_out(attended)
            hidden = hidden + self.moe(self.moe_norm(hidden))
            return self.lm_head(self.output_norm(hidden))

    class MnistCNN(nn.Module):
        """Classify MNIST-shaped images with a conventional two-convolution CNN."""

        def __init__(self):
            """Create the fixed-shape MNIST convolutional inference workload."""
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, images):
            """Return ten class logits for each MNIST-shaped image."""
            return self.classifier(self.features(images))

    class AttentionRNN(nn.Module):
        """Classify token sequences with an Inductor-visible recurrent attention model."""

        def __init__(self):
            """Create the explicit recurrent cell and learned attention pooling layers."""
            super().__init__()
            self.embedding = nn.Embedding(8_192, 256)
            self.input_projection = nn.Linear(256, 256)
            self.recurrent_projection = nn.Linear(256, 256, bias=False)
            self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
            self.classifier = nn.Linear(256, 4)

        def forward(self, token_ids):
            """Unroll a fixed recurrent dependency, then return four class logits."""
            embedded = self.input_projection(self.embedding(token_ids))
            hidden = torch.zeros_like(embedded[:, 0])
            states = []
            for step in range(embedded.shape[1]):
                hidden = torch.tanh(embedded[:, step] + self.recurrent_projection(hidden))
                states.append(hidden)
            states = torch.stack(states, dim=1)
            weights = functional.softmax(self.attention(states), dim=1)
            return self.classifier((states * weights).sum(dim=1))

    device = torch.device("cuda:0")
    return {
        "moe_transformer": (MoETransformer, lambda: (torch.randint(0, 2_048, (8, 128), device=device),)),
        "mnist_cnn": (MnistCNN, lambda: (torch.randn(256, 1, 28, 28, device=device),)),
        "attention_rnn": (AttentionRNN, lambda: (torch.randint(0, 8_192, (64, 32), device=device),)),
    }


# _run_case compiles one workload at one exact Tilesmith optimization-loop count.
def _run_case(torch, tilesmith, name, factory, inputs, agent_model, loops, max_model_calls):
    """Measure eager, TorchInductor, and a single Tilesmith configuration."""
    model = factory().cuda().eval()
    eager = copy.deepcopy(model)
    inductor = torch.compile(copy.deepcopy(model))
    with torch.inference_mode():
        eager_output = eager(*inputs)
        inductor_output = inductor(*inputs)
    torch.testing.assert_close(inductor_output, eager_output, atol=1e-4, rtol=1e-3)

    result = {"workload": name, "loops": loops, "tilesmith_error": None}
    try:
        optimized = tilesmith.compile(
            copy.deepcopy(model),
            example_inputs=inputs,
            agent_model=agent_model,
            loops=loops,
            max_model_calls=max_model_calls,
        )
        with torch.inference_mode():
            optimized_output = optimized(*inputs)
        torch.testing.assert_close(optimized_output, eager_output, atol=1e-4, rtol=1e-3)
        timings = _benchmark(
            torch,
            {"pytorch_base": eager, "torch_compile": inductor, "tilesmith": optimized},
            inputs,
        )
        result["tilesmith_benchmarks"] = [dataclasses.asdict(value) for value in tilesmith.benchmarks(optimized)]
        result["tilesmith_diagnostics"] = [dataclasses.asdict(value) for value in tilesmith.diagnostics(optimized)]
        result["tilesmith_run_id"] = getattr(optimized, "_tilesmith_run_id", None)
        # A callable can mix accepted graphs with ordinary-Inductor fallbacks, so report partial success explicitly.
        if result["tilesmith_benchmarks"]:
            result["tilesmith_status"] = "partial" if result["tilesmith_diagnostics"] else "optimized"
        elif any(value["kind"] == "failed" for value in result["tilesmith_diagnostics"]):
            result["tilesmith_status"] = "failed"
        else:
            result["tilesmith_status"] = "ineligible"
            if not result["tilesmith_diagnostics"]:
                result["tilesmith_diagnostics"].append(
                    {
                        "kind": "ineligible",
                        "message": "no eligible TorchInductor CUDA graph reached Tilesmith",
                    }
                )
        if result["tilesmith_status"] in {"failed", "ineligible"}:
            result["tilesmith_error"] = "; ".join(value["message"] for value in result["tilesmith_diagnostics"])
    except Exception as error:
        timings = _benchmark(torch, {"pytorch_base": eager, "torch_compile": inductor}, inputs)
        result["tilesmith_benchmarks"] = []
        result["tilesmith_diagnostics"] = []
        result["tilesmith_status"] = "error"
        result["tilesmith_error"] = f"{type(error).__name__}: {error}"
    result["timings"] = timings
    base_ms = timings["pytorch_base"]["median_ms"]
    inductor_ms = timings["torch_compile"]["median_ms"]
    result["torch_compile_vs_pytorch_base"] = base_ms / inductor_ms
    if result["tilesmith_status"] in {"optimized", "partial"}:
        tilesmith_ms = timings["tilesmith"]["median_ms"]
        result["tilesmith_vs_pytorch_base"] = base_ms / tilesmith_ms
        result["tilesmith_vs_torch_compile"] = inductor_ms / tilesmith_ms
    else:
        result["tilesmith_vs_pytorch_base"] = None
        result["tilesmith_vs_torch_compile"] = None
    return result


# run_suite keeps all workloads on one GPU so software, driver, and clocks are shared.
def run_suite(agent_model, loops=(1, 2, 3), max_model_calls=24):
    """Run every workload and return JSON-serializable timing evidence."""
    import torch

    import tilesmith

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_suite requires a CUDA GPU")
    if not loops or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in loops):
        raise ValueError("loops must contain positive integers")
    if max_model_calls is not None and (not isinstance(max_model_calls, int) or max_model_calls < 1):
        raise ValueError("max_model_calls must be None or a positive integer")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.benchmark = True
    results = []
    for name, (factory, input_factory) in _workloads(torch).items():
        for loop_count in loops:
            inputs = input_factory()
            print(f"benchmarking {name} with loops={loop_count}", flush=True)
            results.append(_run_case(torch, tilesmith, name, factory, inputs, agent_model, loop_count, max_model_calls))
    return {
        "agent_model": agent_model,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "max_model_calls": max_model_calls,
        "results": results,
    }


# _svg writes one standalone SVG document from already-escaped plot shapes.
def _svg(width, height, shapes):
    """Return one compact SVG document with a white background and common font."""
    return "\n".join(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}.small{font-size:12px}.title{font-size:20px;font-weight:700}</style>',
            *shapes,
            "</svg>",
        )
    )


# _write_latency_chart renders one normalized grouped-bar panel for each workload.
def _write_latency_chart(results, output):
    """Write the eager, TorchInductor, and Tilesmith latency comparison chart."""
    colors = {"pytorch_base": "#94a3b8", "torch_compile": "#2563eb", "tilesmith": "#16a34a"}
    names = tuple(dict.fromkeys(value["workload"] for value in results))
    grouped = {name: [value for value in results if value["workload"] == name] for name in names}
    shapes = ['<text x="40" y="38" class="title">Latency by workload and optimization loops</text>']
    shapes.extend(
        f'<rect x="{40 + index * 255}" y="54" width="12" height="12" fill="{colors[key]}"/><text x="{58 + index * 255}" y="65" class="small">{key.replace("_", " ")}</text>'
        for index, key in enumerate(colors)
    )
    for panel, name in enumerate(names):
        values = sorted(grouped[name], key=lambda value: value["loops"])
        left = 50 + panel * 355
        top = 125
        bottom = 405
        width = 285
        finite = [timing["median_ms"] for value in values for timing in value["timings"].values()]
        ceiling = max(finite, default=1) * 1.12
        shapes.append(f'<text x="{left}" y="105" class="title">{html.escape(name.replace("_", " "))}</text>')
        shapes.append(f'<line x1="{left}" y1="{bottom}" x2="{left + width}" y2="{bottom}" stroke="#64748b"/>')
        shapes.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#64748b"/>')
        for tick in range(5):
            value = ceiling * tick / 4
            y = bottom - (bottom - top) * tick / 4
            shapes.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            shapes.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')
        for group, value in enumerate(values):
            center = left + 52 + group * 92
            shapes.append(f'<text x="{center}" y="430" text-anchor="middle" class="small">loops={value["loops"]}</text>')
            for offset, key in enumerate(colors):
                if key == "tilesmith" and value.get("tilesmith_status") not in {"optimized", "partial"}:
                    continue
                timing = value["timings"].get(key)
                if timing is None:
                    continue
                latency = timing["median_ms"]
                height = (bottom - top) * latency / ceiling
                x = center - 30 + offset * 21
                y = bottom - height
                shapes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="18" height="{height:.1f}" fill="{colors[key]}" rx="2"/>')
                shapes.append(f'<text x="{x + 9:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="small">{latency:.2f}</text>')
    output.write_text(_svg(1_140, 455, shapes), encoding="utf-8")


# _write_speedup_chart renders Tilesmith's measured speedup against TorchInductor.
def _write_speedup_chart(results, output):
    """Write the Tilesmith-versus-TorchInductor speedup chart."""
    names = tuple(dict.fromkeys(value["workload"] for value in results))
    colors = dict(zip(names, ("#7c3aed", "#ea580c", "#0891b2"), strict=False))
    left, top, right, bottom = 90, 95, 830, 410
    finite = [value["tilesmith_vs_torch_compile"] for value in results if value["tilesmith_vs_torch_compile"] is not None]
    if not finite:
        shapes = [
            '<text x="40" y="38" class="title">Tilesmith speedup over TorchInductor</text>',
            '<text x="40" y="92" class="small">No Tilesmith candidate was produced; fallback timings are excluded.</text>',
        ]
        output.write_text(_svg(920, 130, shapes), encoding="utf-8")
        return
    ceiling = max([1.0, *finite], default=1.0) * 1.08
    shapes = ['<text x="40" y="38" class="title">Tilesmith speedup over TorchInductor</text>']
    shapes.extend(
        f'<rect x="{90 + index * 235}" y="54" width="12" height="12" fill="{colors[name]}"/><text x="{108 + index * 235}" y="65" class="small">{name.replace("_", " ")}</text>'
        for index, name in enumerate(colors)
    )
    shapes.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#64748b"/>')
    shapes.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#64748b"/>')
    for tick in range(5):
        value = ceiling * tick / 4
        y = bottom - (bottom - top) * tick / 4
        shapes.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        shapes.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.2f}x</text>')
    for loop in (1, 2, 3):
        x = left + (right - left) * (loop - 1) / 2
        shapes.append(f'<text x="{x:.1f}" y="435" text-anchor="middle" class="small">loops={loop}</text>')
    for name, color in colors.items():
        values = sorted((value for value in results if value["workload"] == name and value["tilesmith_vs_torch_compile"] is not None), key=lambda value: value["loops"])
        points = []
        for value in values:
            x = left + (right - left) * (value["loops"] - 1) / 2
            y = bottom - (bottom - top) * value["tilesmith_vs_torch_compile"] / ceiling
            points.append((x, y, value["tilesmith_vs_torch_compile"]))
        if points:
            shapes.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, speedup in points:
            shapes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            shapes.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="small">{speedup:.2f}x</text>')
    output.write_text(_svg(920, 460, shapes), encoding="utf-8")


# write_artifacts persists raw measurements before generating derivative SVG charts.
def write_artifacts(result, output_dir):
    """Write the benchmark JSON and two standalone SVG summaries."""
    for value in result["results"]:
        if "tilesmith_status" not in value:
            value["tilesmith_status"] = "optimized" if value.get("tilesmith_benchmarks") else "fallback"
            if value["tilesmith_status"] == "fallback":
                value["tilesmith_vs_pytorch_base"] = None
                value["tilesmith_vs_torch_compile"] = None
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_latency_chart(result["results"], destination / "latency.svg")
    _write_speedup_chart(result["results"], destination / "speedup.svg")
