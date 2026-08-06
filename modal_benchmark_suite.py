"""Run the Tilesmith workload suite on one Modal L4 GPU and save local charts."""

import json
from pathlib import Path

import modal

from benchmark_suite import write_artifacts

ROOT = Path(__file__).resolve().parent
SKILL_ROOT = Path.home() / ".codex" / "skills"
CUDA_SKILL_ROOT = SKILL_ROOT / "high-performance-cuda"
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "deepagents>=0.7.1,<0.8", "langchain-openai>=1.4.1,<2")
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_file(ROOT / "benchmark_suite.py", "/root/benchmark_suite.py")
    .add_local_file(SKILL_ROOT / "check-normal" / "SKILL.md", "/root/.codex/skills/check-normal/SKILL.md")
    .add_local_file(SKILL_ROOT / "high-performance-triton" / "SKILL.md", "/root/.codex/skills/high-performance-triton/SKILL.md")
    .add_local_file(Path.home() / "Documents/kernel-design-agents/skills/KernelWiki/SKILL.md", "/root/Documents/kernel-design-agents/skills/KernelWiki/SKILL.md")
    .add_local_file(CUDA_SKILL_ROOT / "SKILL.md", "/root/.codex/skills/high-performance-cuda/SKILL.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/core-principles.md", "/root/.codex/skills/high-performance-cuda/references/core-principles.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/autotuning-and-benchmarking.md", "/root/.codex/skills/high-performance-cuda/references/autotuning-and-benchmarking.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/numerical-correctness.md", "/root/.codex/skills/high-performance-cuda/references/numerical-correctness.md")
    .add_local_file(CUDA_SKILL_ROOT / "references/triton-and-codegen.md", "/root/.codex/skills/high-performance-cuda/references/triton-and-codegen.md")
)
APP = modal.App("tilesmith-workload-benchmarks")
SECRET = modal.Secret.from_local_environ(["OPENROUTER_API_KEY"])


# run_suite places every workload and loop count on one shared L4 software stack.
@APP.function(
    image=IMAGE,
    gpu="L4",
    timeout=14_400,
    env={"PYTHONPATH": "/root/src"},
    secrets=[SECRET],
)
def run_suite(agent_model, loops, max_model_calls):
    """Execute the CUDA suite and return its JSON-serializable evidence."""
    from benchmark_suite import run_suite as run_cuda_suite

    return json.dumps(run_cuda_suite(agent_model, loops=tuple(loops), max_model_calls=max_model_calls), sort_keys=True)


# _parse_loops validates the local CLI's compact comma-separated loop list.
def _parse_loops(value):
    """Convert a comma-separated positive-integer list into an immutable tuple."""
    try:
        loops = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("loops must be comma-separated positive integers") from error
    if not loops or any(loop < 1 for loop in loops):
        raise ValueError("loops must be comma-separated positive integers")
    return loops


# main submits the remote suite and creates charts on the caller's filesystem.
@APP.local_entrypoint()
def main(
    agent_model: str = "openai/gpt-5.5",
    loops: str = "1,2,3",
    max_model_calls: int = 24,
    output_dir: str = "benchmark-results",
):
    """Run the requested sweep and persist its raw evidence and SVG charts locally."""
    result = json.loads(run_suite.remote(agent_model, _parse_loops(loops), max_model_calls))
    write_artifacts(result, output_dir)
    print(f"wrote {Path(output_dir).resolve()}", flush=True)
