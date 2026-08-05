import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from tilesmith import (
    _ComputeSandbox,
    _default_skills,
    _load_skills,
    _read_compute_config,
    _read_settings,
    _run_gpu_shell,
    _run_agent,
    _wait_for_model,
    _with_reference,
)


ROOT = Path(__file__).resolve().parent


# _slow_fixture repeats one pure generated launch to create a correct, obvious smoke-test win.
def _slow_fixture(source, launches):
    lines = source.splitlines(keepends=True)
    launch_lines = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("triton_") and ".run(" in line
    ]
    if len(launch_lines) != 1:
        raise RuntimeError(
            f"expected one generated Triton launch, found {len(launch_lines)}"
        )
    index = launch_lines[0]
    lines[index : index + 1] = [lines[index]] * launches
    return "".join(lines)


class _FixtureEvaluator:
    """Expose the controlled remote workload and its open GPU workbench."""

    # __init__ retains the warm sandbox and its observed chip description.
    def __init__(self, worker):
        self.worker = worker
        self.chip = worker.chip
        self.execution = "one warm ComputeSDK GPU sandbox"
        self.shell_calls = 0

    # evaluate uploads one candidate and runs the controlled correctness and timing path.
    def evaluate(self, source):
        self.worker.write_text("candidate.py", source)
        return self.worker.run("evaluate-fixture")

    # shell gives the agent open access to the fixture's warm ComputeSDK sandbox.
    def shell(self, command, workspace, timeout_seconds):
        self.shell_calls += 1
        return _run_gpu_shell(self.worker, workspace, command, timeout_seconds)

    shell_help = (
        "The GPU shell starts in /tmp/tilesmith/workbench. candidate.py, workload.py, "
        "worker.py, measurement.py, request.json, cache/, and artifacts/ are "
        "available. `python3 worker.py evaluate-fixture` runs the fixture check "
        "and timing. `python3 worker.py run-fixture` invokes the candidate once; "
        "wrap that command with any profiler or measurement tool you need."
    )


# _run creates one ComputeSDK evaluator, builds the controlled baseline, and drives the graph.
def _run(args):
    if args.launches < 2:
        raise ValueError("--launches must be at least 2")
    if (
        not math.isfinite(args.expected_improvement)
        or not 0 <= args.expected_improvement < 1
    ):
        raise ValueError("--expected-improvement must be in [0, 1)")
    if not args.chip.strip():
        raise ValueError("--chip must not be empty")

    workload_source = (ROOT / "workload.py").read_text(encoding="utf-8")
    skill_paths = args.skill if args.skill is not None else _default_skills()
    compute = _read_compute_config(args.chip)
    settings = replace(
        _read_settings(None, "", args.loops, args.max_model_calls),
        min_improvement=0.05,
        model=args.model,
        skills=tuple(skill_paths),
    )
    skills = _load_skills(settings.skills)
    _wait_for_model(settings.model)

    with _ComputeSandbox(compute) as worker:
        worker.write_text("workload.py", workload_source)
        worker.seed_workspace("workload.py")
        extracted = worker.run("extract-fixture")
        source = extracted.pop("source")
        source = _with_reference(workload_source, source)
        baseline_source = _slow_fixture(source, args.launches)
        print(
            json.dumps(
                {"fixture_launches": args.launches, **extracted},
                sort_keys=True,
            )
        )

        evaluator = _FixtureEvaluator(worker)
        track_id = hashlib.sha256(baseline_source.encode()).hexdigest()
        # Paid smoke tests must prove a new agent run instead of inheriting an old fix.
        run_id = f"benchmark-{track_id}-{time.time_ns()}"
        workspace = settings.cache_dir / "tracks" / run_id
        result = _run_agent(baseline_source, settings, evaluator, workspace, skills)

    if result.model_calls < 2:
        raise RuntimeError("agent did not execute a multi-turn graph")
    if evaluator.shell_calls < 1:
        raise RuntimeError("agent did not use the GPU shell")
    if result.benchmarks != args.loops:
        raise RuntimeError(
            f"graph completed {result.benchmarks} benchmarks for {args.loops} loops"
        )
    if result.source == baseline_source:
        raise RuntimeError("agent did not catch the controlled optimization")

    gain = (result.baseline_score - result.confirmed_score) / result.baseline_score
    if gain < args.expected_improvement:
        raise RuntimeError(
            f"measured gain {gain:.1%} is below required "
            f"{args.expected_improvement:.1%}"
        )
    print(
        json.dumps(
            {
                "baseline_score": result.baseline_score,
                "confirmed_score": result.confirmed_score,
                "improvement": gain,
                "loops": args.loops,
                "model_calls": result.model_calls,
                "gpu_shell_calls": evaluator.shell_calls,
                "skill_files": len(skills),
            },
            sort_keys=True,
        )
    )


# _parse_args keeps the paid ComputeSDK benchmark explicit and configurable.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the opt-in Tilesmith GPU agent benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run", help="run the complete Deep Agents benchmark"
    )
    run_parser.add_argument("--loops", type=int, default=1)
    run_parser.add_argument("--max-model-calls", type=int)
    run_parser.add_argument("--chip", default="L4")
    run_parser.add_argument("--expected-improvement", type=float, default=0.25)
    run_parser.add_argument("--launches", type=int, default=4)
    run_parser.add_argument(
        "--model", default=os.environ.get("TILESMITH_MODEL", "gpt-5.5")
    )
    run_parser.add_argument(
        "--skill",
        action="append",
        help="trusted skill or reference file; repeatable",
    )
    return parser.parse_args()


# main runs the explicit paid benchmark command.
def main():
    _run(_parse_args())


if __name__ == "__main__":
    main()
