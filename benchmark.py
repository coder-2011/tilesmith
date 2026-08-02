import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from tilesmith import (
    _ComputeSandbox,
    _default_skills,
    _read_compute_config,
    _read_settings,
    _run_agent,
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
    """Expose the controlled remote workload through the agent's benchmark tool."""

    # __init__ retains the warm sandbox and its observed chip description.
    def __init__(self, worker):
        self.worker = worker
        self.chip = worker.chip
        self.execution = "one warm ComputeSDK GPU sandbox"

    # evaluate uploads one candidate and runs the controlled correctness and timing path.
    def evaluate(self, source):
        self.worker.write_text("candidate.py", source)
        report = self.worker.run("evaluate-fixture")
        report["profiling"]["nsight_compute"] = self.worker.profile(
            "profile-fixture"
        )
        return report


# _run creates one ComputeSDK evaluator, builds the controlled baseline, and drives the graph.
def _run(args):
    if args.launches < 2:
        raise ValueError("--launches must be at least 2")
    if args.expected_improvement < 0 or args.expected_improvement >= 1:
        raise ValueError("--expected-improvement must be in [0, 1)")
    if not args.chip.strip():
        raise ValueError("--chip must not be empty")

    workload_source = (ROOT / "workload.py").read_text(encoding="utf-8")
    skills = args.skill if args.skill is not None else _default_skills()
    compute = _read_compute_config(args.chip)

    with _ComputeSandbox(compute) as worker:
        worker.write_text("workload.py", workload_source)
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
        settings = replace(
            _read_settings(None, "", args.loops),
            min_improvement=0.05,
            model=args.model,
            skills=tuple(skills),
        )
        track_id = hashlib.sha256(baseline_source.encode()).hexdigest()
        workspace = settings.cache_dir / "tracks" / f"benchmark-{track_id}"
        result = _run_agent(baseline_source, settings, evaluator, workspace)

    if result.model_calls < 2:
        raise RuntimeError("agent did not execute a multi-turn graph")
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
                "skill_files": len(skills),
            },
            sort_keys=True,
        )
    )


# _parse_args keeps the paid ComputeSDK benchmark explicit and configurable.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the opt-in Tilesmith H100 agent benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run", help="run the complete Deep Agents benchmark"
    )
    run_parser.add_argument("--loops", type=int, default=1)
    run_parser.add_argument("--chip", default="H100")
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
