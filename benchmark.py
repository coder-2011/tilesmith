import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from tilesmith import (
    _ComputeSandbox,
    _default_skills,
    _evaluator_server,
    _read_compute_config,
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

    # evaluate uploads one candidate and runs the controlled correctness and timing path.
    def evaluate(self, source):
        self.worker.write_text("candidate.py", source)
        return self.worker.run("evaluate-fixture")


# _run creates one ComputeSDK evaluator, builds the controlled baseline, and drives every attempt.
def _run(args):
    if args.attempts < 2:
        raise ValueError(
            "--attempts must be at least 2 so this exercises an agent loop"
        )
    if args.launches < 2:
        raise ValueError("--launches must be at least 2")
    if args.expected_improvement < 0 or args.expected_improvement >= 1:
        raise ValueError("--expected-improvement must be in [0, 1)")
    if not args.chip.strip():
        raise ValueError("--chip must not be empty")
    if not args.config:
        raise ValueError("--config or CLIPROXY_CONFIG is required")

    workload_source = (ROOT / "workload.py").read_text(encoding="utf-8")
    skills = args.skill if args.skill is not None else _default_skills()
    compute = _read_compute_config()

    with _ComputeSandbox(compute, args.chip) as worker:
        worker.write_text("workload.py", workload_source)
        extracted = worker.run("extract-fixture")
        source = extracted.pop("source")
        baseline_source = _slow_fixture(source, args.launches)
        print(
            json.dumps(
                {"fixture_launches": args.launches, **extracted},
                sort_keys=True,
            )
        )

        evaluator = _FixtureEvaluator(worker)
        with tempfile.TemporaryDirectory(prefix="tilesmith-benchmark-") as directory:
            kernel_path = Path(directory) / "generated.py"
            kernel_path.write_text(baseline_source, encoding="utf-8")
            with _evaluator_server(evaluator) as (url, token):
                evaluate_command = shlex.join(
                    [
                        sys.executable,
                        str(ROOT / "tilesmith.py"),
                        "_evaluate",
                        url,
                        token,
                    ]
                )
                command = [
                    "go",
                    "run",
                    ".",
                    "-kernel",
                    str(kernel_path),
                    "-evaluate",
                    evaluate_command,
                    "-config",
                    args.config,
                    "-model",
                    args.model,
                    "-chip",
                    evaluator.chip,
                    "-iterations",
                    str(args.attempts),
                    "-min-improvement",
                    "0.05",
                    "-timeout",
                    "20m",
                ]
                for skill in skills:
                    command.extend(["-skill", skill])
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = []
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.append(line)
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"agent exited with status {return_code}")

    agent_output = "".join(output)
    attempts = re.findall(r"^attempt \d+:", agent_output, flags=re.MULTILINE)
    searches = re.findall(
        r"^research \d+: (\d+) web search calls$",
        agent_output,
        flags=re.MULTILINE,
    )
    installed = re.search(
        r"installed ([0-9.]+)% improvement: ([0-9.eE+-]+) -> ([0-9.eE+-]+)",
        agent_output,
    )
    if len(attempts) != args.attempts:
        raise RuntimeError(
            f"agent completed {len(attempts)} of {args.attempts} attempts"
        )
    search_count = sum(int(count) for count in searches)
    if search_count < 1:
        raise RuntimeError("agent did not use hosted web search")
    if installed is None:
        raise RuntimeError("agent did not catch the controlled optimization")

    gain = float(installed.group(1)) / 100
    if gain < args.expected_improvement:
        raise RuntimeError(
            f"measured gain {gain:.1%} is below required "
            f"{args.expected_improvement:.1%}"
        )
    print(
        json.dumps(
            {
                "baseline_score": float(installed.group(2)),
                "confirmed_score": float(installed.group(3)),
                "improvement": gain,
                "model_attempts": len(attempts),
                "skill_files": len(skills),
                "web_searches": search_count,
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
        "run", help="run the complete multi-attempt benchmark"
    )
    run_parser.add_argument("--attempts", type=int, default=4)
    run_parser.add_argument("--chip", default="H100")
    run_parser.add_argument("--config", default=os.environ.get("CLIPROXY_CONFIG", ""))
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
