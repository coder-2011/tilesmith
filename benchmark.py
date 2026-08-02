import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "tilesmith"
FUNCTION_NAME = "run_gpu"


# _default_skills finds the user's relevant Codex guidance without making it a repository dependency.
def _default_skills():
    skill_root = Path.home() / ".codex" / "skills"
    cuda_root = skill_root / "high-performance-cuda"
    candidates = [
        skill_root / "check-normal" / "SKILL.md",
        cuda_root / "SKILL.md",
        cuda_root / "references" / "core-principles.md",
        cuda_root / "references" / "autotuning-and-benchmarking.md",
        cuda_root / "references" / "numerical-correctness.md",
        cuda_root / "references" / "triton-and-codegen.md",
    ]
    return [str(path) for path in candidates if path.is_file()]


# _remote invokes the deployed H100 evaluator without embedding Modal credentials.
def _remote(profile, action, workload_source, kernel_source):
    os.environ["MODAL_PROFILE"] = profile
    import modal

    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    return function.remote(action, workload_source, kernel_source)


# _deploy publishes the current evaluator while leaving its H100 scaled to zero when idle.
def _deploy(profile):
    modal_command = shutil.which("modal")
    if modal_command is None:
        raise RuntimeError("modal is not installed")
    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = profile
    subprocess.run(
        [modal_command, "deploy", str(ROOT / "modal_app.py")],
        cwd=ROOT,
        env=environment,
        check=True,
    )


# _slow_fixture repeats one pure generated launch to create a correct, obvious smoke-test win.
def _slow_fixture(source, launches):
    lines = source.splitlines(keepends=True)
    launch_lines = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("triton_") and ".run(" in line
    ]
    if len(launch_lines) != 1:
        raise RuntimeError(f"expected one generated Triton launch, found {len(launch_lines)}")
    index = launch_lines[0]
    lines[index : index + 1] = [lines[index]] * launches
    return "".join(lines)


# _evaluate checks and times the candidate path supplied by the Go agent.
def _evaluate(args):
    kernel_path = os.environ.get("TILESMITH_KERNEL")
    if not kernel_path:
        raise RuntimeError("TILESMITH_KERNEL is required for evaluation")
    workload_source = Path(args.workload).read_text(encoding="utf-8")
    kernel_source = Path(kernel_path).read_text(encoding="utf-8")
    result = _remote(args.profile, "evaluate", workload_source, kernel_source)
    print(json.dumps(result, sort_keys=True))
    print(f"{result['score']:.9g}")


# _run deploys one evaluator, creates the controlled baseline, and drives every agent attempt.
def _run(args):
    if args.attempts < 2:
        raise ValueError("--attempts must be at least 2 so this exercises an agent loop")
    if args.launches < 2:
        raise ValueError("--launches must be at least 2")
    if args.expected_improvement < 0 or args.expected_improvement >= 1:
        raise ValueError("--expected-improvement must be in [0, 1)")
    if not args.config:
        raise ValueError("--config or CLIPROXY_CONFIG is required")

    workload_path = Path(args.workload).resolve()
    workload_source = workload_path.read_text(encoding="utf-8")
    skills = args.skill if args.skill is not None else _default_skills()
    _deploy(args.profile)
    extracted = _remote(args.profile, "extract", workload_source, "")
    source = extracted.pop("source")
    baseline_source = _slow_fixture(source, args.launches)
    print(json.dumps({"fixture_launches": args.launches, **extracted}, sort_keys=True))

    with tempfile.TemporaryDirectory(prefix="tilesmith-benchmark-") as directory:
        kernel_path = Path(directory) / "generated.py"
        kernel_path.write_text(baseline_source, encoding="utf-8")
        evaluate_command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "evaluate",
                "--profile",
                args.profile,
                "--workload",
                str(workload_path),
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
            "NVIDIA H100 (sm_90)",
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
        searches = re.findall(r"^research \d+: (\d+) web search calls$", agent_output, flags=re.MULTILINE)
        installed = re.search(
            r"installed ([0-9.]+)% improvement: ([0-9.eE+-]+) -> ([0-9.eE+-]+)",
            agent_output,
        )
        if len(attempts) != args.attempts:
            raise RuntimeError(f"agent completed {len(attempts)} of {args.attempts} attempts")
        search_count = sum(int(count) for count in searches)
        if search_count < 1:
            raise RuntimeError("agent did not use hosted web search")
        if installed is None:
            raise RuntimeError("agent did not catch the controlled optimization")

        gain = float(installed.group(1)) / 100
        if gain < args.expected_improvement:
            raise RuntimeError(
                f"measured gain {gain:.1%} is below required {args.expected_improvement:.1%}"
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


# _parse_args keeps the paid benchmark separate from the evaluator subprocess mode.
def _parse_args():
    parser = argparse.ArgumentParser(description="Run the opt-in Tilesmith H100 agent benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run the complete multi-attempt benchmark")
    run_parser.add_argument("--attempts", type=int, default=4)
    run_parser.add_argument("--config", default=os.environ.get("CLIPROXY_CONFIG", ""))
    run_parser.add_argument("--expected-improvement", type=float, default=0.25)
    run_parser.add_argument("--launches", type=int, default=4)
    run_parser.add_argument("--model", default=os.environ.get("TILESMITH_MODEL", "gpt-5.5"))
    run_parser.add_argument("--profile", default=os.environ.get("MODAL_PROFILE", "tilesmith-h100"))
    run_parser.add_argument("--skill", action="append", help="trusted skill or reference file; repeatable")
    run_parser.add_argument("--workload", default=str(ROOT / "workload.py"))

    evaluate_parser = commands.add_parser("evaluate", help="evaluate one agent candidate")
    evaluate_parser.add_argument("--profile", required=True)
    evaluate_parser.add_argument("--workload", required=True)
    return parser.parse_args()


# main dispatches the explicit paid run or one agent-requested evaluation.
def main():
    args = _parse_args()
    if args.command == "run":
        _run(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
