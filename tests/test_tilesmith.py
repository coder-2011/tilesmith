import asyncio
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Sequence
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

import tilesmith


class _ToolModel(FakeMessagesListChatModel):
    """Provide deterministic tool calls while exercising the real Deep Agents graph."""

    seen_tools: ClassVar[set[str]] = set()

    # bind_tools accepts the graph's built-in and Tilesmith tools without changing the script.
    def bind_tools(self, tools: Sequence[Any], **kwargs):
        for configured in tools:
            if isinstance(configured, dict):
                self.seen_tools.add(configured.get("type", ""))
            else:
                self.seen_tools.add(getattr(configured, "name", ""))
        return self


class _Evaluator:
    """Return deterministic scores for host-policy and graph integration tests."""

    chip = "NVIDIA H100 (sm_90)"
    execution = "a deterministic test evaluator"

    # __init__ records arbitrary GPU shell commands without executing them.
    def __init__(self):
        self.shell_commands = []

    # evaluate makes numbered and explicitly optimized candidates progressively faster.
    def evaluate(self, source):
        if "broken" in source:
            raise RuntimeError("candidate did not compile")
        if "optimized-2" in source:
            return {"score": 0.4}
        if "optimized" in source:
            return {"score": 0.5}
        if source.startswith("candidate-"):
            return {"score": 1 - int(source.split("-")[1]) / 100}
        return {"score": 1.0}

    # shell exercises the real agent tool boundary with deterministic output.
    def shell(self, command, workspace, timeout_seconds):
        self.shell_commands.append((command, timeout_seconds))
        return {"exit_code": 0, "stdout": "NVIDIA H100"}

    shell_help = "An unrestricted deterministic H100 shell is available."


class TilesmithAgentTests(unittest.TestCase):
    """Cover the graph boundary and the host-owned benchmark invariants."""

    # test_backend_preserves_pytorch_configuration keeps Tilesmith from disabling Inductor features.
    def test_backend_preserves_pytorch_configuration(self):
        calls = []
        patches = []

        # fake_compile_fx records the exact boundary without requiring PyTorch on the test host.
        def fake_compile_fx(graph_module, example_inputs, **kwargs):
            calls.append((graph_module, example_inputs, kwargs))
            return "compiled"

        # fake_config_patch records the one cache setting needed to keep the hook reachable.
        @contextlib.contextmanager
        def fake_config_patch(**kwargs):
            patches.append(kwargs)
            yield

        torch_module = ModuleType("torch")
        functorch_module = ModuleType("torch._functorch")
        functorch_config_module = ModuleType("torch._functorch.config")
        functorch_config_module.patch = fake_config_patch
        functorch_module.config = functorch_config_module
        inductor_module = ModuleType("torch._inductor")
        compile_module = ModuleType("torch._inductor.compile_fx")
        compile_module.compile_fx = fake_compile_fx
        modules = {
            "torch": torch_module,
            "torch._functorch": functorch_module,
            "torch._functorch.config": functorch_config_module,
            "torch._inductor": inductor_module,
            "torch._inductor.compile_fx": compile_module,
        }
        backend = tilesmith._Backend(settings=None)

        with patch.dict(sys.modules, modules):
            result = backend("graph", ("input",))

        self.assertEqual(result, "compiled")
        self.assertEqual(calls[0][:2], ("graph", ("input",)))
        self.assertEqual(set(calls[0][2]), {"inner_compile"})
        self.assertIs(calls[0][2]["inner_compile"].__self__, backend)
        self.assertEqual(patches, [{"enable_autograd_cache": False}])

    # test_backend_wraps_external_cuda_source keeps non-Triton Inductor wrappers eligible.
    def test_backend_wraps_external_cuda_source(self):
        compiled_type = type("CompiledFxGraph", (), {})
        compiled = compiled_type()
        compiled._fx_graph_cache_key = "graph-key"
        compiled.cache_key = "code-key"
        compiled.current_callable = Mock(name="inductor_call")
        compiled.device_idxs = {0}
        compiled.device_types = {"cuda"}
        compiled.mutated_input_idxs = ()
        compiled.source_code = (
            "from torch._inductor.select_algorithm import extern_kernels\n"
            "def call(args):\n"
            "    return extern_kernels.mm(*args)\n"
        )

        torch_module = ModuleType("torch")
        torch_module.SymInt = type("SymInt", (), {})
        torch_module.is_tensor = Mock(return_value=False)
        compile_module = ModuleType("torch._inductor.compile_fx")
        compile_module.compile_fx_inner = Mock(return_value=compiled)
        output_module = ModuleType("torch._inductor.output_code")
        output_module.CompiledFxGraph = compiled_type
        constants = Mock()
        constants.unwrap.return_value = {}
        output_module.CompiledFxGraphConstantsWithGm = Mock(return_value=constants)
        modules = {
            "torch": torch_module,
            "torch._inductor.compile_fx": compile_module,
            "torch._inductor.output_code": output_module,
        }

        backend = tilesmith._Backend(Mock(compute=None))
        graph_module = Mock(code="def forward(self, x):\n    return x")
        with (
            patch.dict(sys.modules, modules),
            patch.object(tilesmith, "_is_raw_generated_call", return_value=True),
        ):
            result = backend._inner_compile(graph_module, [])

        self.assertIs(result, compiled)
        self.assertIsInstance(compiled.current_callable, tilesmith._LazyKernel)
        self.assertIn("extern_kernels.mm", compiled.current_callable.source)
        self.assertIn(tilesmith.REFERENCE_START, compiled.current_callable.source)

    # test_agency_controls_prompt_scope keeps the public policy explicit and rejects ambiguous levels.
    def test_agency_controls_prompt_scope(self):
        low = Mock(agency="low", loops=1, max_model_calls=None, skills=())
        medium = Mock(agency="medium", loops=1, max_model_calls=None, skills=())
        high = Mock(agency="high", loops=1, max_model_calls=None, skills=())

        low_prompt = tilesmith._agent_instructions(low, _Evaluator())
        medium_prompt = tilesmith._agent_instructions(medium, _Evaluator())
        high_prompt = tilesmith._agent_instructions(high, _Evaluator())

        self.assertIn("Keep the existing wrapper", low_prompt)
        self.assertIn("Do not fuse, split, add, or remove kernels", low_prompt)
        self.assertIn("One turn works as follows", low_prompt)
        self.assertIn("candidate.py` is the only editable remote file", low_prompt)
        self.assertIn("kernel fusion or decomposition", medium_prompt)
        self.assertIn("Preserve the high-level algorithm", medium_prompt)
        self.assertIn("fuse, split, add, or remove kernels", high_prompt)
        self.assertEqual(tilesmith._read_settings(None, "").agency, "medium")
        with self.assertRaisesRegex(ValueError, "agency must be 'low', 'medium'"):
            tilesmith._read_settings(None, "", agency="unbounded")

    # test_track_enforces_one_new_benchmark_per_loop checks the costly stateful edge cases together.
    def test_track_enforces_one_new_benchmark_per_loop(self):
        evaluator = _Evaluator()
        track = tilesmith._BenchmarkTrack(
            "baseline\n",
            {"reference": "torch_inductor_generated", "score": 1.05},
            evaluator,
            2,
            0.01,
        )

        track.start_loop(0)
        baseline = track.benchmark("baseline")
        rejected = track.benchmark("broken")
        insufficient = track.benchmark("candidate-0")
        measured = track.benchmark("candidate-7")
        measured["caller_only"] = True
        blocked = track.benchmark("candidate-8")
        track.start_loop(1)
        cached = track.benchmark("candidate-7")
        second = track.benchmark("candidate-8")

        self.assertTrue(baseline["cached"])
        self.assertEqual(baseline["score"], 1.0)
        self.assertEqual(baseline["progress"]["status"], "needs_new_candidate")
        self.assertFalse(rejected["correct"])
        self.assertEqual(rejected["progress"]["status"], "needs_new_candidate")
        self.assertFalse(insufficient["accepted"])
        self.assertEqual(insufficient["progress"]["status"], "needs_speedup")
        self.assertAlmostEqual(measured["score"], 0.93)
        self.assertIn("already benchmarked", blocked["error"])
        self.assertTrue(cached["cached"])
        self.assertNotIn("caller_only", cached)
        self.assertAlmostEqual(second["score"], 0.92)
        self.assertEqual(second["progress"]["percent"], 100)
        self.assertEqual(track.benchmarks, 2)

    # test_optimizer_failure_falls_back keeps optional agent infrastructure out of model semantics.
    def test_optimizer_failure_falls_back(self):
        calls = []
        owner = Mock()

        # baseline records the boxed input that ordinary Inductor receives.
        def baseline(inputs):
            calls.append(inputs)
            return "baseline"

        kernel = tilesmith._LazyKernel(
            baseline,
            "source",
            {},
            None,
            "graph-key",
            owner=owner,
        )
        owner.current_callable = kernel
        with (
            patch.object(kernel, "_install", side_effect=RuntimeError("offline")),
            self.assertWarnsRegex(UserWarning, "using ordinary Inductor"),
        ):
            result = kernel(["input"])

        self.assertEqual(result, "baseline")
        self.assertEqual(calls, [["input"]])
        self.assertIs(kernel.installed, baseline)
        self.assertIs(owner.current_callable, baseline)

    # test_attempt_workspace_rotates_five_files checks source lineage and reference repair together.
    def test_attempt_workspace_rotates_five_files(self):
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = tilesmith._AttemptWorkspace(Path(directory), source)
            path = workspace.prepare_next()
            for attempt in range(1, 6):
                candidate = workspace.directory / path.lstrip("/")
                candidate.write_text(f"candidate-{attempt}\n", encoding="utf-8")
                workspace.read_candidate(path)
                if attempt < 5:
                    path = workspace.prepare_next()

            attempts = sorted((Path(directory) / "attempts").glob("*.py"))
            bodies = [
                tilesmith._split_reference(path.read_text())[1] for path in attempts
            ]
            self.assertEqual(
                [path.name for path in attempts],
                [f"attempt_{i:02d}.py" for i in range(5)],
            )
            self.assertEqual(bodies, [f"candidate-{i}\n" for i in range(1, 6)])
            self.assertTrue(
                all(
                    path.read_text().startswith(tilesmith.REFERENCE_START)
                    for path in attempts
                )
            )

    # test_run_tracks_separate_evidence_from_shared_artifacts covers repeated identical compiles.
    def test_run_tracks_separate_evidence_from_shared_artifacts(self):
        source = "def call(args):\n    return args\n"
        cache_key = "same-graph"
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            artifact = tilesmith._write_cache(cache_dir, cache_key, source)
            tilesmith._record_cache_hit(cache_dir, "first", cache_key, artifact)
            tilesmith._record_cache_hit(cache_dir, "second", cache_key, artifact)

            first = tilesmith._run_graph_directory(cache_dir, "first", cache_key)
            second = tilesmith._run_graph_directory(cache_dir, "second", cache_key)
            first_record = json.loads((first / "cache.json").read_text())
            second_record = json.loads((second / "cache.json").read_text())
            cached_source = artifact.read_text()

        self.assertNotEqual(first, second)
        self.assertEqual(first_record["status"], "cache_hit")
        self.assertEqual(second_record["artifact"], "artifacts/same-graph.py")
        self.assertEqual(cached_source, source)

    # test_gpu_shell_round_trips_candidate protects the local/remote edit boundary.
    def test_gpu_shell_round_trips_candidate(self):
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )
        remote_source = source.replace("return args", "return optimized")
        worker = Mock()
        worker.shell.return_value = {"exit_code": 0, "stdout": "profile complete"}
        worker.read_text.return_value = remote_source

        with tempfile.TemporaryDirectory() as directory:
            workspace = tilesmith._AttemptWorkspace(Path(directory), source)
            workspace.prepare_next()
            report = tilesmith._run_gpu_shell(worker, workspace, "ncu ...", 600)
            synced_source = workspace.read_latest()

        worker.write_text.assert_called_once_with("workbench/candidate.py", source)
        worker.shell.assert_called_once_with("ncu ...", 600)
        self.assertEqual(synced_source, remote_source)
        self.assertIn("copied to local attempt", report["candidate_sync"])

    # test_compute_shell_returns_worker_report keeps benchmark evidence inside gpu_shell.
    def test_compute_shell_returns_worker_report(self):
        sandbox = object.__new__(tilesmith._ComputeSandbox)
        sandbox.command_timeout_ms = 60_000
        command_result = Mock(
            duration_ms=12.5,
            exit_code=0,
            stderr="",
            stdout='{"result_bytes": 31}\n',
        )
        worker_result = '{"correct": true, "score": 0.5}'

        with (
            patch.object(sandbox, "_command", return_value=command_result),
            patch.object(
                sandbox, "read_text", return_value=worker_result
            ) as read_text,
        ):
            report = sandbox.shell("python3 worker.py compare", 30)

        read_text.assert_called_once_with(f"{tilesmith.REMOTE_WORKSPACE}/result.json")
        self.assertEqual(report["worker_result"], {"correct": True, "score": 0.5})

    # test_compute_sandbox_exit_orders_teardown protects the paid resource boundary.
    def test_compute_sandbox_exit_orders_teardown(self):
        events = []
        sandbox = object.__new__(tilesmith._ComputeSandbox)
        sandbox.sandbox = object()
        sandbox.sandbox_scope = type("Scope", (), {})()
        sandbox.client = type("Client", (), {})()
        sandbox.sandbox_scope.__aexit__ = Mock(return_value="sandbox-destroyed")
        sandbox.client.__aexit__ = Mock(return_value="client-closed")
        sandbox.runner = Mock()
        sandbox.runner.run.side_effect = lambda operation: events.append(operation) or operation
        sandbox.runner.close.side_effect = lambda: events.append("runner-closed")

        result = sandbox.__exit__(None, None, None)

        self.assertEqual(result, "sandbox-destroyed")
        self.assertEqual(events, ["sandbox-destroyed", "client-closed", "runner-closed"])

    # test_compute_bridge_accepts_large_responses protects generated-source and trace reads.
    def test_compute_bridge_accepts_large_responses(self):
        from tilesmith.sandboxes.computesdk import _BRIDGE_RESPONSE_LIMIT_BYTES, _Bridge

        # exercise captures the subprocess options without starting npm or a provider.
        async def exercise():
            process = Mock()
            process.returncode = 0
            process.stdout = asyncio.StreamReader()
            bridge = _Bridge((), "priority", True, 1.0)
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ) as create_process:
                with patch.object(
                    bridge,
                    "call",
                    new=AsyncMock(return_value={"providers": []}),
                ):
                    await bridge.start()
            self.assertEqual(
                create_process.await_args.kwargs["limit"],
                _BRIDGE_RESPONSE_LIMIT_BYTES,
            )
            process.stdout.feed_eof()
            await bridge._reader_task
            bridge._stderr.close()

        asyncio.run(exercise())

    # test_real_graph_installs_measured_file proves LangGraph retries an unbenchmarked loop.
    def test_real_graph_installs_measured_source(self):
        _ToolModel.seen_tools.clear()
        optimized = 'def call(args):\n    return "optimized"\n'
        optimized_2 = 'def call(args):\n    return "optimized-2"\n'
        model = _ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {
                                "command": "cp /attempts/attempt_01.py execute-copy.py"
                            },
                            "id": "execute-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "gpu_shell",
                            "args": {"command": "nvidia-smi", "timeout_seconds": 30},
                            "id": "gpu-shell-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/attempts/attempt_01.py",
                                "content": optimized,
                            },
                            "id": "write-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/attempts/attempt_02.py",
                                "content": optimized_2,
                            },
                            "id": "write-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        settings = tilesmith._Settings(
            agency="high",
            cache_dir=Path("/tmp"),
            compute=None,
            loops=2,
            max_model_calls=6,
            min_improvement=0.01,
            model="test",
            run_id="test-run",
            skills=(),
            timeout=10.0,
        )
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )

        evaluator = _Evaluator()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tilesmith, "_chat_model", return_value=model):
                result = tilesmith._run_agent(
                    source,
                    settings,
                    evaluator,
                    Path(directory) / "track",
                )
            attempts = sorted((Path(directory) / "track" / "attempts").glob("*.py"))
            profiles = Path(directory) / "track" / "profiles"
            reports = sorted(profiles.glob("*.json"))
            first_trace = (profiles / "trace_00.json").read_text()
            second_trace = (profiles / "trace_01.json").read_text()
            execute_copy_exists = (
                Path(directory) / "track" / "execute-copy.py"
            ).is_file()

        self.assertIn(optimized_2, result.source)
        self.assertEqual(result.benchmarks, 2)
        self.assertEqual(result.model_calls, 6)
        self.assertEqual(result.confirmed_score, 0.4)
        self.assertEqual(evaluator.shell_commands, [("nvidia-smi", 30)])
        self.assertTrue(execute_copy_exists)
        self.assertIn("execute-1", first_trace)
        self.assertNotIn("execute-1", second_trace)
        self.assertIn("write-2", second_trace)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(
            [path.name for path in reports],
            [
                "baseline.json",
                "loop_00.json",
                "loop_01.json",
                "trace_00.json",
                "trace_01.json",
            ],
        )
        self.assertTrue(
            {
                "execute",
                "gpu_shell",
                "read_exact_source",
                "web_search",
                "write_file",
            }
            <= _ToolModel.seen_tools
        )

    # test_model_call_limit_counts_provider_calls protects the public budget invariant.
    def test_model_call_limit_counts_provider_calls(self):
        model = _ToolModel(responses=[AIMessage(content="no candidate edit")])
        settings = tilesmith._Settings(
            agency="high",
            cache_dir=Path("/tmp"),
            compute=None,
            loops=1,
            max_model_calls=1,
            min_improvement=0.01,
            model="test",
            run_id="test-run",
            skills=(),
            timeout=10.0,
        )
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )

        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory) / "track"
            with (
                patch.object(tilesmith, "_chat_model", return_value=model),
                self.assertRaisesRegex(RuntimeError, "max_model_calls=1"),
            ):
                tilesmith._run_agent(source, settings, _Evaluator(), track)
            final_trace = (track / "profiles" / "trace_01.json").read_text()

        self.assertIn('"model_call_limit_reached": true', final_trace)
        self.assertIn('"model_calls": 1', final_trace)


if __name__ == "__main__":
    unittest.main()
