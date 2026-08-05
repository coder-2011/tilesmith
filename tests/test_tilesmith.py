"""Exercise Tilesmith's host-owned graph, cache, and sandbox invariants."""

import asyncio
import contextlib
import json
import sys
import tempfile
import unittest
import warnings
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar
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
        """Preserve PyTorch configuration while installing the inner compiler."""
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
        torch_module._TorchCompileInductorWrapper = Mock(return_value=Mock(config={"max_autotune": True}))
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
        backend = tilesmith._Backend(settings=None, dynamic=False)

        with patch.dict(sys.modules, modules):
            result = backend("graph", ("input",), mode="max-autotune", options={"shape_padding": True})

        self.assertEqual(result, "compiled")
        self.assertEqual(calls[0][:2], ("graph", ("input",)))
        self.assertEqual(calls[0][2]["config_patches"], {"max_autotune": True})
        self.assertIs(calls[0][2]["inner_compile"].__self__, backend)
        torch_module._TorchCompileInductorWrapper.assert_called_once_with("max-autotune", {"shape_padding": True}, False)
        self.assertEqual(patches, [{"enable_autograd_cache": False}])

    # test_backend_wraps_external_cuda_source keeps non-Triton Inductor wrappers eligible.
    def test_backend_wraps_external_cuda_source(self):
        """Keep external-library CUDA wrappers eligible for optimization."""
        compiled_type = type("CompiledFxGraph", (), {})
        compiled = compiled_type()
        compiled._fx_graph_cache_key = "graph-key"
        compiled.cache_key = "code-key"
        compiled.current_callable = Mock(name="inductor_call")
        compiled.device_idxs = {0}
        compiled.device_types = {"cuda"}
        compiled.mutated_input_idxs = ()
        compiled.source_code = "from torch._inductor.select_algorithm import extern_kernels\ndef call(args):\n    return extern_kernels.mm(*args)\n"

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
        """Render each agency policy and reject unknown levels."""
        low = Mock(agency="low", loops=1, max_model_calls=None, skills=())
        medium = Mock(agency="medium", loops=1, max_model_calls=None, skills=())
        high = Mock(agency="high", loops=1, max_model_calls=None, skills=())

        low_prompt = tilesmith._agent_instructions(low, _Evaluator(), ())
        medium_prompt = tilesmith._agent_instructions(medium, _Evaluator(), ())
        high_prompt = tilesmith._agent_instructions(high, _Evaluator(), ())

        self.assertIn("Keep the existing wrapper", low_prompt)
        self.assertIn("Do not fuse, split, add, or remove kernels", low_prompt)
        self.assertIn("## Workspace contract", low_prompt)
        self.assertIn("only local file you may edit", low_prompt)
        self.assertIn("## Turn lifecycle", low_prompt)
        self.assertIn("## GPU workspace (NVIDIA H100 (sm_90))", low_prompt)
        self.assertIn("persistent NVIDIA H100 (sm_90) ComputeSDK sandbox", low_prompt)
        self.assertIn("candidate.py` is the only remote file you may edit", low_prompt)
        self.assertIn("Do not modify, delete, or rely on any of them", low_prompt)
        self.assertIn("kernel fusion or decomposition", medium_prompt)
        self.assertIn("Preserve the high-level algorithm", medium_prompt)
        self.assertIn("fuse, split, add, or remove kernels", high_prompt)
        self.assertEqual(tilesmith._read_settings(None, "").agency, "medium")
        with self.assertRaisesRegex(ValueError, "agency must be 'low', 'medium'"):
            tilesmith._read_settings(None, "", agency="unbounded")

    # test_local_prompt_omits_gpu_workspace keeps the default no-sandbox path free of remote instructions.
    def test_local_prompt_omits_gpu_workspace(self):
        """Omit GPU-shell instructions when no sandbox evaluator is available."""
        settings = Mock(agency="low", loops=1, max_model_calls=None)
        evaluator = Mock(execution="the local CUDA device", chip="NVIDIA L4", spec=[])
        evaluator.shell_help = None

        prompt = tilesmith._agent_instructions(settings, evaluator, ())

        self.assertNotIn("## GPU workspace", prompt)
        self.assertNotIn("`gpu_shell`", prompt)
        self.assertNotIn("candidate.py` is the only remote file you may edit", prompt)

    # test_skill_snapshot_stabilizes_prompt protects cache identity from concurrent file edits.
    def test_skill_snapshot_stabilizes_prompt(self):
        """Keep prompts tied to the skill snapshot used by the cache key."""
        settings = Mock(agency="low", loops=1, max_model_calls=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILL.md"
            path.write_text("snapshot-before-edit", encoding="utf-8")
            skills = tilesmith._load_skills((path,))
            path.write_text("contents-after-edit", encoding="utf-8")
            prompt = tilesmith._agent_instructions(settings, _Evaluator(), skills)

        self.assertIn("snapshot-before-edit", prompt)
        self.assertNotIn("contents-after-edit", prompt)

    # test_track_enforces_one_new_benchmark_per_loop checks the costly stateful edge cases together.
    def test_track_enforces_one_new_benchmark_per_loop(self):
        """Enforce one fresh accepted measurement per requested loop."""
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
        """Latch ordinary Inductor after an optimization failure."""
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

    # test_warning_promotion_keeps_fallback_latched covers a failure after state publication.
    def test_warning_promotion_keeps_fallback_latched(self):
        """Avoid repeating optimization when warnings are promoted to errors."""
        baseline = Mock(return_value="baseline")
        kernel = tilesmith._LazyKernel(baseline, "source", {}, None, "graph-key")

        with (
            patch.object(kernel, "_install", side_effect=RuntimeError("offline")) as install,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                kernel(["first"])
            result = kernel(["second"])

        self.assertEqual(result, "baseline")
        self.assertEqual(install.call_count, 1)
        baseline.assert_called_with(["second"])

    # test_attempt_workspace_rotates_five_files checks source lineage and reference repair together.
    def test_attempt_workspace_rotates_five_files(self):
        """Rotate five attempts while preserving their trusted reference header."""
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
            bodies = [tilesmith._split_reference(path.read_text())[1] for path in attempts]
            self.assertEqual(
                [path.name for path in attempts],
                [f"attempt_{i:02d}.py" for i in range(5)],
            )
            self.assertEqual(bodies, [f"candidate-{i}\n" for i in range(1, 6)])
            self.assertTrue(all(path.read_text().startswith(tilesmith.REFERENCE_START) for path in attempts))

    # test_run_tracks_separate_evidence_from_shared_artifacts covers repeated identical compiles.
    def test_run_tracks_separate_evidence_from_shared_artifacts(self):
        """Record per-run evidence for one shared accepted artifact."""
        source = "def call(args):\n    return args\n"
        cache_key = "same-graph"
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            artifact = tilesmith._write_cache(cache_dir, cache_key, source)
            tilesmith._record_cache_hit(cache_dir, "first", cache_key, artifact.path)
            tilesmith._record_cache_hit(cache_dir, "second", cache_key, artifact.path)

            first = tilesmith._run_graph_directory(cache_dir, "first", cache_key)
            second = tilesmith._run_graph_directory(cache_dir, "second", cache_key)
            first_record = json.loads((first / "cache.json").read_text())
            second_record = json.loads((second / "cache.json").read_text())
            cached_source = artifact.source.decode()

        self.assertNotEqual(first, second)
        self.assertEqual(first_record["status"], "cache_hit")
        self.assertEqual(second_record["artifact"], "artifacts/same-graph.py")
        self.assertEqual(cached_source, source)

    # test_gpu_shell_round_trips_candidate protects the local/remote edit boundary.
    def test_gpu_shell_round_trips_candidate(self):
        """Round-trip only the active candidate through the GPU shell."""
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )
        remote_source = source.replace("return args", "return optimized")
        active_source = source.replace("return args", "return active")
        worker = Mock()
        worker.shell.return_value = {"exit_code": 0, "stdout": "profile complete"}
        worker.read_text.return_value = remote_source

        with tempfile.TemporaryDirectory() as directory:
            workspace = tilesmith._AttemptWorkspace(Path(directory), source)
            workspace.prepare_next()
            workspace.write_latest(active_source)
            workspace.read_candidate("/attempts/attempt_00.py")
            report = tilesmith._run_gpu_shell(worker, workspace, "ncu ...", 600)
            synced_source = workspace.read_latest()

        worker.write_text.assert_called_once_with("workbench/candidate.py", active_source)
        worker.shell.assert_called_once_with("ncu ...", 600)
        self.assertEqual(synced_source, remote_source)
        self.assertIn("copied to local attempt", report["candidate_sync"])

    # test_compute_shell_returns_worker_report keeps benchmark evidence inside gpu_shell.
    def test_compute_shell_returns_worker_report(self):
        """Return a worker report alongside an arbitrary GPU-shell command."""
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
            patch.object(sandbox, "read_text", return_value=worker_result) as read_text,
        ):
            report = sandbox.shell("python3 worker.py compare", 30)

        read_text.assert_called_once_with(f"{tilesmith.REMOTE_WORKSPACE}/result.json")
        self.assertEqual(report["worker_result"], {"correct": True, "score": 0.5})

    # test_compute_sandbox_exit_orders_teardown protects the paid resource boundary.
    def test_compute_sandbox_exit_orders_teardown(self):
        """Destroy paid resources before closing their event-loop owner."""
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

    # test_compute_config_resolves_official_providers protects the one-name provider path.
    def test_compute_config_resolves_official_providers(self):
        """Resolve official providers without TileSmith-specific credential translation."""
        with patch.dict("os.environ", {"COMPUTESDK_PROVIDER": "e2b"}, clear=True):
            config = tilesmith._read_compute_config("L4")

        self.assertEqual(config.provider_name, "e2b")
        self.assertEqual(config.provider.package, "@computesdk/e2b")
        self.assertEqual(config.provider.factory, "e2b")
        self.assertEqual(config.provider.version, "1.7.52")
        self.assertEqual(config.provider.config, {})

        with (
            patch.dict(
                "os.environ",
                {
                    "COMPUTESDK_PROVIDER": "acme",
                    "COMPUTESDK_PROVIDER_PACKAGE": "acme-compute",
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "version must be explicit"),
        ):
            tilesmith._read_compute_config("L4")

    # test_compute_bridge_accepts_large_responses protects generated-source and trace reads.
    def test_compute_bridge_accepts_large_responses(self):
        """Configure the bridge for bounded large generated-source responses."""
        from tilesmith.sandboxes.computesdk import _BRIDGE_RESPONSE_LIMIT_BYTES, _Bridge

        # exercise captures the subprocess options without starting npm or a provider.
        async def exercise():
            process = Mock()
            process.returncode = 0
            process.stdout = asyncio.StreamReader()
            process.stderr = asyncio.StreamReader()
            bridge = _Bridge((), "priority", True, 1.0)
            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    new=AsyncMock(return_value=process),
                ) as create_process,
                patch.object(
                    bridge,
                    "call",
                    new=AsyncMock(return_value={"providers": []}),
                ),
            ):
                await bridge.start()
            self.assertEqual(
                create_process.await_args.kwargs["limit"],
                _BRIDGE_RESPONSE_LIMIT_BYTES,
            )
            process.stdout.feed_eof()
            process.stderr.feed_eof()
            await bridge._reader_task
            await bridge._stderr_task

        asyncio.run(exercise())

    # test_compute_bridge_rejects_ambiguous_values covers protocol safety edges together.
    def test_compute_bridge_rejects_ambiguous_values(self):
        """Reject ambiguous JSON values and invalid request deadlines."""
        from tilesmith.sandboxes.computesdk import (
            _Bridge,
            _json_bytes,
            _prepare_redactions,
            _redact,
            _redact_value,
        )

        redactions = _prepare_redactions(("abc", "abcdef", "api key/value"))
        value = {"api+key%2Fvalue": ["abcdef"]}
        self.assertEqual(_redact_value(value, redactions), {"[REDACTED]": ["[REDACTED]"]})
        with self.assertRaisesRegex(ValueError, "JSON serializable"):
            _json_bytes({"value": float("nan")})

        with patch.dict("os.environ", {"E2B_API_KEY": "provider-secret"}, clear=True):
            bridge = _Bridge((), "priority", True, 1.0)
        self.assertEqual(_redact("leaked provider-secret", bridge._redactions), "leaked [REDACTED]")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            asyncio.run(bridge.call("noop", timeout=0))

    # test_sandbox_destroy_survives_caller_cancellation protects paid-resource teardown.
    def test_sandbox_destroy_survives_caller_cancellation(self):
        """Share one teardown task when its first waiter is cancelled."""
        from tilesmith.sandboxes.computesdk import Sandbox

        async def exercise():
            started = asyncio.Event()
            finish = asyncio.Event()
            bridge = Mock()

            async def destroy(*_args, **_kwargs):
                started.set()
                await finish.wait()

            bridge.call = AsyncMock(side_effect=destroy)
            sandbox = Sandbox(bridge, "sandbox-1", "test")
            first = asyncio.create_task(sandbox.destroy())
            await started.wait()
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(sandbox.destroy())
            finish.set()
            await second
            self.assertTrue(sandbox._destroyed)
            bridge.call.assert_awaited_once()

        asyncio.run(exercise())

    # test_real_graph_installs_measured_file proves LangGraph retries an unbenchmarked loop.
    def test_real_graph_installs_measured_source(self):
        """Install the exact candidate confirmed by the real graph loop."""
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
                            "args": {"command": "cp /attempts/attempt_01.py execute-copy.py"},
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
                    (),
                )
            attempts = sorted((Path(directory) / "track" / "attempts").glob("*.py"))
            profiles = Path(directory) / "track" / "profiles"
            reports = sorted(profiles.glob("*.json"))
            first_trace = (profiles / "trace_00.json").read_text()
            second_trace = (profiles / "trace_01.json").read_text()
            execute_copy_exists = (Path(directory) / "track" / "execute-copy.py").is_file()

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
        """Stop before the provider call beyond the explicit shared budget."""
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
                tilesmith._run_agent(source, settings, _Evaluator(), track, ())
            final_trace = (track / "profiles" / "trace_01.json").read_text()

        self.assertIn('"model_call_limit_reached": true', final_trace)
        self.assertIn('"model_calls": 1', final_trace)


if __name__ == "__main__":
    unittest.main()
