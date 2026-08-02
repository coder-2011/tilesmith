import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar, Sequence
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage

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

    # evaluate makes numbered and explicitly optimized candidates progressively faster.
    def evaluate(self, source):
        if "optimized-2" in source:
            return {"profiling": {"events": ["kernel-2"]}, "score": 0.4}
        if "optimized" in source:
            return {"profiling": {"events": ["kernel-1"]}, "score": 0.5}
        if source.startswith("candidate-"):
            return {"score": 1 - int(source.split("-")[1]) / 100}
        return {"profiling": {"events": ["baseline"]}, "score": 1.0}


class TilesmithAgentTests(unittest.TestCase):
    """Cover the graph boundary and the host-owned benchmark invariants."""

    # test_track_enforces_one_new_benchmark_per_loop checks the costly stateful edge cases together.
    def test_track_enforces_one_new_benchmark_per_loop(self):
        evaluator = _Evaluator()
        track = tilesmith._BenchmarkTrack("baseline\n", {"score": 1.0}, evaluator, 2)

        track.start_loop(0)
        baseline = json.loads(track.benchmark("baseline"))
        measured = json.loads(track.benchmark("candidate-7"))
        blocked = json.loads(track.benchmark("candidate-8"))
        track.start_loop(1)
        cached = json.loads(track.benchmark("candidate-7"))
        second = json.loads(track.benchmark("candidate-8"))

        self.assertTrue(baseline["cached"])
        self.assertEqual(baseline["progress"]["status"], "needs_new_candidate")
        self.assertAlmostEqual(measured["score"], 0.93)
        self.assertIn("already benchmarked", blocked["error"])
        self.assertTrue(cached["cached"])
        self.assertAlmostEqual(second["score"], 0.92)
        self.assertEqual(second["progress"]["percent"], 100)
        self.assertEqual(track.benchmarks, 2)

    # test_history_keeps_five_benchmarks proves unrelated tools do not consume the retention budget.
    def test_history_keeps_five_benchmarks(self):
        messages = []
        for attempt in range(6):
            benchmark_id = f"benchmark-{attempt}"
            read_id = f"read-{attempt}"
            messages.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "benchmark_candidate",
                                "args": {"path": f"/attempts/attempt_{attempt:02d}.py"},
                                "id": benchmark_id,
                                "type": "tool_call",
                            },
                            {
                                "name": "read_file",
                                "args": {"path": "/kernel.py"},
                                "id": read_id,
                                "type": "tool_call",
                            },
                        ],
                    ),
                    ToolMessage(
                        content=f"report-{attempt}",
                        name="benchmark_candidate",
                        tool_call_id=benchmark_id,
                    ),
                    ToolMessage(
                        content="kernel source",
                        name="read_file",
                        tool_call_id=read_id,
                    ),
                ]
            )

        tilesmith._BenchmarkHistoryEdit().apply(messages, count_tokens=lambda _: 0)

        benchmark_results = messages[1::3]
        read_results = messages[2::3]
        self.assertEqual(
            [message.content for message in benchmark_results],
            ["[cleared]", *[f"report-{attempt}" for attempt in range(1, 6)]],
        )
        self.assertEqual(messages[0].tool_calls[0]["args"], {})
        self.assertTrue(
            all(message.content == "kernel source" for message in read_results)
        )

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
                _, annotated = workspace.read_candidate(path)
                if attempt < 5:
                    path = workspace.prepare_next(annotated)

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

    # test_real_graph_installs_measured_file proves LangGraph retries an unbenchmarked loop.
    def test_real_graph_installs_measured_source(self):
        _ToolModel.seen_tools.clear()
        optimized = 'def call(args):\n    return "optimized"\n'
        optimized_2 = 'def call(args):\n    return "optimized-2"\n'
        model = _ToolModel(
            responses=[
                AIMessage(content="still planning"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {
                                "command": "bash -lc 'test -f attempts/attempt_01.py'"
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
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "benchmark_candidate",
                            "args": {"path": "/attempts/attempt_01.py"},
                            "id": "benchmark-1",
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
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "benchmark_candidate",
                            "args": {"path": "/attempts/attempt_02.py"},
                            "id": "benchmark-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        settings = tilesmith._Settings(
            cache_dir=Path("/tmp"),
            compute=None,
            loops=2,
            min_improvement=0.01,
            model="test",
            skills=(),
            timeout=10.0,
        )
        source = tilesmith._with_reference(
            "def forward(self, x):\n    return x",
            "def call(args):\n    return args\n",
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tilesmith, "_chat_model", return_value=model):
                result = tilesmith._run_agent(
                    source,
                    settings,
                    _Evaluator(),
                    Path(directory) / "track",
                )
            attempts = sorted((Path(directory) / "track" / "attempts").glob("*.py"))
            reports = sorted((Path(directory) / "track" / "profiles").glob("*.json"))

        self.assertIn(optimized_2, result.source)
        self.assertEqual(result.benchmarks, 2)
        self.assertEqual(result.model_calls, 8)
        self.assertEqual(result.confirmed_score, 0.4)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(
            [path.name for path in reports],
            ["baseline.json", "loop_00.json", "loop_01.json"],
        )
        self.assertTrue(
            {"execute", "web_search", "write_file"} <= _ToolModel.seen_tools
        )


if __name__ == "__main__":
    unittest.main()
