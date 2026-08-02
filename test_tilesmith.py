import hashlib
import json
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

import tilesmith


class _ToolModel(FakeMessagesListChatModel):
    """Provide deterministic tool calls while exercising the real Deep Agents graph."""

    # bind_tools accepts the graph's built-in and Tilesmith tools without changing the script.
    def bind_tools(self, tools: Sequence[Any], **kwargs):
        return self


class _Evaluator:
    """Return deterministic scores for host-policy and graph integration tests."""

    chip = "NVIDIA H100 (sm_90)"

    # evaluate makes numbered and explicitly optimized candidates progressively faster.
    def evaluate(self, source):
        if "optimized" in source:
            return {"score": 0.5}
        if source.startswith("candidate-"):
            return {"score": 1 - int(source.split("-")[1]) / 100}
        return {"score": 1.0}


class TilesmithAgentTests(unittest.TestCase):
    """Cover the graph boundary and the host-owned benchmark invariants."""

    # test_track_bounds_history_and_reuses_results checks the costly stateful edge cases together.
    def test_track_bounds_history_and_reuses_results(self):
        evaluator = _Evaluator()
        track = tilesmith._BenchmarkTrack("baseline\n", {"score": 1.0}, evaluator, 7)

        for attempt in range(1, 8):
            track.benchmark(f"candidate-{attempt}")

        self.assertEqual(track.benchmarks, 7)
        self.assertEqual(len(track.recent), 5)
        oldest_hash = hashlib.sha256(b"candidate-3\n").hexdigest()
        self.assertEqual(track.recent[0]["source_sha256"], oldest_hash)

        cached = json.loads(track.benchmark("candidate-7"))
        exhausted = json.loads(track.benchmark("candidate-8"))
        self.assertTrue(cached["cached"])
        self.assertIn("budget exhausted", exhausted["error"])
        self.assertEqual(track.benchmarks, 7)

    # test_real_graph_installs_measured_source proves Deep Agents executes the benchmark tool loop.
    def test_real_graph_installs_measured_source(self):
        optimized = 'def call(args):\n    return "optimized"\n'
        model = _ToolModel(
            responses=[
                AIMessage(
                    content=[
                        {
                            "type": "server_tool_call",
                            "name": "web_search",
                            "id": "search-1",
                            "args": {"query": "Triton H100 optimization"},
                        }
                    ],
                    tool_calls=[
                        {
                            "name": "benchmark_candidate",
                            "args": {"source": optimized},
                            "id": "benchmark-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        settings = tilesmith._Settings(
            cache_dir=Path("/tmp"),
            chip="",
            compute=None,
            iterations=2,
            min_improvement=0.01,
            model="test",
            sandbox="local",
            skills=(),
            timeout=10.0,
        )

        with patch.object(tilesmith, "_chat_model", return_value=model):
            result = tilesmith._run_agent(
                "def call(args):\n    return args\n",
                settings,
                _Evaluator(),
            )

        self.assertEqual(result.source, optimized)
        self.assertEqual(result.benchmarks, 1)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.web_searches, 1)
        self.assertEqual(result.confirmed_score, 0.5)


if __name__ == "__main__":
    unittest.main()
