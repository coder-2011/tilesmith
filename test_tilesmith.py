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

    # test_track_reuses_results_and_enforces_budget checks the costly stateful edge cases together.
    def test_track_reuses_results_and_enforces_budget(self):
        evaluator = _Evaluator()
        track = tilesmith._BenchmarkTrack("baseline\n", {"score": 1.0}, evaluator, 1)

        measured = json.loads(track.benchmark("candidate-7"))
        cached = json.loads(track.benchmark("candidate-7"))
        exhausted = json.loads(track.benchmark("candidate-8"))
        self.assertAlmostEqual(measured["score"], 0.93)
        self.assertTrue(cached["cached"])
        self.assertIn("budget exhausted", exhausted["error"])
        self.assertEqual(track.benchmarks, 1)

    # test_real_graph_installs_measured_source proves Deep Agents executes the benchmark tool loop.
    def test_real_graph_installs_measured_source(self):
        optimized = 'def call(args):\n    return "optimized"\n'
        model = _ToolModel(
            responses=[
                AIMessage(
                    content="",
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
            benchmark_budget=2,
            cache_dir=Path("/tmp"),
            chip="",
            compute=None,
            min_improvement=0.01,
            model="test",
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
        self.assertEqual(result.confirmed_score, 0.5)


if __name__ == "__main__":
    unittest.main()
