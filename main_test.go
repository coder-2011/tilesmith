package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestPropose verifies hosted search, strict benchmark tools, and exact-source enforcement.
func TestPropose(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			t.Errorf("unexpected path %q", request.URL.Path)
		}
		var payload map[string]any
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		tools, ok := payload["tools"].([]any)
		if !ok || len(tools) != 2 {
			t.Fatalf("missing agent tools: %#v", payload["tools"])
		}
		function, ok := tools[1].(map[string]any)
		if !ok || function["name"] != "benchmark_candidate" || function["strict"] != true {
			t.Fatalf("benchmark tool is not strict: %#v", tools[1])
		}

		count := requests.Add(1)
		response.Header().Set("Content-Type", "application/json")
		switch count {
		case 1:
			choice, ok := payload["tool_choice"].(map[string]any)
			if !ok || choice["type"] != "web_search" {
				t.Fatalf("first turn did not require web search: %#v", payload["tool_choice"])
			}
			fmt.Fprint(response, `{"output":[{"type":"web_search_call"},{"type":"function_call","call_id":"call-1","name":"benchmark_candidate","arguments":"{\"source\":\"def call(args):\\n    return args\"}"}]}`)
		case 2:
			inputs, ok := payload["input"].([]any)
			if !ok {
				t.Fatalf("missing response input: %#v", payload["input"])
			}
			foundOutput := false
			for _, value := range inputs {
				item, ok := value.(map[string]any)
				if ok && item["type"] == "function_call_output" && item["call_id"] == "call-1" {
					foundOutput = true
				}
			}
			if !foundOutput {
				t.Fatalf("benchmark result was not returned to the model: %#v", inputs)
			}
			fmt.Fprint(response, `{"output":[{"type":"message","content":[{"type":"output_text","text":"def call(args):\n    return tuple(args)"}]}]}`)
		case 3:
			fmt.Fprint(response, `{"output":[{"type":"function_call","call_id":"call-2","name":"benchmark_candidate","arguments":"{\"source\":\"def call(args):\\n    return tuple(args)\"}"}]}`)
		case 4:
			fmt.Fprint(response, `{"output":[{"type":"message","content":[{"type":"output_text","text":"def call(args):\n    return tuple(args)"}]}]}`)
		default:
			t.Fatalf("unexpected request %d", count)
		}
	}))
	defer server.Close()

	benchmark := func(source []byte) (benchmarkResult, error) {
		score := 0.9
		if bytes.Contains(source, []byte("tuple")) {
			score = 0.8
		}
		return benchmarkResult{
			Evaluation: evaluation{Score: score},
			Output:     fmt.Sprintf(`{"correct":true,"score":%.1f}`, score),
		}, nil
	}
	candidate, searches, benchmarks, score, err := propose(
		context.Background(),
		server.Client(),
		server.URL,
		"test-key",
		"test-model",
		"trusted skill context",
		[]agentMessage{{Role: "user", Content: "optimize this"}},
		true,
		true,
		benchmark,
	)
	if err != nil {
		t.Fatal(err)
	}
	if searches != 1 {
		t.Fatalf("got %d searches, want 1", searches)
	}
	if benchmarks != 2 || score != 0.8 {
		t.Fatalf("got %d benchmarks at %.3g, want 2 at 0.8", benchmarks, score)
	}
	if requests.Load() != 4 {
		t.Fatalf("got %d model requests, want 4", requests.Load())
	}
	if !bytes.Equal(candidate, []byte("def call(args):\n    return tuple(args)\n")) {
		t.Fatalf("unexpected candidate %q", candidate)
	}
}

// TestWaitForModel verifies that readiness waits through an initially incomplete registry.
func TestWaitForModel(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/models" {
			t.Errorf("unexpected path %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("readiness request omitted proxy authentication")
		}

		count := requests.Add(1)
		response.Header().Set("Content-Type", "application/json")
		if count < 3 {
			fmt.Fprint(response, `{"data":[]}`)
			return
		}
		fmt.Fprint(response, `{"data":[{"id":"requested-model"}]}`)
	}))
	defer server.Close()

	err := waitForModel(context.Background(), server.Client(), server.URL, "test-key", "requested-model", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 3 {
		t.Fatalf("got %d model requests, want 3", requests.Load())
	}
}

// TestRememberAttempt verifies that rolling context keeps complete recent turns and releases the oldest.
func TestRememberAttempt(t *testing.T) {
	var recent []agentMessage
	for attempt := 1; attempt <= 7; attempt++ {
		recent = rememberAttempt(recent, []byte(fmt.Sprintf("candidate-%d", attempt)), fmt.Sprintf("result-%d", attempt))
	}
	if len(recent) != 10 {
		t.Fatalf("got %d recent messages, want 10", len(recent))
	}
	if recent[0].Content != "candidate-3" || recent[9].Content != "result-7" {
		t.Fatalf("unexpected rolling context: %#v", recent)
	}
}

// TestLoadSkillsIncludesChip verifies that hardware legality is part of the agent's trusted context.
func TestLoadSkillsIncludesChip(t *testing.T) {
	instructions, _, err := loadSkills(nil, "NVIDIA H100 (sm_90)")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(instructions, "Target GPU: NVIDIA H100 (sm_90)") {
		t.Fatalf("agent instructions omitted chip: %q", instructions)
	}
}
