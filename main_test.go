package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// TestPropose verifies that the agent sends hosted search and preserves non-message output items.
func TestPropose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			t.Errorf("unexpected path %q", request.URL.Path)
		}
		var payload map[string]any
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		tools, ok := payload["tools"].([]any)
		if !ok || len(tools) != 1 {
			t.Fatalf("missing hosted search tool: %#v", payload["tools"])
		}
		choice, ok := payload["tool_choice"].(map[string]any)
		if !ok || choice["type"] != "web_search" {
			t.Fatalf("first turn did not require web search: %#v", payload["tool_choice"])
		}
		response.Header().Set("Content-Type", "application/json")
		fmt.Fprint(response, `{"output":[{"type":"web_search_call"},{"type":"message","content":[{"type":"output_text","text":"def call(args):\n    return args"}]}]}`)
	}))
	defer server.Close()

	candidate, searches, err := propose(
		context.Background(),
		server.Client(),
		server.URL,
		"test-key",
		"test-model",
		"trusted skill context",
		[]agentMessage{{Role: "user", Content: "optimize this"}},
		true,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if searches != 1 {
		t.Fatalf("got %d searches, want 1", searches)
	}
	if !bytes.Equal(candidate, []byte("def call(args):\n    return args\n")) {
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
