package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

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
