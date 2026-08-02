"""Use the ComputeSDK gateway protocol needed by Tilesmith's remote evaluator.

Derived from computesdk/computesdk 2.5.4 at commit
e3a9bf071a2c301f98aeab4b96923f6abb252387, the last implementation before
the TypeScript SDK removed hosted control-plane transport.

MIT License

Copyright (c) 2025 computesdk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast
from urllib.parse import quote

import httpx

COMPUTESDK_COMMIT = "e3a9bf071a2c301f98aeab4b96923f6abb252387"
COMPUTESDK_VERSION = "2.5.4"
DEFAULT_GATEWAY_URL = "https://tributary.edge.computesdk.com"
MAX_COMMAND_TIMEOUT_SECONDS = 300
_READY_ATTEMPTS = 30
_READY_INITIAL_DELAY_SECONDS = 0.5
_READY_MAX_DELAY_SECONDS = 5.0
_READY_BACKOFF = 1.5


class ComputeSDKError(RuntimeError):
    """Report a gateway, sandbox, transport, or response error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Retain the HTTP status when the remote service supplied one."""
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Describe one completed sandbox command."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class Sandbox:
    """Expose the gateway-backed sandbox operations needed by Tilesmith."""

    def __init__(
        self,
        *,
        sandbox_id: str,
        sandbox_url: str,
        token: str,
        provider: str,
        gateway: httpx.Client,
        request_timeout: float,
        redactions: Iterable[str],
    ) -> None:
        """Bind one allocated sandbox to its gateway cleanup route."""
        self.sandbox_id = sandbox_id
        self.provider = provider
        self._gateway = gateway
        self._redactions = tuple(redactions)
        self._client = httpx.Client(
            base_url=sandbox_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=request_timeout,
        )

    def health(self) -> bool:
        """Confirm that the sandbox daemon is responding."""
        _request(self._client, "GET", "/health", redactions=self._redactions)
        return True

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_ms: int | None = None,
    ) -> CommandResult:
        """Run one shell command and return its terminal result."""
        if not command:
            raise ValueError("command must not be empty")
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        body = _json_body(
            command=command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            background=True if timeout_ms is not None else None,
        )
        deadline = (
            time.monotonic() + timeout_ms / 1_000 if timeout_ms is not None else None
        )
        response = _request(
            self._client,
            "POST",
            "/run/command",
            json=body,
            redactions=self._redactions,
        )
        data = _response_object(response)

        if timeout_ms is not None:
            terminal_id = data.get("terminal_id")
            command_id = data.get("cmd_id")
            if not isinstance(terminal_id, str) or not isinstance(command_id, str):
                raise ComputeSDKError(
                    "ComputeSDK did not return command tracking identifiers"
                )
            while data.get("status") not in {"completed", "failed"}:
                assert deadline is not None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ComputeSDKError(f"Command timed out after {timeout_ms} ms")
                wait_seconds = min(remaining, MAX_COMMAND_TIMEOUT_SECONDS)
                timeout_text = f"{wait_seconds:g}"
                try:
                    response = _request(
                        self._client,
                        "GET",
                        f"/terminals/{terminal_id}/commands/{command_id}/wait",
                        params={"timeout": timeout_text},
                        headers={"X-Request-Timeout": timeout_text},
                        timeout=wait_seconds + 5,
                        redactions=self._redactions,
                    )
                except ComputeSDKError as error:
                    if error.status_code not in {408, 504}:
                        raise
                    continue
                data = _response_object(response)

        return _command_result(data)

    def write_text(self, path: str, content: str) -> None:
        """Write one UTF-8 text file into the sandbox."""
        _request(
            self._client,
            "POST",
            "/files",
            json={"path": path, "content": content},
            redactions=self._redactions,
        )

    def read_text(self, path: str) -> str:
        """Read one UTF-8 text file from the sandbox."""
        try:
            response = _request(
                self._client,
                "GET",
                f"/files/{quote(path, safe='/')}",
                params={"content": "true"},
                redactions=self._redactions,
            )
        except ComputeSDKError as error:
            if error.status_code == 404:
                raise FileNotFoundError(path) from error
            raise
        if isinstance(response, str):
            return response
        data = _response_object(response)
        content = data.get("content")
        if not isinstance(content, str):
            raise ComputeSDKError("ComputeSDK returned invalid file content")
        return content

    def destroy(self) -> None:
        """Destroy the remote sandbox once and close its HTTP client."""
        if self._client.is_closed:
            return
        try:
            _request(
                self._gateway,
                "DELETE",
                f"/v1/sandboxes/{self.sandbox_id}",
                redactions=self._redactions,
            )
        except ComputeSDKError as error:
            if error.status_code != 404:
                raise
        finally:
            self._client.close()


@contextmanager
def ephemeral_sandbox(
    *,
    api_key: str,
    token_id: str,
    token_secret: str,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    request_timeout: float = 30.0,
    timeout_ms: int | None = None,
    template_id: str | None = None,
    gpu: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Sandbox]:
    """Create one gateway-routed sandbox and always destroy it."""
    if not api_key:
        raise ValueError("api_key must not be empty")
    if not token_id or not token_secret:
        raise ValueError("Modal token_id and token_secret must not be empty")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError("timeout_ms must be greater than zero")

    headers = {
        "Content-Type": "application/json",
        "X-ComputeSDK-API-Key": api_key,
        "X-Provider": "modal",
        "X-Modal-Token-Id": token_id,
        "X-Modal-Token-Secret": token_secret,
    }
    body = _json_body(
        timeout=timeout_ms,
        templateId=template_id,
        gpu=gpu,
        envs=dict(env) if env is not None else None,
    )
    redactions = (api_key, token_id, token_secret, *(env or {}).values())

    with httpx.Client(
        base_url=gateway_url.rstrip("/"),
        headers=headers,
        timeout=request_timeout,
    ) as gateway:
        response = _request(
            gateway,
            "POST",
            "/v1/sandboxes",
            json=body,
            redactions=redactions,
        )
        data = _response_object(response)
        sandbox_token = cast(str, data.get("token") or api_key)
        sandbox = Sandbox(
            sandbox_id=cast(str, data["sandboxId"]),
            sandbox_url=cast(str, data["url"]),
            token=sandbox_token,
            provider=cast(str, data.get("provider", "modal")),
            gateway=gateway,
            request_timeout=request_timeout,
            redactions=(*redactions, sandbox_token),
        )

        try:
            _wait_until_ready(sandbox)
            yield sandbox
        except BaseException as operation_error:
            try:
                sandbox.destroy()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    f"sandbox operation failed ({operation_error!r}) and cleanup failed"
                ) from cleanup_error
            raise
        else:
            sandbox.destroy()


def _wait_until_ready(sandbox: Sandbox) -> None:
    """Poll a new sandbox until its daemon responds."""
    delay = _READY_INITIAL_DELAY_SECONDS
    for attempt in range(_READY_ATTEMPTS):
        try:
            sandbox.health()
            return
        except ComputeSDKError:
            if attempt == _READY_ATTEMPTS - 1:
                raise
        time.sleep(delay)
        delay = min(delay * _READY_BACKOFF, _READY_MAX_DELAY_SECONDS)


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json: object | None = None,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
    redactions: Iterable[str] = (),
) -> object:
    """Send one request and normalize transport and HTTP failures."""
    try:
        response = client.request(
            method,
            path,
            json=json,
            params=params,
            headers=headers,
            timeout=client.timeout if timeout is None else timeout,
        )
    except httpx.TimeoutException as error:
        raise ComputeSDKError(f"{method} {path} timed out") from error
    except httpx.RequestError as error:
        raise ComputeSDKError(f"{method} {path} could not reach ComputeSDK") from error

    if response.is_error:
        raise ComputeSDKError(
            _error_message(response, redactions), status_code=response.status_code
        )
    if response.status_code == 204 or not response.content:
        return None
    if "application/json" not in response.headers.get("content-type", ""):
        return response.text
    try:
        return response.json()
    except ValueError as error:
        raise ComputeSDKError(f"{method} {path} returned invalid JSON") from error


def _response_object(response: object) -> dict[str, object]:
    """Unwrap one ComputeSDK data envelope."""
    if not isinstance(response, dict):
        raise ComputeSDKError("ComputeSDK returned an unexpected response")
    response_object = cast(dict[str, object], response)
    value = response_object.get("data", response_object)
    if not isinstance(value, dict):
        raise ComputeSDKError("ComputeSDK returned an unexpected data payload")
    return cast(dict[str, object], value)


def _json_body(**values: object | None) -> dict[str, object]:
    """Omit unset JSON fields without dropping falsey values."""
    return {name: value for name, value in values.items() if value is not None}


def _command_result(data: Mapping[str, object]) -> CommandResult:
    """Validate that one command response is terminal and complete."""
    status = data.get("status")
    if status not in {None, "completed", "failed"}:
        raise ComputeSDKError(
            f"ComputeSDK returned nonterminal command status {status!r}"
        )
    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")
    exit_code = data.get("exit_code")
    duration_ms = data.get("duration_ms", 0)
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ComputeSDKError("ComputeSDK returned invalid command output")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ComputeSDKError(
            "ComputeSDK returned a terminal command without an exit code"
        )
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool):
        raise ComputeSDKError("ComputeSDK returned an invalid command duration")
    return CommandResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )


def _error_message(response: httpx.Response, redactions: Iterable[str]) -> str:
    """Extract one bounded, redacted error message from an HTTP response."""
    message = response.text.strip()
    try:
        payload: object = response.json()
        if isinstance(payload, dict):
            payload_object = cast(dict[str, object], payload)
            candidate = payload_object.get("error") or payload_object.get("message")
            if isinstance(candidate, str):
                message = candidate
    except ValueError:
        pass
    for secret in redactions:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return (message or f"ComputeSDK returned HTTP {response.status_code}")[:2_000]
