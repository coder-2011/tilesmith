"""Bind modern TypeScript ComputeSDK providers to Python.

The binding launches the pinned ``computesdk`` npm package in a small JSON-lines
bridge. Provider packages remain the implementation authority, so the Python API
supports any official or custom ComputeSDK provider without calling a provider
SDK directly.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, cast
from urllib.parse import quote, quote_plus

COMPUTESDK_VERSION = "4.1.4"
# These official provider releases all depend on the pinned ComputeSDK core above.
OFFICIAL_PROVIDER_VERSIONS = {
    "archil": "0.4.7",
    "blaxel": "1.6.18",
    "cloudflare": "1.6.14",
    "codesandbox": "1.5.51",
    "daytona": "1.7.32",
    "declaw": "0.2.8",
    "e2b": "1.7.52",
    "hopx": "0.2.27",
    "modal": "1.9.4",
    "namespace": "1.6.14",
    "runloop": "1.3.54",
    "tensorlake": "0.1.9",
    "upstash": "0.3.9",
    "vercel": "1.7.31",
}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
_BRIDGE_RESPONSE_LIMIT_BYTES = 128 * 1024 * 1024
_BRIDGE_PATH = Path(__file__).with_name("_computesdk_bridge.cjs")
_PACKAGE_PATTERN = re.compile(r"^(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*$")
_SECRET_NAME_PARTS = frozenset({"CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN"})


class ComputeSDKError(RuntimeError):
    """Report a TypeScript bridge, provider, transport, or response error."""


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Describe one explicitly versioned ComputeSDK provider factory."""

    package: str
    factory: str
    version: str
    config: Mapping[str, object] = field(default_factory=dict[str, object], repr=False)

    def __post_init__(self) -> None:
        """Reject ambiguous npm inputs before starting a subprocess."""
        if not _PACKAGE_PATTERN.fullmatch(self.package):
            raise ValueError(f"Invalid ComputeSDK provider package: {self.package!r}")
        if not self.factory or any(character.isspace() for character in self.factory):
            raise ValueError("ComputeSDK provider factory must be a non-empty export name")
        if not self.version or any(character.isspace() for character in self.version):
            raise ValueError("ComputeSDK provider version must be explicit and non-empty")
        object.__setattr__(self, "config", dict(self.config))
        _json_bytes(self.config)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Describe one completed ComputeSDK command."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


@dataclass(slots=True)
class _Pending:
    """Retain one in-flight bridge response and its redaction policy."""

    future: asyncio.Future[object]
    redactions: tuple[str, ...]
    redact_result: bool


class _Bridge:
    """Own one Node process and multiplex concurrent ComputeSDK requests."""

    def __init__(
        self,
        provider: ProviderSpec,
        request_timeout: float,
    ) -> None:
        """Capture immutable process configuration without starting Node."""
        self._provider = provider
        self._request_timeout = request_timeout
        config_strings = _config_strings(provider.config)
        self._redactions = _prepare_redactions((*config_strings, *_environment_secrets()))
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start npm's pinned package environment and configure the selected provider."""
        npm = shutil.which("npm")
        node = shutil.which("node")
        if npm is None or node is None:
            raise ComputeSDKError("Modern ComputeSDK execution requires both node and npm on PATH")
        if not _BRIDGE_PATH.is_file():
            raise ComputeSDKError(f"ComputeSDK bridge is missing from the installed package: {_BRIDGE_PATH}")

        packages = [f"--package=computesdk@{COMPUTESDK_VERSION}"]
        packages.append(f"--package={self._provider.package}@{self._provider.version}")
        command = [npm, "exec", "--yes", "--silent", *packages, "--", node, str(_BRIDGE_PATH)]
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # readFile responses contain source or traces in one JSON line.
            limit=_BRIDGE_RESPONSE_LIMIT_BYTES,
        )
        self._reader_task = asyncio.create_task(self._read_responses(), name="tilesmith-computesdk-bridge")
        self._stderr_task = asyncio.create_task(
            self._read_stderr(),
            name="tilesmith-computesdk-stderr",
        )
        specification = {
            "package": self._provider.package,
            "factory": self._provider.factory,
            "config": self._provider.config,
        }
        try:
            await self.call("configure", {"provider": specification})
        except BaseException:
            # A failed package load must not leave npm or Node running in the background.
            with suppress(BaseException):
                await self.close()
            raise

    async def call(
        self,
        action: str,
        payload: Mapping[str, object] | None = None,
        *,
        timeout: float | None = None,
        cancellable: bool = False,
        redactions: Sequence[str] = (),
        redact_result: bool = True,
    ) -> object:
        """Send one request and await its matching response."""
        request_timeout = self._request_timeout if timeout is None else timeout
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("ComputeSDK request timeout must be finite and positive")
        if redactions:
            # Provider diagnostics may mention request secrets after their response has completed.
            request_redactions = _prepare_redactions(redactions)
            self._redactions = tuple(sorted({*self._redactions, *request_redactions}, key=len, reverse=True))
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            message = self._process_failure("ComputeSDK bridge is not running")
            raise ComputeSDKError(message)

        loop = asyncio.get_running_loop()
        request_id = self._next_id
        request = {"id": request_id, "action": action, **(payload or {})}
        encoded = _json_bytes(request) + b"\n"
        self._next_id += 1
        pending = _Pending(
            future=loop.create_future(),
            redactions=self._redactions,
            redact_result=redact_result,
        )
        self._pending[request_id] = pending

        try:
            deadline = loop.time() + request_timeout
            async with asyncio.timeout_at(deadline):
                async with self._write_lock:
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                return await pending.future
        except TimeoutError as error:
            if cancellable:
                await self._notify_cancel(request_id)
            message = _redact(f"ComputeSDK {action} timed out", pending.redactions)
            raise ComputeSDKError(message) from error
        except asyncio.CancelledError:
            if cancellable:
                await self._notify_cancel(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        """Ask Node to destroy owned sandboxes, then stop the bridge process."""
        process = self._process
        if process is None:
            return
        close_error: BaseException | None = None
        if process.returncode is None:
            try:
                await self.call("close", timeout=30.0)
            except BaseException as error:
                close_error = error
                process.terminate()
        try:
            async with asyncio.timeout(5.0):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except BaseException as reader_error:
                if close_error is None:
                    close_error = reader_error
        if self._stderr_task is not None:
            try:
                await self._stderr_task
            except BaseException as stderr_error:
                if close_error is None:
                    close_error = stderr_error
        self._process = None
        if close_error is not None:
            raise close_error

    async def _notify_cancel(self, request_id: int) -> None:
        """Abort creation and wait briefly for the provider's orphan cleanup."""
        with suppress(BaseException):
            await self.call("cancel", {"targetId": request_id}, timeout=30.0)

    async def _read_responses(self) -> None:
        """Demultiplex protocol responses and streamed command output."""
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while line := await process.stdout.readline():
                try:
                    message = cast(object, json.loads(line))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ComputeSDKError("ComputeSDK bridge returned invalid JSON") from error
                data = _object(message, "ComputeSDK bridge message")
                request_id = data.get("id")
                if not isinstance(request_id, int) or isinstance(request_id, bool):
                    continue
                pending = self._pending.get(request_id)
                if pending is None:
                    continue
                if pending.future.done():
                    continue
                error_value = data.get("error")
                if error_value is not None:
                    pending.future.set_exception(self._remote_error(error_value, pending.redactions))
                else:
                    result = data.get("result")
                    if pending.redact_result:
                        result = _redact_value(result, pending.redactions)
                    pending.future.set_result(result)
        except BaseException as error:
            self._fail_pending(error)
            if process.returncode is None:
                process.terminate()
            raise
        finally:
            if process.returncode is None:
                await process.wait()
            self._fail_pending(ComputeSDKError(self._process_failure("ComputeSDK bridge exited")))

    async def _read_stderr(self) -> None:
        """Drain provider diagnostics into one fixed-size in-memory tail."""
        process = self._process
        assert process is not None and process.stderr is not None
        while chunk := await process.stderr.read(8_192):
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-8_000]

    def _remote_error(self, value: object, redactions: Sequence[str]) -> ComputeSDKError:
        """Convert a structured TypeScript error without leaking credentials."""
        data = _object(value, "ComputeSDK error")
        name = data.get("name", "Error")
        message = data.get("message", "ComputeSDK operation failed")
        rendered = f"{name}: {message}"
        return ComputeSDKError(_redact(rendered, redactions)[:4_000])

    def _process_failure(self, prefix: str) -> str:
        """Append bounded, secret-redacted Node diagnostics to a bridge failure."""
        details = bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
        message = f"{prefix}: {details}" if details else prefix
        return _redact(message, self._redactions)[:8_000]

    def _fail_pending(self, error: BaseException) -> None:
        """Wake every caller when the shared bridge can no longer answer."""
        for pending in self._pending.values():
            if not pending.future.done():
                if isinstance(error, ComputeSDKError):
                    pending.future.set_exception(ComputeSDKError(_redact(str(error), pending.redactions)))
                else:
                    pending.future.set_exception(error)


class ComputeSDKClient:
    """Expose Tilesmith's selected ComputeSDK provider to Python."""

    def __init__(
        self,
        provider: ProviderSpec,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Retain one validated provider without provisioning resources."""
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("ComputeSDK request_timeout must be finite and positive")
        self._bridge = _Bridge(provider, request_timeout)
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        """Start the TypeScript bridge for an async context."""
        await self.start()
        return self

    async def __aexit__(self, _error_type: object, error: BaseException | None, _traceback: object) -> None:
        """Close the bridge while preserving an operation and cleanup failure together."""
        try:
            await self.close()
        except BaseException as cleanup_error:
            if error is not None:
                raise BaseExceptionGroup(
                    "ComputeSDK operation and client cleanup both failed",
                    [error, cleanup_error],
                ) from None
            raise

    async def start(self) -> None:
        """Install the pinned npm packages exactly once."""
        if self._closed or self._close_task is not None:
            raise ComputeSDKError("ComputeSDK client is closed")
        if self._started:
            return
        async with self._start_lock:
            if not self._started:
                try:
                    await self._bridge.start()
                    self._started = True
                except BaseException:
                    self._closed = True
                    with suppress(BaseException):
                        await self._bridge.close()
                    raise

    async def close(self) -> None:
        """Destroy bridge-owned sandboxes and stop Node."""
        if self._close_task is not None:
            await asyncio.shield(self._close_task)
            return
        if self._closed:
            return
        self._close_task = asyncio.create_task(
            self._close_bridge(),
            name="tilesmith-computesdk-close",
        )
        await asyncio.shield(self._close_task)

    # _close_bridge performs the one shared client shutdown operation.
    async def _close_bridge(self) -> None:
        """Close Node once while every concurrent caller awaits the same task."""
        try:
            await self._bridge.close()
        finally:
            self._closed = True

    async def create(
        self,
        options: Mapping[str, object] | None = None,
        *,
        request_timeout: float | None = None,
    ) -> Sandbox:
        """Create a sandbox through the selected ComputeSDK provider."""
        await self.start()
        create_options = dict(options or {})
        redactions = _mapping_values(create_options.get("envs"))
        result = await self._bridge.call(
            "create",
            {"options": create_options},
            timeout=request_timeout,
            cancellable=True,
            redactions=redactions,
        )
        data = _object(result, "ComputeSDK sandbox")
        return Sandbox(
            self._bridge,
            _string(data.get("sandboxId"), "sandboxId"),
            _string(data.get("provider"), "provider"),
        )

    @asynccontextmanager
    async def ephemeral_sandbox(
        self,
        options: Mapping[str, object] | None = None,
        *,
        request_timeout: float | None = None,
    ) -> AsyncGenerator[Sandbox]:
        """Create one sandbox and always attempt to destroy it."""
        sandbox = await self.create(options, request_timeout=request_timeout)
        try:
            yield sandbox
        except BaseException as operation_error:
            try:
                await sandbox.destroy()
            except BaseException as cleanup_error:
                grouped = BaseExceptionGroup(
                    "Sandbox operation and cleanup both failed",
                    [operation_error, cleanup_error],
                )
                grouped.__dict__["computesdk_sandbox_id"] = sandbox.sandbox_id
                grouped.__dict__["computesdk_teardown"] = "failed"
                raise grouped from None
            operation_error.__dict__["computesdk_sandbox_id"] = sandbox.sandbox_id
            operation_error.__dict__["computesdk_teardown"] = "destroyed"
            raise
        else:
            try:
                await sandbox.destroy()
            except BaseException as cleanup_error:
                cleanup_error.__dict__["computesdk_sandbox_id"] = sandbox.sandbox_id
                cleanup_error.__dict__["computesdk_teardown"] = "failed"
                raise


class Sandbox:
    """Expose the complete provider-neutral ComputeSDK sandbox interface."""

    def __init__(self, bridge: _Bridge, sandbox_id: str, provider: str) -> None:
        """Bind a stable Python handle to its live TypeScript sandbox object."""
        self.sandbox_id = sandbox_id
        self.provider = provider
        self._bridge = bridge
        self._destroyed = False
        self._destroy_task: asyncio.Task[None] | None = None

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_ms: int | None = None,
        redact_output: bool = True,
    ) -> CommandResult:
        """Run one shell command, redacting its output unless it is an opaque transfer."""
        _nonempty(command, "command")
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")
        options: dict[str, object] = {}
        if cwd is not None:
            options["cwd"] = cwd
        if env is not None:
            options["env"] = dict(env)
        if timeout_ms is not None:
            options["timeout"] = timeout_ms
        host_timeout = None if timeout_ms is None else timeout_ms / 1_000 + 5.0
        result = await self._bridge.call(
            "runCommand",
            {
                "sandboxId": self.sandbox_id,
                "command": command,
                "options": options,
            },
            timeout=host_timeout,
            redactions=_mapping_values(env),
            redact_result=redact_output,
        )
        return _command_result(result)

    async def wait_until_ready(self, timeout_ms: int) -> None:
        """Wait until the sandbox can execute a real command, not merely report running."""
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")
        try:
            await self._bridge.call(
                "sandboxWaitUntilReady",
                {"sandboxId": self.sandbox_id, "timeoutMs": timeout_ms},
                timeout=timeout_ms / 1_000 + 5.0,
            )
        except ComputeSDKError as error:
            seconds = timeout_ms / 1_000
            raise ComputeSDKError(f"Sandbox did not become command-ready within {seconds:g} seconds: {error}") from error

    async def read_file(self, path: str) -> str:
        """Read one UTF-8 file through the provider filesystem."""
        _nonempty(path, "path")
        try:
            result = await self._bridge.call(
                "readFile",
                {"sandboxId": self.sandbox_id, "path": path},
                redact_result=False,
            )
        except ComputeSDKError as error:
            # Providers do not share a typed not-found error, so confirm absence explicitly.
            try:
                exists = await self.exists(path)
            except ComputeSDKError:
                raise error from None
            if not exists:
                raise FileNotFoundError(path) from error
            raise
        return _string(result, "file contents")

    async def write_file(self, path: str, content: str) -> None:
        """Write one UTF-8 file through the provider filesystem."""
        await self._bridge.call(
            "writeFile",
            {"sandboxId": self.sandbox_id, "path": _nonempty(path, "path"), "content": content},
        )

    async def exists(self, path: str) -> bool:
        """Return whether a provider filesystem path exists."""
        result = await self._bridge.call("exists", {"sandboxId": self.sandbox_id, "path": _nonempty(path, "path")})
        if not isinstance(result, bool):
            raise ComputeSDKError("ComputeSDK returned an invalid filesystem existence result")
        return result

    async def destroy(self) -> None:
        """Destroy this sandbox exactly once through its provider object."""
        if self._destroyed:
            return
        if self._destroy_task is None:
            self._destroy_task = asyncio.create_task(
                self._destroy_once(),
                name=f"tilesmith-destroy-{self.sandbox_id}",
            )
        task = self._destroy_task
        try:
            await asyncio.shield(task)
        finally:
            if task.done() and not task.cancelled() and task.exception() is not None:
                self._destroy_task = None

    # _destroy_once keeps teardown running after a caller cancellation.
    async def _destroy_once(self) -> None:
        """Dispatch one provider teardown and publish completion afterward."""
        await self._bridge.call("sandboxDestroy", {"sandboxId": self.sandbox_id})
        self._destroyed = True


def _json_bytes(value: object) -> bytes:
    """Serialize a protocol value and reject unsupported Python objects early."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError(f"ComputeSDK values must be JSON serializable: {error}") from error


def _config_strings(value: object) -> tuple[str, ...]:
    """Collect provider configuration strings so errors cannot expose credentials."""
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Mapping):
        values = cast(Mapping[object, object], value).values()
        return tuple(item for child in values for item in _config_strings(child))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = cast(Sequence[object], value)
        return tuple(item for child in values for item in _config_strings(child))
    return ()


def _mapping_values(value: object) -> tuple[str, ...]:
    """Return secret-like string values from an optional environment mapping."""
    if not isinstance(value, Mapping):
        return ()
    items = cast(Mapping[object, object], value).items()
    secrets = []
    for name, item in items:
        if not isinstance(name, str) or not isinstance(item, str) or not item:
            continue
        if _SECRET_NAME_PARTS.intersection(name.upper().split("_")):
            secrets.append(item)
    return tuple(secrets)


# _environment_secrets protects credentials read directly by provider packages.
def _environment_secrets() -> tuple[str, ...]:
    """Collect values from environment variables conventionally named as secrets."""
    return tuple(value for name, value in os.environ.items() if value and _SECRET_NAME_PARTS.intersection(name.upper().split("_")))


def _prepare_redactions(secrets: Sequence[str]) -> tuple[str, ...]:
    """Expand raw secrets into longest-first plain and URL-encoded variants."""
    variants = {variant for secret in secrets if secret for variant in (secret, quote(secret, safe=""), quote_plus(secret))}
    return tuple(sorted(variants, key=len, reverse=True))


def _redact(message: str, redactions: Sequence[str]) -> str:
    """Replace prepared variants longest-first so overlaps cannot expose suffixes."""
    for secret in redactions:
        message = message.replace(secret, "[REDACTED]")
    return message


# _redact_value removes prepared variants from every string in a diagnostic JSON value.
def _redact_value(value: object, redactions: Sequence[str]) -> object:
    """Recursively redact successful values that are not opaque file transports."""
    if isinstance(value, str):
        return _redact(value, redactions)
    if isinstance(value, list):
        return [_redact_value(item, redactions) for item in value]
    if isinstance(value, dict):
        return {_redact(key, redactions): _redact_value(item, redactions) for key, item in value.items()}
    return value


def _nonempty(value: str, name: str) -> str:
    """Validate one required non-empty string."""
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _object(value: object, description: str) -> dict[str, object]:
    """Validate one JSON object returned by the bridge."""
    if not isinstance(value, dict):
        raise ComputeSDKError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _string(value: object, description: str) -> str:
    """Validate one string returned by the bridge."""
    if not isinstance(value, str):
        raise ComputeSDKError(f"ComputeSDK returned invalid {description}")
    return value


def _integer(value: object, description: str) -> int:
    """Validate one integer returned by the bridge."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComputeSDKError(f"ComputeSDK returned invalid {description}")
    return value


def _number(value: object, description: str) -> float:
    """Validate one finite TypeScript number returned by the bridge."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ComputeSDKError(f"ComputeSDK returned invalid {description}")
    result = float(value)
    if not math.isfinite(result):
        raise ComputeSDKError(f"ComputeSDK returned invalid {description}")
    return result


def _command_result(value: object) -> CommandResult:
    """Validate and normalize a TypeScript command result."""
    data = _object(value, "command result")
    return CommandResult(
        stdout=_string(data.get("stdout"), "command stdout"),
        stderr=_string(data.get("stderr"), "command stderr"),
        exit_code=_integer(data.get("exitCode"), "command exitCode"),
        duration_ms=_number(data.get("durationMs"), "command durationMs"),
    )
