# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.3] - 2026-05-23

### Fixed

- Added `aiohttp>=3.9.0` as an explicit dependency. `python-socketio`'s async
  client requires `aiohttp` for the initial HTTP handshake regardless of which
  transport (polling or websocket) is selected, but it is only declared as an
  optional extra upstream. Without it, downstream consumers (e.g.
  `pine-mcp-server`, the CLI) failed at import/connect time with
  `ModuleNotFoundError: aiohttp` and had to work around it with
  `uvx --with aiohttp`.

## [0.3.2] - 2026-05-21

### Fixed

- `SocketIOManager.connect()` now captures the server's `connect_error` payload
  and re-raises it as `pine_assistant.errors.ConnectionError` with the actual
  reason (e.g. "invalid token", "device mismatch"). Previously the upstream
  python-socketio swallowed the payload and only the generic
  `"One or more namespaces failed to connect"` surfaced, making auth failures
  undiagnosable.
- When the server accepts the WebSocket but never emits `ready` (the Pine
  backend's behavior when auth is rejected), the SDK now raises
  `ConnectionError("Socket.IO connected but no 'ready' event after Xs. This
  usually means access_token or user_id is invalid/expired — re-run the auth
  flow.")` instead of a bare `asyncio.TimeoutError`.
- Eliminated a spurious `"Emit failed for session:leave: / is not a connected
  namespace"` warning when callers do `leave_session()` immediately followed
  by `disconnect()`. The scheduled emit task now no-ops if the socket has
  already been torn down.

### Added

- `PINE_DEVICE_ID` environment variable is now honored by
  `_get_or_create_device_id`. Use it to pin a stable device identity when the
  SDK runs in a sandboxed subprocess (e.g. an MCP server spawned by Claude
  Desktop or Cursor) where `~/.pine/device_id` may not be writable.
- When persistence of a freshly-generated device_id to `~/.pine/device_id`
  fails, a warning is logged pointing at `PINE_DEVICE_ID` as the fix.

## [0.1.0] - 2026-02-16

### Added

- Async SDK client (`AsyncPineAI`) with Socket.IO and HTTP transport layers
- Synchronous wrapper (`PineAI`) for non-async usage
- Authentication flow (email verification with code)
- Session management (create, list, join, leave)
- Real-time chat with async generator streaming
- Task lifecycle support (start, watch, cancel)
- Form handling and payment event support
- Pydantic models for all event and data types
- Stream buffering for text and work-log events
- CLI tool (`pine`) with auth, chat, sessions, and tasks commands
- Full type annotations with `py.typed` marker
- Integration test suite

[0.1.0]: https://github.com/19PINE-AI/pine-assistant-python/releases/tag/v0.1.0
