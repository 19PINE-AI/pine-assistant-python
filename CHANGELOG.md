# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-08-08

Aligned to the supported protocol scope: the subset of the task-session
Socket.IO protocol whose names, payloads, and semantics carry a compatibility
guarantee. What the SDK models is now that subset and nothing else.

### Added

- `turn_timeout` on `chat()`, `listen()` and `chat_sync()` — a wall-clock bound
  on one turn, keeping whatever arrived before it elapsed. It defaults to none,
  which leaves a turn waiting indefinitely on a session that never closes it.
- `is_supported_event()` and `SUPPORTED_EVENTS` — whether an event carries the
  guarantee.
- `session:llm_thinking`, `session:tool_status` and `session:restriction`, with
  models. None were modelled before; `session:tool_status` is where an outbound
  call reports its number, duration, credits, and textual outcome.
- `AsyncPineAI.rebuild()` — pages through history until the cursor is
  exhausted. Recovery is an unconditional rebuild: joining never resumes from a
  cursor, and a short or empty page does not mean a range is done.
- `AsyncPineAI.on_reconnect()` — fires after a reconnect has re-joined, so
  callers can rebuild. A connection can stay open after delivery has stopped.
- `AsyncPineAI.emit_event()` — the escape hatch for sending anything outside the
  supported surface.
- Protocol fixtures and contract tests under `tests/protocol`, and
  `tests/integration/record_fixtures.py` to record them from a live session.
  Re-recording is the only way server drift gets noticed.

### Changed

- A turn now ends when the agent has spoken and then gone quiet, rather than
  when it has spoken at some point during the turn. The old rule set a flag on
  the first reply and never cleared it, so the rest of the turn ran on the
  two-second timeout — including a tool call, where silence means the work is
  taking a while. `session:tool_status` is no longer counted as speech for this
  purpose: it reports what the agent is doing, not what it says.
- `session:state` values `credits_exhausted` and `task_paused` now end a turn.
  Both stop on the account rather than on the agent, so nothing further arrives
  from the session. `task_stale` is dropped from that set: the server has no
  such state, and staleness is `is_stale` on the session object over REST.
- `session:join` now carries `since_revision` "0", on first join and on
  reconnect. The incremental-synchronization fields in the response are ignored.
- Events are deduplicated on the event identifier together with the message
  type. Keying on the identifier alone drops real events, since identifiers
  collide across types.
- A turn begins and ends on supported events only. It previously hinged on
  `session:ask_for_location`, `session:interactive_auth_confirmation`,
  `session:three_way_call` and `session:reward`, none of which are maintained.

### Fixed

- Sessions joined through `join_session()` were never re-joined after a
  reconnect. Membership was tracked on the fire-and-forget emit path only, while
  joining goes out through the request/response path.

### Removed

Everything below is still emitted by the server and still reaches callers
untouched — the SDK just no longer models it. Send with `emit_event()`.

- `send_auth_confirmation()`, `send_location_response()`,
  `send_location_selection()`.
- `NotificationEvent`, the `notification:*` constants, and the `session:reward`
  and `session:payment` models.
- The out-of-scope `S2CEvent` and `C2SEvent` members, including
  `session:work_log`, `session:work_log_part` and `session:thinking`. The
  reasoning stream in scope is `session:llm_thinking`, a different event that
  the SDK did not previously carry.
- The `action` argument on `chat()` and `send_message()`, and `request_work_log`
  on `get_history()`.
- Wall-clock filtering of events older than the moment a turn began. It
  contradicts rebuilding from history, and a clock offset made it drop real
  events.
- `session:input_state`, `session:required_action` and `session:task_ready`,
  with the `InputState`, `InputStateCode`, `RequiredActionData` and
  `TaskReadyData` models. The scope no longer covers them. A session stopped on
  its credit balance is reported by `session:state`, whose values include
  `credits_exhausted` and `task_paused`.
- The turn no longer ends when `session:input_state` reports that input is
  accepted. That event is observed to arrive before the agent has said
  anything, so ending on it truncates the reply.

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
