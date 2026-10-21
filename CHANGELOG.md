# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0rc3] - 2026-10-21

Release candidate for 0.5.0.

### Upgrade from 0.5.0rc2

`client_now_date` must now have a UTC offset, such as
`2026-10-21T09:00:00-04:00`. A value without one raises `ValueError`. Pine
ignored such values before.

### New

- **Update a managed user.** `platform.managed_users.update()` changes the
  email, name, phone or assistant. `phone=""` removes the phone.
- **Choose the assistant.** The assistant places and answers phone calls for
  the user. Before, every managed user had Pine's built-in assistant. Now you
  can choose its identity (voice and name) from
  `settings.assistant_identities()` and rename it, with `assistant=` on
  `create()` or `update()`.
- **Choose the caller ID.** `sessions.update_config(caller_id="platform")`
  makes calls show a platform number instead of the user's phone.
- **Time zone on Socket.IO.** `chat()` and `send_message()` take
  `client_now_date`, as REST `send_message()` does.

### Fixed

- Socket.IO messages sent a time without a UTC offset, so the agent always
  used UTC.

### Smaller changes

- `ManagedUser.assistant` is `{identity_id, name}`, or `None` for Pine's
  default assistant.
- `assistant={"identity_id": ...}` alone also replaces a custom name.
- `sessions.get_config()` reads the caller ID.
- An invalid `assistant`, `caller_id` or `client_now_date` raises `ValueError`
  before any request.
- `PineAI` has the same new REST methods.

## [0.5.0rc2] - 2026-10-19

Internal release candidate for 0.5.0.

### Added

- Pine Platform API support for partners (businesses that build on Pine).
  `AsyncPineAI` and `PineAI` accept a partner's secret key as `api_key`.
  `client.platform.managed_users.create()` and `.get()` take your `external_id`,
  return typed `ManagedUser` values, and raise `PlatformError` without upstream
  bodies.
- Managed-user mode: `api_key` plus `user_id`, where `user_id` is the
  `ManagedUser.id` (the Pine user ID). REST requests send the key and a
  `Pine-Managed-User-Id` header. The Socket.IO handshake sends
  `{"token", "managed_user_id"}`, and envelopes carry the same ID as
  `user_id`. `connect()` sends no REST request first. Injected HTTP clients
  cannot add or override the header.
- In key mode, `user_id` must be a string in canonical decimal form: 1 to
  9223372036854775807, with no sign, spaces or leading zeros. An `int` is
  refused. `validate_user_id` exposes this check.
- A client with `api_key` and no `user_id` can call only the Platform API. A
  user-scoped call from it raises `ValueError` before any request is sent.
  These also raise `ValueError`: `api_key` with `access_token`, identity
  overrides passed to `connect()` in key mode, `platform.*` on a client without
  `api_key`, and `auth.verify_code()` or `set_token()` on a client with
  `api_key` (checked before the code is sent). A Platform key passed as
  `access_token` raises `ValueError` that says to pass it as `api_key`.
- `API_KEY_PREFIXES` (`pine_sk_live_`, `pine_sk_test_`), `validate_external_id`,
  `validate_user_id` and `MANAGED_USER_HEADER` are public. The
  managed-user header is sent only with the API key it was configured with, and
  identity headers are applied after caller headers, so neither can be
  overridden per request. Error messages never contain the key, the
  `external_id` or the user ID.
- `sessions.history()` on the async and sync clients reads one page of a
  session's stored messages over REST (`GET /v2/sessions/{id}/messages`), so
  a client without Socket.IO can read the agent's replies. It returns
  `SessionHistoryPage` and raises `SessionError`.
- Docs for managed users: `docs/sdk-api.md` (SDK methods), `docs/http-api.md`
  (raw HTTP API) and `docs/socketio.md` (Socket.IO, reference only).
- In key mode, a `connect()` that gets no `ready` event within `ready_timeout`
  raises `ConnectionError` with the hint to check `api_key` and `user_id`. The
  Pine backend refuses a bad handshake by closing the connection, with no
  `connect_error`.

## [0.5.0rc1] - 2026-09-18

Internal release candidate for 0.5.0. See [MIGRATION.md](MIGRATION.md) before
upgrading from any released 0.3.x version or current main's 0.4.0 surface.

### Added

- `AsyncPineAI` now supports `async with` and `aclose()`. `disconnect()` only
  closes Socket.IO; `aclose()` also closes SDK-owned HTTP resources.
- Typed REST resources: `auth.me()`, ticket mint/redeem, and validated session
  list/get responses.
- Explicit `api_base_path`, `http_client`, and `http_transport` options.
  Injected HTTP clients remain caller-owned.
- `PineAI` now has real synchronous REST `auth` and `sessions` resources built
  on `httpx.Client`.

### Changed

- HTTP errors expose a stable error code and `status_code` without reproducing
  upstream response bodies.
- `sessions.list()` sends `ensure_copilot=false` by default when the backend
  supports that read-only query option.

### Removed

- `PineAI` no longer offers synchronous Socket.IO methods. Use `AsyncPineAI`
  for realtime connections and streams; the former wrapper owned an event loop
  and exposed coroutine-backed resources from a synchronous client.

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
