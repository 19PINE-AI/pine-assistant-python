# Migrating to 0.5.0

Version 0.5.0 keeps the supported 0.4.0 asynchronous realtime API on
`AsyncPineAI` and adds typed REST resources. This release candidate is for
validation; do not use `0.5.0rc1` as a default production dependency.

## 0.3.3 to 0.5.0

Current main contains unreleased 0.4.0 protocol-scope changes. Read the 0.4.0
entry in `CHANGELOG.md` first: it removed unsupported events and narrowed the
realtime contract. On 0.5.0, use `async with AsyncPineAI(...)` or call
`await client.aclose()` after realtime and REST work. `disconnect()` still only
ends Socket.IO.

Session reads now return typed models:

```python
# Before
listed = await client.sessions.list(limit=20)
session_id = listed["sessions"][0]["id"]

# 0.5.0
listed = await client.sessions.list(limit=20)
session_id = listed.sessions[0].id
```

`sessions.list()` now requests `ensure_copilot=false`. This is read-only only
against a backend that implements the option; older backends may ignore it.
Pass `ensure_copilot=True` only when the legacy side effect is wanted.

## Synchronous client

`PineAI.auth` and `PineAI.sessions` now return synchronous values rather than
coroutines. Synchronous Socket.IO methods (`connect`, `chat_sync`, and related
streaming helpers) are removed because a safe synchronous realtime client needs
an explicit event-loop ownership model. Move those calls to `AsyncPineAI`.

```python
# Before: this exposed an async resource from PineAI
result = await PineAI(access_token="...").sessions.list()

# 0.5.0
with PineAI(access_token="...") as client:
    result = client.sessions.list()
```

## Errors and HTTP configuration

Errors preserve `code` and `status_code`, but no longer include upstream
response text. Configure a nonstandard prefix with `api_base_path`; `/api`
remains the default. An injected `httpx.Client` or `httpx.AsyncClient` remains
open after the SDK client closes.

The SDK continues to support Python 3.10 and later.
