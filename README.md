# pine-assistant

[![PyPI version](https://img.shields.io/pypi/v/pine-assistant)](https://pypi.org/project/pine-assistant/)
[![Python versions](https://img.shields.io/pypi/pyversions/pine-assistant)](https://pypi.org/project/pine-assistant/)
[![license](https://img.shields.io/pypi/l/pine-assistant)](./LICENSE)

Pine AI SDK for Python. Let Pine AI handle your digital chores.

## Install

```bash
pip install pine-assistant          # SDK only
pip install pine-assistant[cli]     # SDK + CLI
```

## Quick Start (Async)

```python
from pine_assistant import AsyncPineAI

client = AsyncPineAI(access_token="...", user_id="...")
async with client:
    await client.connect()

    session = await client.sessions.create()
    await client.join_session(session["id"])
    await client.rebuild(session["id"])          # load the session's messages

    async for event in client.chat(session["id"], "Negotiate my Comcast bill",
                                   turn_timeout=120):
        print(event.type, event.data)
```

A client tracks one session. Concurrent sessions need one client each.

`disconnect()` only ends the real-time connection. `aclose()` (including the
end of `async with`) also releases the SDK-owned HTTP client. If you pass an
`httpx.AsyncClient`, it remains your responsibility to close it. Use
`api_base_path` to select an API prefix; it defaults to `/api` for compatibility.

## REST identity and sessions

```python
async with AsyncPineAI(access_token="...") as client:
    identity = await client.auth.me()               # AuthIdentity(user_id="...")
    sessions = await client.sessions.list(limit=20) # SessionListResponse
    session = await client.sessions.get("123")     # SessionInfo
```

The list call sends `ensure_copilot=false` by default. It is read-only when
used with a backend that supports this query option; older backends may ignore
it and retain their legacy Copilot behavior.

Use `await client.sessions.end_task(session_id)` to request a user-ended task;
the synchronous client provides the same method without `await`. The backend
checks ownership and eligibility, and the method returns the updated
`SessionInfo`. This is not proof that the task objective succeeded or that an
external action has already stopped. The SDK never retries this write: after a
timeout or server error, query the session and its history before deciding
whether to retry, because the state may already have changed.

Use `sessions.send_message()` when an application needs the REST write
acknowledgement without joining Socket.IO. It returns a typed status: `received`
means the message was persisted, `delivered` means it was handed to the Agent,
and `delivery_failed` means that handoff failed. None means that a task has
finished. A timeout or connection error leaves persistence unknown, so the SDK
never retries a send; recover Socket history before deciding what to do next.

```python
status = await client.sessions.send_message(
    "123", "Continue the task", request_id="ui-click-42",
)
if status.status == "delivered":
    print(status.message_id, status.revision)

outcomes = await client.sessions.outcomes("123")
for outcome in outcomes.items:
    print(outcome.outcome_id, outcome.outcome_narrative)
```

`outcomes()` returns newest-first persisted Outcomes. Follow `next_cursor` to
request older pages with `before=...`; `total` is a first-page snapshot and is
not a pagination signal.

For structured form answers, use the async method on a connected client:

```python
await client.connect()
try:
    receipt = await client.submit_form_response(
        "123", "456", {"contact_name": "Example User"},
    )
finally:
    await client.disconnect()
```

The method re-reads the original agent form from authenticated history, preserves
its message and request IDs, validates visible required fields, and JSON-encodes
array answers like the web app. Callers cannot override field privacy levels.
The backend's `session:message_status` receipt supplies the persisted reply ID:
`delivered` means it reached the agent and `received` means it was persisted but
delivery was not observed before the deadline. `unknown` requires checking
history before deciding whether to submit again. Each Socket.IO connection sends
a form only once because a late receipt cannot identify an attempt; reconnects
never retry or replay form submissions. A transport ACK is not a delivery
receipt. The legacy synchronous `send_form_response()` is deprecated because it
cannot verify the original form request.

## Platform API for partners

A *partner* is a business that builds on Pine through the Platform API. It
holds a secret key, `pine_sk_live_...`. Keep it on your servers: read it
from an environment variable or secret store, and never ship it in browser,
mobile or other client-side code, prompts, or logs. To get a key, contact
support@19pine.ai. Each end user of your product becomes a *managed user*,
identified by your own `external_id` (1-128 of letters, digits, `.`, `_`, `:`,
`|`, `-`, not starting with `.`; no personal data).

All managed users of one partner use that partner's credits. A partner can
limit the total credits that its managed users use.

The SDK methods for managed users are listed in
[docs/sdk-api.md](https://github.com/19PINE-AI/pine-assistant-python/blob/main/docs/sdk-api.md).
If you do not use Python, see the HTTP API in
[docs/http-api.md](https://github.com/19PINE-AI/pine-assistant-python/blob/main/docs/http-api.md)
and the Socket.IO reference in
[docs/socketio.md](https://github.com/19PINE-AI/pine-assistant-python/blob/main/docs/socketio.md).

Create the managed user once, and store its `id` in your own user table. The
call is idempotent per `external_id`: a repeat returns the stored user
unchanged, even if the profile you send differs. `phone` is optional, in E.164
form, and must be a number you have verified. `get()` finds a user by
`external_id` if you did not store the `id`. `update()` changes only the
fields you give; `phone=""` removes the phone.

```python
import os

from pine_assistant import AsyncPineAI

api_key = os.environ["PINE_API_KEY"]

async with AsyncPineAI(api_key=api_key) as partner_client:
    user = await partner_client.platform.managed_users.create(
        "usr_01J9Z3K8QF",
        email="ada@example.com",
        name="Ada Lovelace",
        phone="+14155550100",
    )
    same_user = await partner_client.platform.managed_users.get("usr_01J9Z3K8QF")
    moved_user = await partner_client.platform.managed_users.update("usr_01J9Z3K8QF", phone="+14155550123")
# Save user.id with your own user record.
```

`email` is optional. A managed user without an email cannot do email tasks,
only phone tasks.

`user.id` is the Pine user ID, a decimal string such as `"123456789"`. To act
as that user, pass it as `user_id` together with the key. Every user-scoped
REST request then sends the key and a `Pine-Managed-User-Id` header. The
Socket.IO handshake sends the key and `managed_user_id`, and every Socket.IO
message carries the same ID. The SDK checks the form of `user_id` when you
create the client. It accepts only a string, never an `int`.

```python
async with AsyncPineAI(api_key=api_key, user_id=user.id) as client:
    session = await client.sessions.create()
    await client.connect()
    await client.join_session(session["id"])
    message = "Call my dentist to move Tuesday's appointment"
    async for event in client.chat(session["id"], message):
        print(event.type, event.data)
```

A client with `api_key` and no `user_id` can call only the Platform API. A
user-scoped call from it, such as `sessions.list()` or `connect()`, raises
`ValueError` before any request is sent. These also raise `ValueError`:
`api_key` together with `access_token`, `platform.*` on a client without
`api_key`, and `auth.verify_code()` or `set_token()` on a client with `api_key`.
A key client keeps its key; use a separate client for a user token.

Every error below is a `PineAIError` with `code` and `status_code`, so catch
`PineAIError`. Some methods raise a subclass: `SessionError` from
`sessions.list`, `get`, `get_config`, `update_config`, `send_message`,
`history` and `outcomes`, and
`AuthError` from `auth.*`. Other `sessions` methods, such as `create` and
`start_task`, raise `PineAIError` itself.

- 400 `invalid_request`: the backend did not accept the managed-user ID.
- 404 `platform_managed_user_not_found`: the ID is not one of your managed users.
- 403 `platform_managed_user_unavailable`: the managed user is banned or
  archived.
- `client.auth.mint_ticket()` always fails with 403
  `platform_route_not_allowed`, raised as `AuthError`. Managed users cannot mint
  login tickets. Other routes closed to a partner key fail in the same way.

Billing and admission can also reject sessions and chat for a managed user:
402 `platform_spend_limit_reached`, 403 `platform_access_suspended`, 409
`platform_session_state_conflict`, 409 `platform_session_limit_reached` from
`sessions.start_task` when the running-task limit is reached.

If the server refuses the Socket.IO handshake, it closes the connection and
sends no error code. `connect()` then raises `ConnectionError` when
`ready_timeout` (15 seconds by default) ends. Its `code` is always
`connection_error`, so it does not tell you the cause. The message tells you to
check `api_key` and `user_id`.

`PineAI` accepts the same `api_key` and `user_id` for synchronous REST. Platform
API failures raise `PlatformError` with a stable `code` and `status_code` (for
example 401 for a bad key, 403 for a suspended partner, 404
`platform_managed_user_not_found`, 409 `platform_managed_user_limit_reached`).
Like every SDK error, it never reproduces the upstream body, the key, the
`external_id` or the user ID.

## Quick Start (Sync REST)

`PineAI` is a synchronous REST client. It returns values directly for auth and
session resources; use `AsyncPineAI` for Socket.IO and streaming.

```python
from pine_assistant import PineAI

with PineAI(access_token="...") as client:
    print(client.auth.me().user_id)
    for session in client.sessions.list(limit=20).sessions:
        print(session.id, session.title)
```

## Quick Start (CLI)

```bash
pine auth login                          # Email verification
pine chat                                # Interactive REPL
pine send "Negotiate my Comcast bill"    # One-shot message
pine sessions list                       # List sessions
pine task start <session-id>             # Start task
```

## The supported surface

The SDK models the supported protocol scope: the events whose names,
payloads, and semantics change compatibly or with notice.

**Connection and session**

| Event | What it is for |
|---|---|
| `ready` | Authentication succeeded and the connection is usable. Nothing is sent before it |
| `session:join` | Enter a session and read its current state. Sent both ways under this name |
| `session:history` | Read persisted messages. Also the only recovery mechanism in this scope |
| `session:error` | The only channel for server-reported failures |

**Conversation**

| Event | What it is for |
|---|---|
| `session:message` | Your input. Sent to the server, and returned under the same name in history |
| `session:text` | A complete agent message — the durable record |
| `session:text_part` | Streaming increments of one message, assembled by `message_id` |
| `session:rich_content` | A structured document, such as a search report. Its body is **not** repeated in `session:text`; ignore this event and the content is lost |
| `session:llm_thinking` | Reasoning and tool-call trace. Search has no event of its own — it appears here as a `tool_call` step |

**Session state**

| Event | What it is for |
|---|---|
| `session:state` | Where the task stands in its lifecycle |
| `session:message_status` | What became of a message you sent — the only way to tell a rejected or rate-limited one from one still being worked on |
| `session:update_title` | The session title, as the agent revises it |
| `session:restriction` | An account restriction. The only statement that a task will not complete |

**Interaction**

| Event | What it is for |
|---|---|
| `session:form_to_user` | Structured data collection — how a task asks for the account details it needs to act. Sent both ways under this name, and the most frequent interaction here |

**Task and result**

| Event | What it is for |
|---|---|
| `session:task_finished` | The result. `completion.result_title`, `result_description` and `outcome_narrative` carry the text; `completion.summary` is quantified, and `brief` is its only prose |
| `session:tool_status` | The record of one asynchronous operation. An outbound call reports here: the number, the duration, the credits, and `summary.text`. It updates in place, reusing its `message_id`, so expect several with the same one |

Payloads may gain fields at any time — tolerate fields you do not recognise.

A `tool_call` step in `session:llm_thinking` describes the same operation as the
matching `session:tool_status`. Do not show both.

A turn commonly delivers `session:text_part` alone: the composer reopens once
the agent has finished speaking, and the complete `session:text` is the durable
record, read back from history. Assemble the parts by `message_id` rather than
waiting for the complete message to arrive live.

## When a turn ends

`chat()` yields until the agent has spoken and then gone quiet — two seconds of
silence following text, a form, or a document. Silence following anything else
is read as work still running: an agent that says "placing the call now" and
starts a call is not finished, and the wait stays long.

A turn also ends when `session:state` settles — `task_finished`,
`task_cancelled`, `credits_exhausted` or `task_paused`. The last two stop on the
account rather than on the agent.

Nothing else ends a turn, so a turn whose last event is neither content nor a
settled state waits. Pass `turn_timeout` to bound it in wall-clock seconds;
whatever arrived before the deadline is still yielded.

```python
async for event in client.chat(sid, "...", turn_timeout=120):
    ...
```

Without `turn_timeout` a turn is waited on indefinitely. Note also that a task
outlives the turn that started it: an outbound call reports through
`session:tool_status` minutes after `chat()` has returned, which `subscribe()`
is for.

## Everything else passes through

The server emits many more events. The SDK delivers every one of them unchanged
rather than dropping them, but it models none of them:

```python
from pine_assistant import is_supported_event

async for event in client.chat(session_id, "..."):
    if not is_supported_event(event.type):
        continue          # or handle it yourself, at your own risk
```

An unsupported event may be renamed, have its payload changed, or stop being
emitted, without notice and without a version change. Tolerating one is
required; depending on one is not. To send one, use `client.emit_event(...)`.

Some of them are questions to the user that the SDK has no interface for.
Ignoring one leaves the conversation suspended, and the composer stays open —
show the message text and let the user answer in ordinary conversation. Never
fabricate an answer: the formats have no representation for refusal, and an
empty submission is indistinguishable from empty answers, so the agent may act
on it. Sending nothing is safe.

## What to respond to

Pine works the way a person would: a message is acknowledged, then reasoned
about, and only then answered. Acknowledgements and `session:llm_thinking`
arrive before the real response — a form, a text answer, or a task ready to run.

Respond only to what asks you something: `session:form_to_user`, a direct
question, and the task lifecycle. Replying to an acknowledgement starts a loop
in which each side answers the other's filler.

## Continuing an existing session

```python
result = await client.sessions.list(limit=20)

await client.join_session(existing_session_id)
messages = await client.rebuild(existing_session_id)
async for event in client.chat(existing_session_id, "What is the status?"):
    ...
```

To hand a session back to the user in the web app:

```python
print(AsyncPineAI.session_url(session_id))
```

## Recovery

State is rebuilt, never resumed. `join_session()` always joins from scratch,
and `rebuild()` pages through history until the cursor is exhausted — a short
or empty page does not mean the range is done.

```python
remove = client.on_reconnect(lambda: asyncio.create_task(reload(session_id)))
```

Rebuild on every join, on every reconnect, and whenever a session you are
tracking has been silent for a while: a connection can stay open after delivery
has stopped.

`rebuild()` returns messages of every type, including unsupported ones.
Filtering them is yours to do.

## When a session cannot proceed

`session:state` reports where the task stands, and several of its values say
that nothing further will arrive until something changes outside the session:

```python
from pine_assistant import S2CEvent

if event.type == S2CEvent.SESSION_STATE:
    state = (event.data or {}).get("content")
    if state in ("credits_exhausted", "task_paused"):
        ...   # waiting on the account, not on the agent
    if state in ("task_finished", "task_cancelled"):
        ...   # the task is over
```

Two more events state a stop outright:

```python
if event.type == S2CEvent.SESSION_RESTRICTION:
    ...   # an account restriction — the task will not complete
if event.type == S2CEvent.SESSION_ERROR:
    ...   # the only channel for server-reported failures
```

An expired session is read over REST, from the `is_stale` field on the session
object — expiry is a property of the session, not one of its states. On finding
one expired, create a new session and reference the old one in your first
message:

```python
new = await client.sessions.create()
client.send_message(new["id"], "...", referenced_sessions=[{"session_id": old_id}])
```

## Before an account is used

Two conditions have no remedy once a session is running:

- **Metered billing.** The account must be billed against a credit balance. On
  the alternative path a session halts at a payment step the SDK cannot answer.
- **Phone verification.** A Pine user account must verify its phone at
  provisioning time. It has no in-session remedy and no in-session signal. A
  managed user needs no verification: its phone is the one the partner gives.
  Without a phone, the agent still places calls, from a number of the
  platform's pool.

## Attachments

```python
attachments = await client.sessions.upload_attachment("bill.pdf")
```

## License

MIT
