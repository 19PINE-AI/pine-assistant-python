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
await client.connect()

session = await client.sessions.create()
await client.join_session(session["id"])
await client.rebuild(session["id"])          # load the session's messages

async for event in client.chat(session["id"], "Negotiate my Comcast bill"):
    print(event.type, event.data)

await client.disconnect()
```

A client tracks one session. Concurrent sessions need one client each.

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
| `session:input_state` | Whether input is accepted, and the reason when it is not. This is where a blocked session says why |
| `session:message_status` | What became of a message you sent — the only way to tell a rejected or rate-limited one from one still being worked on |
| `session:required_action` | Whether the session is waiting on you |
| `session:update_title` | The session title, as the agent revises it |
| `session:restriction` | An account restriction. The only statement that a task will not complete |

**Interaction**

| Event | What it is for |
|---|---|
| `session:form_to_user` | Structured data collection — how a task asks for the account details it needs to act. Sent both ways under this name, and the most frequent interaction here |

**Task and result**

| Event | What it is for |
|---|---|
| `session:task_ready` | What the task will cost in credits, and whether it is authorised. When the balance covers it the server starts the task itself and this is informational; when it does not, the session waits |
| `session:task_finished` | The result. `completion.result_title`, `result_description` and `outcome_narrative` carry the text; `completion.summary` is quantified, and `brief` is its only prose |
| `session:tool_status` | The record of one asynchronous operation. An outbound call reports here: the number, the duration, the credits, and `summary.text`. It updates in place, reusing its `message_id`, so expect several with the same one |

Payloads may gain fields at any time — tolerate fields you do not recognise.

A `tool_call` step in `session:llm_thinking` describes the same operation as the
matching `session:tool_status`. Do not show both.

A turn commonly delivers `session:text_part` alone: the composer reopens once
the agent has finished speaking, and the complete `session:text` is the durable
record, read back from history. Assemble the parts by `message_id` rather than
waiting for the complete message to arrive live.

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

## Blocked sessions

When the composer is disabled, `session:input_state` carries the reason. Read it
from there rather than inferring it from which events did or did not arrive.

```python
from pine_assistant import InputState, S2CEvent

if event.type == S2CEvent.SESSION_INPUT_STATE:
    state = InputState.model_validate(event.data)
    if state.awaiting_credits:
        ...   # cost is on session:task_ready; retry once the balance is restored
    if state.needs_phone_verification:
        ...   # no in-session remedy
```

An expired session has no reason code of its own — it presents only as a
disabled composer. Expiry is the `is_stale` field on the session object, over
REST. On finding one expired, create a new session and reference the old one in
your first message:

```python
new = await client.sessions.create()
client.send_message(new["id"], "...", referenced_sessions=[{"session_id": old_id}])
```

## Before an account is used

Two conditions have no remedy once a session is running:

- **Metered billing.** The account must be billed against a credit balance. On
  the alternative path a session halts at a payment step the SDK cannot answer.
- **Phone verification.** Must be completed at provisioning time.

## Attachments

```python
attachments = await client.sessions.upload_attachment("bill.pdf")
```

## License

MIT
