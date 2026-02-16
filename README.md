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

async for event in client.chat(session["id"], "Negotiate my Comcast bill"):
    print(event.type, event.data)

await client.disconnect()
```

## Quick Start (CLI)

```bash
pine auth login                          # Email verification
pine chat                                # Interactive REPL
pine send "Negotiate my Comcast bill"    # One-shot message
pine sessions list                       # List sessions
pine task start <session-id>             # Start task (Pro)
```

## Handling Events

Pine AI behaves like a human assistant. After you send a message, it sends
acknowledgments, then work logs, then the real response (form, text, or task_ready).
**Don't respond to acknowledgments** — only respond to forms, specific questions,
and task lifecycle events, or you'll create an infinite loop.

## Continuing Existing Sessions

```python
# List all sessions
result = await client.sessions.list(limit=20)

# Continue an existing session
await client.join_session(existing_session_id)
history = await client.get_history(existing_session_id)
async for event in client.chat(existing_session_id, "What is the status?"):
    ...
```

## Attachments

```python
# Upload a document for dispute tasks
attachments = await client.sessions.upload_attachment("bill.pdf")
```

## Stream Buffering

Text streaming is buffered internally. You receive one merged text event,
not individual chunks. Work log parts are debounced (3s silence).

## Payment

Pro subscription recommended. For non-subscribers:

```python
from pine_assistant import AsyncPineAI
print(f"Pay at: {AsyncPineAI.session_url(session_id)}")
```

## License

MIT
