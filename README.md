# pine-ai

Pine AI SDK for Python. Let Pine AI handle your digital chores.

## Install

```bash
pip install pine-ai          # SDK only
pip install pine-ai[cli]     # SDK + CLI
```

## Quick Start (Async)

```python
from pine_ai import AsyncPineAI

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

## Stream Buffering

Text streaming is buffered internally. You receive one merged text event,
not individual chunks. Work log parts are debounced (3s silence).

## Payment

Pro subscription recommended. For non-subscribers:

```python
from pine_ai import AsyncPineAI
print(f"Pay at: {AsyncPineAI.session_url(session_id)}")
```

## License

MIT
