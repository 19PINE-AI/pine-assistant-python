# Socket.IO for managed users (reference only)

> **Caution:** this page is a reference, not a supported contract.

This page covers only part of the Socket.IO protocol. While you use Socket.IO,
the server can also send events that this page does not describe. For the full
protocol, contact us.

## Connect

- URL: `wss://www.19pine.ai`, path `/api/v2/socket.io/`, WebSocket transport.
- Handshake `auth`: `{"token": "<key>", "managed_user_id": "<id>"}`.
  `managed_user_id` is the managed user's `id` and must be a JSON string.
- When the connection is ready, the server sends `ready`. Send nothing before
  it.
- If the server refuses the handshake, it closes the connection. It sends no
  `connect_error` and no error code, and it never sends `ready`.

The server checks the key and the user again while the connection is open. It
closes the connection if the partner is suspended, the key is revoked, or
the user is banned or archived.

## Envelope

Every event that the client sends is one envelope:

```json
{
  "metadata": {
    "event_id": "<uuid>",
    "request_id": "<uuid>",
    "timestamp": "<ISO 8601>",
    "source": {"role": "user", "user_id": "<id>", "device_id": "<your device id>"}
  },
  "type": "<event name>",
  "payload": {"session_id": "<session id>", "message_id": null, "type": "<event name>", "data": {}}
}
```

`metadata.source.user_id` must be the managed user's `id`. Otherwise the server
replies with `session:error` and does not process the event.

## Events

| Event | Direction | `payload.data` | Use |
|---|---|---|---|
| `session:join` | client → server | `{"since_revision": "<revision>"}` | Join a session. Do this before the other events. Use `"0"` if you have nothing stored. |
| `session:leave` | client → server | `null` | Leave a session. |
| `session:history` | both | request: `{"max_messages", "max_bytes", "order", "from_message_id"}` | Read stored messages. The reply has `messages` and `next_message_id`. |
| `session:messages_since` | both | request: `{"since_revision", "limit"}` | Read the changes after a revision. See "Handle a disconnect". |
| `session:message` | client → server | `{"content": "...", "attachments": []}` | Send a message. |
| `session:message_status` | server → client | `{"status", "message_id", "request_id"}` | Report the state of a message: `received`, `delivered` or a failure. |
| `session:form_to_user` | server → client | `{"form": ...}` | The agent asks the user to fill in a form. |
| `session:form_to_user` | client → server | `{"content": {...}}` | Reply to a form. Set `payload.message_id` and `metadata.request_id` to the values of the form. |

The server also sends other events, such as agent text and session state. See
"The supported surface" in the [README](../README.md).

## Handle a disconnect

A connection can drop at any time. The server does not send again the events
that you missed while you were disconnected. After a drop:

1. Connect again and wait for `ready`.
2. Send `session:join` again for each session.
3. Load what you missed, in one of the two ways below.

### Way 1: read the history again

This way always works.

Read the stored messages with `session:history` or with
`GET /v2/sessions/{id}/messages`. Start from the newest message and read older
pages until `next_message_id` is empty. Then replace your copy of the session.

A stored message can also change or be deleted later. So read all pages, not
only the pages up to the last message that you have.

The Python SDK uses this way: `rebuild()` reads all pages, and `on_reconnect()`
tells you when to call it.

### Way 2: catch up from a revision

This way loads only the changes, but the server does not support it on every
connection. Use Way 1 when it is not available.

Each stored message has `payload.revision`, a number as a string. A message gets
a new, higher revision each time it is created, changed or deleted. Keep the
highest revision that you have stored for each session. A live event without
`revision` is not stored yet; do not use it to move your revision.

1. Join with `{"since_revision": "<your revision>"}`. Use `"0"` if you have
   nothing stored.
2. Read the join reply:
   - No `max_message_revision`: this connection does not support Way 2. Use
     Way 1.
   - `resync: true`: you have nothing stored, your revision is not valid, or
     too many messages changed. Use Way 1, then set your revision to
     `max_message_revision`.
   - `max_message_revision` equals your revision: you missed nothing.
3. Otherwise, send `session:messages_since` with
   `{"since_revision": "<your revision>", "limit": 50}`. The limit is 100 at
   most.
4. The reply has `messages`, `next_revision` and `has_more`. `messages` are in
   the order of their change, oldest first. For each message:
   - `payload.deleted` is `true`: remove the message with that `message_id`.
   - Otherwise: add the message, or replace the message with the same
     `message_id`.
5. Set your revision to `next_revision`. If `has_more` is `true`, go back to
   step 3.

The reply has the `metadata.request_id` of your request, so you can match
replies when you send more than one.
