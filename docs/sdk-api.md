# Python SDK for managed users

This page lists the SDK API that a partner uses to act for its managed users.
For the raw HTTP API, see [docs/http-api.md](http-api.md).

## Get a key

To get a Platform API secret key, contact support@19pine.ai. Keep the key on
your servers.

## Managed users

All managed users of one partner use that partner's credits. A partner can
limit the total credits that its managed users use.

### Clients

`AsyncPineAI` has REST and real-time Socket.IO. `PineAI` has REST only, with
the same arguments.

| Arguments | Mode |
|---|---|
| `api_key=key` | Partner client. It can call only `platform.*`. Other calls raise `ValueError`. |
| `api_key=key, user_id=user.id` | Acts for that managed user. |
| `api_key` with `access_token` | Not allowed: `ValueError`. |

`user_id` must be a string such as `"123456789"`, never an `int`. A client
with `api_key` keeps its key: `set_token()` and `auth.verify_code()` raise
`ValueError`.

```python
async with AsyncPineAI(api_key=key) as partner_client:
    user = await partner_client.platform.managed_users.create(
        "usr_42", email="ada@example.com", name="Ada Lovelace",
    )
# Store user.id with your own user record.

async with AsyncPineAI(api_key=key, user_id=user.id) as client:
    ...
```

### Platform API (partner client)

| Method | Returns |
|---|---|
| `platform.managed_users.create(external_id, *, email, name, phone=None)` | `ManagedUser`. Idempotent per `external_id`: a repeat returns the stored user unchanged. |
| `platform.managed_users.get(external_id)` | `ManagedUser` |

`ManagedUser` has `id`, `external_id`, `email`, `name`, `phone` and
`created_at`. Pass `id` as `user_id`.

> **Caution:** use only the end user's own email address and phone number, and
> verify them first. Never use an email address or phone number that does not
> belong to that user. Pine trusts these values and does tasks with them: for
> example, the agent can send email for the user. If Pine finds misuse, Pine
> can revoke the key and cancel the partner's access.

### REST (managed-user client)

| Method | Returns |
|---|---|
| `auth.me()` | `AuthIdentity` with `user_id` |
| `sessions.create()` | the new session (`dict`) |
| `sessions.list(...)` | `SessionListResponse` |
| `sessions.get(session_id)` | `SessionInfo` |
| `sessions.send_message(session_id, content, ...)` | `SessionMessageStatus` |
| `sessions.history(session_id, *, limit=30, before_message_id=None)` | `SessionHistoryPage`: one page of stored messages, newest first |
| `sessions.start_task(session_id)` | `dict` |
| `sessions.stop_task(session_id)` | `dict` |
| `sessions.outcomes(session_id, ...)` | `SessionOutcomesPage` |

### How a task runs

The agent talks with the user, prepares the task, and then starts it on its
own, as it does for a normal Pine user. `sessions.get()` returns the session's
`state`. The states are listed in [docs/http-api.md](http-api.md#how-a-task-runs).
Use `sessions.start_task()` only for a task that did not start on its own or
that is paused.

Without Socket.IO, send with `sessions.send_message()` and read the replies
with `sessions.history()`. Each message has `message_id`, `type`, `version`,
`data` (a JSON string) and `created_at`. To read older messages, pass
`next_message_id` as `before_message_id`. When `next_message_id` is `None`,
there are no older messages.
A new reply can take a few seconds to appear in this list. If the `state`
changed but the reply is missing, read again.

### Real time (`AsyncPineAI` only)

We recommend these methods to talk with the agent. You get its replies, its
questions and each change of `state` as they happen.

| Method | What it does |
|---|---|
| `connect()` | Opens Socket.IO and waits for the server to be ready. |
| `join_session(session_id)` | Joins a session. Do this before the methods below. |
| `rebuild(session_id)` | Loads all stored messages of the session. |
| `chat(session_id, content, ...)` | Sends a message and yields the events of the turn. |
| `send_message(session_id, content, ...)` | Sends a message and does not wait. |
| `listen(session_id, ...)` / `subscribe(session_id)` | Yields events without sending. |
| `submit_form_response(session_id, message_id, answers)` | Replies to an agent form. |
| `leave_session(session_id)` | Leaves a session. |
| `disconnect()` / `aclose()` | Closes Socket.IO / closes everything. |

### Answer a form

When the agent needs information, it sends a `session:form_to_user` form. Read
its fields, then pass one answer per field `name` to `submit_form_response()`.

- **Live:** the form arrives as an event from `chat()`, `listen()` or
  `subscribe()`. Use `event.message_id`, and read the fields from `event.data`.
- **Later**, for example after a reconnect: find the newest
  `session:form_to_user` message from the agent in `rebuild()` or
  `sessions.history()`. For `sessions.history()`, read `data` with
  `json.loads()`.

Answer only a form whose `form.is_submitted` is `false`.

```python
from pine_assistant.models import FormToUserData

if event.type == "session:form_to_user":
    form = FormToUserData.model_validate(event.data).form
    answers = {field.name: ... for field in form.fields}
    await client.submit_form_response(session_id, event.message_id, answers)
```

### Errors

Every error from the server is a `PineAIError` with `code` and `status_code`.
Some methods raise a subclass: `PlatformError` from `platform.*`, `AuthError`
from `auth.*`, and `SessionError` from `sessions.list`, `get`,
`send_message`, `history` and `outcomes`. `connect()` raises `ConnectionError`.

The codes for managed users are listed in [docs/http-api.md](http-api.md).
Error messages never contain the key, the `external_id` or the user ID.

### Helpers

| Name | Use |
|---|---|
| `validate_user_id(value)` | Checks the form of a managed user's `id`. |
| `validate_external_id(value)` | Checks the form of an `external_id`. |
| `API_KEY_PREFIXES` | The prefixes that a valid key starts with. |
| `MANAGED_USER_HEADER` | `"Pine-Managed-User-Id"` |
