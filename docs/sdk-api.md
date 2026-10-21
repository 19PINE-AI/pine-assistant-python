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
| `platform.managed_users.create(external_id, *, email, name, phone=None, assistant=None)` | `ManagedUser`. Idempotent per `external_id`: a repeat returns the stored user unchanged. |
| `platform.managed_users.get(external_id)` | `ManagedUser` |
| `platform.managed_users.update(external_id, *, email=None, name=None, phone=None, assistant=None)` | `ManagedUser` after the change. |
| `settings.assistant_identities()` | `AssistantIdentityCatalog`: the identities to choose an assistant from. |

`ManagedUser` has `id`, `external_id`, `email`, `name`, `phone`, `assistant`
and `created_at`. Pass `id` as `user_id`. `assistant` is a
`ManagedUserAssistant` with `identity_id` and `name`, or `None` when the user
has no assistant.

### Change a managed user

`update()` changes only the arguments that you give. An argument that is
`None` is not sent, and its stored value does not change.

- `phone=""` removes the phone. Any other phone must be in E.164 form.
- `external_id` cannot change.
- With no arguments, `update()` changes nothing and returns the stored user.
- The agent uses the new profile from the user's next message. A call in
  progress does not change.

```python
user = await partner_client.platform.managed_users.update("usr_42", phone="+14155550123")
user = await partner_client.platform.managed_users.update("usr_42", phone="")
```

### Assistant identity

The assistant identity sets the assistant's voice and its default name. Each
user can have one assistant. Without one, the user has no assistant, as
before.

1. Read the catalog with `settings.assistant_identities()`. A partner
   client reads it without a user. Use only an identity whose `status` is
   `"active"`. To hear an identity, play its `greeting_audio_url`.
2. Pass `assistant` to `create()` or `update()` as a `dict` with
   `identity_id`, `name` or both:

| `assistant` | Result |
|---|---|
| `{"identity_id": "agt_..."}` | The assistant gets the identity and the identity's own name. On update, this also replaces a custom name: to keep it, send `name` too. |
| `{"identity_id": "agt_...", "name": "Zoe Park"}` | The assistant gets the identity and this name. |
| `{"name": "Zoe Park"}` | `update()` only: the current identity gets a new name. A user with no identity gets 400. |

The `name` has at most 100 characters. It can contain only letters (CJK
included) and spaces between words. One word is enough. A `dict` with other
keys or with values that are not strings raises `ValueError` before any
request is sent.

```python
catalog = await partner_client.settings.assistant_identities()
identity = next(item for item in catalog.identities if item.status == "active")
user = await partner_client.platform.managed_users.update(
    "usr_42", assistant={"identity_id": identity.id, "name": "Zoe Park"},
)
```

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
| `sessions.get_config(session_id)` | `SessionConfig` |
| `sessions.update_config(session_id, *, caller_id)` | `SessionConfig` after the change |
| `sessions.send_message(session_id, content, ..., timezone=None)` | `SessionMessageStatus` |
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
A reply appears in this list only after the agent finishes it. This can take
a minute or more. Socket.IO shows the reply at once. If the `state` changed
but the reply is missing, read again later.

### Caller ID

When the agent calls a third party, such as a business, the call shows a
caller ID. `caller_id` is a session setting that selects it:

| `caller_id` | The call shows |
|---|---|
| `"user"`, or not set | The user's assistant number if it has one, else the user's phone, else a number of the platform's pool. |
| `"platform"` | A number of the platform's pool. |

- Set or change it at any time. The agent uses the value for its next call. A
  call already placed, including its retries, keeps its caller ID. A change
  while the session is paused applies when the session resumes.
- `caller_id` does not change how the agent reaches the user. The agent still
  uses the user's phone for that.
- A 503 `session_config_push_failed` means that the value is saved, but the
  running agent did not get it yet. Send the same value again.
- Any other value raises `ValueError` before any request is sent.

```python
session = await client.sessions.create()
await client.sessions.update_config(session["id"], caller_id="platform")
await client.sessions.send_message(session["id"], "Call the clinic to book a visit")
```

### The user's time

The agent must know the user's time zone to understand words such as
"tomorrow at 9 am". Pass the user's time zone as `timezone` with each
message. The SDK then sends the user's current local time with the message,
as the Pine app does.

- `timezone` is an IANA time zone name, such as `"America/New_York"`, or a
  `tzinfo`. Daylight saving time is handled for you.
- An unknown name raises `ValueError` before any request is sent.
- If a session never got a time zone, the agent uses UTC. After one message
  with it, the agent keeps that time zone until a message brings a new one.

```python
await client.sessions.send_message(session_id, "Book a table for tomorrow at 7 pm", timezone="America/New_York")
```

`chat()` and `send_message()` on Socket.IO take the same `timezone`.

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
from `auth.*`, and `SessionError` from `sessions.list`, `get`, `get_config`,
`update_config`, `send_message`, `history` and `outcomes`. `connect()` raises
`ConnectionError`.

`PlatformError` from `platform.managed_users.update()`:

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_request` | A value is not valid. For example: the phone is not E.164, the identity is unknown or not active, or a name is sent for a user with no identity. Nothing is changed. |
| 401 | `unauthorized` | The key is unknown or revoked. |
| 403 | `platform_access_suspended` | Your access is suspended. |
| 404 | `platform_managed_user_not_found` | No managed user of yours has this `external_id`. |
| 429 | `too_many_requests` | Too many requests. |

`SessionError` from `sessions.get_config()` and `update_config()`: 403
`forbidden` when the client is neither a managed user nor a test account, and 503
`session_config_push_failed` (see "Caller ID").

The codes for managed users are listed in [docs/http-api.md](http-api.md).
Error messages never contain the key, the `external_id` or the user ID.

### Helpers

| Name | Use |
|---|---|
| `validate_user_id(value)` | Checks the form of a managed user's `id`. |
| `validate_external_id(value)` | Checks the form of an `external_id`. |
| `API_KEY_PREFIXES` | The prefixes that a valid key starts with. |
| `MANAGED_USER_HEADER` | `"Pine-Managed-User-Id"` |
