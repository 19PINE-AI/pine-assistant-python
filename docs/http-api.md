# HTTP API for managed users

This page describes the raw HTTP API that a partner (a business that builds on
Pine) uses to act for its managed users.

## Get a key

To get a Platform API secret key, contact support@19pine.ai.

Keep the key on your servers. Do not put it in browser or mobile code, in
prompts, or in logs.

## Managed users

All managed users of one partner use that partner's credits. A partner can
limit the total credits that its managed users use.

Each end user of your product is a *managed user*. You name the user with your
own `external_id` when you create it. Pine gives it a Pine user ID, `id`. Store
`id` in your own user table: you need it for every request that acts for the
user.

### Response format

All paths start with `https://www.19pine.ai/api`.

- Success: `{"status": "success", "data": ...}`
- Error: `{"status": "error", "error": {"code": "...", "message": "..."}}`

Use `error.code` in your code. The `message` text can change.

### Create a managed user

`POST /platform/v1/managed-users` with `Authorization: Bearer <key>`.

| Field | Required | Rule |
|---|---|---|
| `external_id` | yes | 1-128 letters, digits, `.`, `_`, `:`, `|`, `-`; must not start with `.`; no personal data |
| `email` | yes | email address |
| `name` | yes | 1-100 characters |
| `phone` | no | E.164, for example `+14155550100`; a number you have verified |

> **Caution:** use only the end user's own email address and phone number, and
> verify them first. Never use an email address or phone number that does not
> belong to that user. Pine trusts these values and does tasks with them: for
> example, the agent can send email for the user. If Pine finds misuse, Pine
> can revoke the key and cancel the partner's access.

The call is idempotent per `external_id`. The first call returns 201. A
repeated call returns 200 with the stored user unchanged, even if the fields
you send are different.

The response `data` is the managed user:
`{"id", "external_id", "email", "name", "phone", "created_at"}`. `phone` is
`null` when you did not give one.

Errors: 400 `invalid_request`, 401 `unauthorized`, 403
`platform_access_suspended`, 409 `platform_managed_user_limit_reached`.

### Get a managed user

`GET /platform/v1/managed-users/{external_id}` with `Authorization: Bearer <key>`.
Percent-encode `external_id` in the path (for example `|` is `%7C`).

It returns the same managed user as create. A user that does not exist, or that
belongs to another partner, returns 404 `platform_managed_user_not_found`.

### Act for a managed user

Send both headers on a `/v2` request:

```
Authorization: Bearer <key>
Pine-Managed-User-Id: <id>
```

`<id>` is the `id` from create or get, as a decimal string: 1 to
9223372036854775807, with no sign, spaces or leading zeros. Do not send
`Pine-Managed-User-Id` on `/platform/v1` requests.

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_request` | One of the two headers is missing, the header is sent more than once, or the ID is not in the form above. |
| 401 | `unauthorized` | The key is wrong or revoked. |
| 403 | `platform_access_suspended` | The partner is suspended. |
| 403 | `platform_managed_user_unavailable` | The managed user is banned or archived. |
| 403 | `platform_route_not_allowed` | The route is closed to keys: login tickets, phone verification, login linking, push registration, purchases and subscriptions, and computer settings. |
| 404 | `platform_managed_user_not_found` | The ID is not a managed user of this partner. |

### Session routes

These routes take the two headers above.

| Request | What it does | SDK |
|---|---|---|
| `GET /v2/auth/me` | Returns `{"user_id": "<id>"}`. | `auth.me()` |
| `POST /v2/sessions` | Creates a session. | `sessions.create()` |
| `GET /v2/sessions` | Lists the user's sessions. | `sessions.list()` |
| `GET /v2/sessions/{id}` | Returns one session. | `sessions.get()` |
| `POST /v2/sessions/{id}/messages` | Sends a message. Body: `{"content": "..."}`. | `sessions.send_message()` |
| `GET /v2/sessions/{id}/messages` | Reads the stored messages, newest first. | `sessions.history()` |
| `POST /v2/sessions/{id}/start` | Starts a prepared or paused task. | `sessions.start_task()` |
| `POST /v2/sessions/{id}/stop` | Stops a running task. | `sessions.stop_task()` |
| `GET /v2/sessions/{id}/outcomes` | Lists the session's outcomes. | `sessions.outcomes()` |

Routes that make the agent work can also return these errors:

| Status | Code | Cause |
|---|---|---|
| 402 | `platform_spend_limit_reached` | The partner's credit limit is reached. |
| 409 | `platform_session_limit_reached` | Start: the partner's running-task limit is reached. |
| 409 | `platform_session_state_conflict` | Message: the session is finished or cancelled. |

`GET /v2/sessions/{id}/messages` takes two optional query values: `limit`
(1-100, default 30) and `before_message_id`. Each message has `message_id`,
`type`, `version`, `data` (a JSON string) and `created_at`. To read older
messages, send the returned `next_message_id` as `before_message_id`. An empty
`next_message_id` means there are no older messages.

Other `/v2` routes accept the key and header in the same way. This page does
not describe them.

### How a task runs

1. Create a session and send the user's request as a message.
2. The agent talks with the user to get the information that it needs.
3. The agent prepares the task. The task then starts on its own, as it does for
   a normal Pine user.
4. The agent does the task. It can ask the user more questions.
5. The task ends.

The session's `state` shows where the task is:

| `state` | Meaning |
|---|---|
| `init` | The session is new. |
| `chat` | The agent talks with the user. |
| `task_ready` | The task is prepared. It usually starts on its own. If the running-task limit is reached, it stays here: call `start` later. |
| `task_processing` | The task runs. |
| `task_paused` | The task is paused, for example because the credit limit is reached. When the cause is removed, call `start` to continue. |
| `task_finished` | The task is done. Read the result with `outcomes`. |
| `task_cancelled` | The task was cancelled. |

### Talk with the agent

We recommend Socket.IO ([docs/socketio.md](socketio.md)) to talk with the
agent. You get its replies, its questions and each change of `state` as they
happen.

Over HTTP only, send messages with `POST /v2/sessions/{id}/messages`. Read the
replies and the `state` with `GET /v2/sessions/{id}/messages` and
`GET /v2/sessions/{id}`.
