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
| `email` | no | email address; omit it or send `""` for a user without an email |
| `name` | yes | 1-100 characters |
| `phone` | no | E.164, for example `+14155550100`; a number you have verified |
| `assistant` | no | `{"identity_id": "...", "name": "..."}`; see "Assistant identity" |

No value may be `null`: omit a field instead. Unknown fields return 400.

> **Caution:** use only the end user's own email address and phone number, and
> verify them first. Never use an email address or phone number that does not
> belong to that user. Pine trusts these values and does tasks with them: for
> example, the agent can send email for the user. If Pine finds misuse, Pine
> can revoke the key and cancel the partner's access.

The call is idempotent per `external_id`. The first call returns 201. A
repeated call returns 200 with the stored user unchanged, even if the fields
you send are different.

The response `data` is the managed user:
`{"id", "external_id", "email", "name", "phone", "assistant", "created_at"}`.
`email` is `""` when the user has no email. A managed user without an email
cannot do email tasks.
`phone` is `null` when the user has no phone. `assistant` is
`{"identity_id", "name"}`, or `null` when the user uses Pine's default assistant.

Errors: 400 `invalid_request`, 401 `unauthorized`, 403
`platform_access_suspended`, 409 `platform_managed_user_limit_reached`.

### Get a managed user

`GET /platform/v1/managed-users/{external_id}` with `Authorization: Bearer <key>`.
Percent-encode `external_id` in the path (for example `|` is `%7C`).

It returns the same managed user as create. A user that does not exist, or that
belongs to another partner, returns 404 `platform_managed_user_not_found`.

### Update a managed user

`PATCH /platform/v1/managed-users/{external_id}` with `Authorization: Bearer <key>`.

Send only the fields that you change. A field that you omit keeps its stored
value. `{}` changes nothing.

| Field | Rule |
|---|---|
| `email` | email address, or `""` to remove the email |
| `name` | 1-100 characters |
| `phone` | E.164, or `""` to remove the phone |
| `assistant` | `{"identity_id": "...", "name": "..."}`; either member can be omitted; see "Assistant identity" |

The rules for `null` and unknown fields are the same as for create.
`external_id` cannot change. The agent uses the new profile from the user's
next message. A call in progress does not change. The response is the same as
for get.

Errors: 400 `invalid_request` (nothing is changed), 401 `unauthorized`, 403
`platform_access_suspended`, 404 `platform_managed_user_not_found`.

### Assistant identity

The assistant is the voice of Pine on the phone. It places calls for the
user, for example to a business, and it answers calls for the user. Each
user has one assistant. By default, it is Pine's built-in assistant.

You can choose another identity from the catalog. The identity sets the
assistant's voice and the name that it gives on calls. You can also give it
your own name.

The catalog:

`GET /v2/app/settings/assistance_identities?identities=catalog`

You can send it without any header. The response `data` is
`{"version", "identities": [{"id", "first_name", "last_name", "locale", "description", "greeting_audio_url", "status"}]}`.
Use only an identity whose `status` is `"active"`. To hear an identity, play
its `greeting_audio_url`.

| `assistant` | Result |
|---|---|
| `{"identity_id": "agt_..."}` | The assistant gets the identity and the identity's own name. On update, this also replaces a custom name: to keep it, send `name` too. |
| `{"identity_id": "agt_...", "name": "Zoe Park"}` | The assistant gets the identity and this name. |
| `{"name": "Zoe Park"}` | Update only: the current identity gets a new name. A user with no identity gets 400. |

The `name` has at most 100 characters. It can contain only letters (CJK
included) and spaces between words. One word is enough. A user created
without `assistant` uses Pine's default assistant.

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
| `GET /v2/sessions/{id}/config` | Returns the session config, for example `{"caller_id": "platform"}`, or `{}`. | `sessions.get_config()` |
| `PUT /v2/sessions/{id}/config` | Sets the caller ID. Body: `{"caller_id": "user"}` or `{"caller_id": "platform"}`. See "Caller ID". | `sessions.update_config()` |
| `POST /v2/sessions/{id}/messages` | Sends a message. Body: `{"content": "...", "client_now_date": "..."}`; `client_now_date` is optional, see "The user's time". | `sessions.send_message()` |
| `GET /v2/sessions/{id}/messages` | Reads the stored messages, newest first. | `sessions.history()` |
| `POST /v2/sessions/{id}/start` | Starts a prepared or paused task, or continues a stopped session. See "Stop and continue". | `sessions.start_task()` |
| `POST /v2/sessions/{id}/stop` | Stops the agent's work and keeps the session. See "Stop and continue". | `sessions.stop_task()` |
| `GET /v2/sessions/{id}/outcomes` | Lists the session's outcomes. | `sessions.outcomes()` |

Routes that make the agent work can also return these errors:

| Status | Code | Cause |
|---|---|---|
| 402 | `platform_spend_limit_reached` | The partner's credit limit is reached. |
| 409 | `platform_session_limit_reached` | Start: the partner's running-task limit is reached. |
| 409 | `platform_session_state_conflict` | Message: the session is finished or cancelled. |

Stop and start can also return:

| Status | Code | Cause |
|---|---|---|
| 400 | `session_state_not_allowed` | Stop: the session is finished or cancelled. |
| 400 | `session_not_in_ready_state` | Start: there is nothing to start or continue. |
| 429 | `too_many_requests` | Stop or start: another start or stop of the same session is in progress. Send the request again. |

`GET /v2/sessions/{id}/messages` takes two optional query values: `limit`
(1-100, default 30) and `before_message_id`. Each message has `message_id`,
`type`, `version`, `data` (a JSON string) and `created_at`. To read older
messages, send the returned `next_message_id` as `before_message_id`. An empty
`next_message_id` means there are no older messages.
A reply appears in this list only after the agent finishes it. This can take
a minute or more. Socket.IO shows the reply at once. If the `state` changed
but the reply is missing, read again later.

Other `/v2` routes accept the key and header in the same way. This page does
not describe them.

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
- Any other key or value returns 400 `invalid_request`.
- 503 `session_config_push_failed` means that the value is saved, but the
  running agent did not get it yet. Send the same request again.

### The user's time

The agent must know the user's time zone to understand words such as
"tomorrow at 9 am". Send the user's current local time, with its UTC offset,
as `client_now_date` with each message. The Pine app does the same.

- The value is an RFC 3339 time with an offset, for example
  `2026-10-21T09:00:00-04:00`. Pine uses only the offset.
- A value without an offset is ignored.
- If a session never got `client_now_date`, the agent uses UTC. After one
  message with it, the agent keeps that offset until a message brings a new
  one.
- On Socket.IO, put the same field in the `session:message` data.

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
| `task_paused` | The task is paused because you stopped it, or because a limit is reached. Call `start` to continue (after the limit allows it). |
| `task_finished` | The task is done. Read the result with `outcomes`. |
| `task_cancelled` | The task was cancelled. |

### Stop and continue

`POST /v2/sessions/{id}/stop` stops the agent and keeps the session:

| `state` when you stop | What happens |
|---|---|
| `chat` | The agent starts no new action, and the task does not become ready. A reply or action already in progress finishes. The `state` stays `chat`. |
| `task_ready` | The task does not start on its own. |
| `task_processing` | The task pauses (`task_paused`). The agent starts no new action. A phone call that is not dialed yet is not dialed, and a call in progress is hung up. |
| `init`, `task_paused`, or already stopped | Nothing changes. The answer is 200. |

To continue, call `POST /v2/sessions/{id}/start`. A message sent while the
session is stopped is saved but does not continue it. You can send the user's
message before or after `start`; before is better, because the agent then reads
the message before it goes on. If you start first, the agent may continue the
stopped step (for example, place the call) before the message arrives.

If `start` is refused by a limit, the session stays stopped; call `start` again
later.

### Talk with the agent

We recommend Socket.IO ([docs/socketio.md](socketio.md)) to talk with the
agent. You get its replies, its questions and each change of `state` as they
happen.

Over HTTP only, send messages with `POST /v2/sessions/{id}/messages`. Read the
replies and the `state` with `GET /v2/sessions/{id}/messages` and
`GET /v2/sessions/{id}`.
