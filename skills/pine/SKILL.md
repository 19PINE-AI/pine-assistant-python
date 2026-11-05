---
name: pine
description: Connect to Pine AI over Socket.IO or REST to delegate supported real-world tasks and check, continue, or stop existing Pine tasks. See the skill instructions for the integration's currently supported scope.
---

# Pine Assistant

Pine AI is an assistant that carries out real-world tasks. This integration currently supports starting phone tasks only. Pine can search for public business phone numbers and do the preparation needed to complete those phone tasks. This is the integration's current release scope, not a limit on Pine's overall capabilities. When a user's goal can be handled by phone, treat it as a phone task without requiring them to explicitly say “call.” If they explicitly request a new task through email, browser operation, fax, or another non-phone method, explain that this integration does not yet support starting that kind of task; do not silently change it to a phone call.

For a new or continuing phone task, read [Phone tasks](references/phone-tasks.md) before composing the task instruction or interpreting call results. Existing Pine tasks can still be queried, answered, or stopped using the available operations; do not reject an existing task merely because it used another execution method. Ordinary informational questions can be answered in this conversation without delegating to Pine.

A clear task request is authorization: do not ask a separate “start?” question. Gather only information that is necessary to act.

Use the [client AsyncAPI](https://5t4g1ng.19pine.ai/api/docs/client.asyncapi.json) for Socket.IO connection details and event payloads, and the [client OpenAPI](https://5t4g1ng.19pine.ai/api/docs/client.openapi.json) for HTTP request and response schemas. Its server `/api` resolves on `https://5t4g1ng.19pine.ai`; routes below are relative to that server. The API provides capabilities to create, list, and get sessions; send a message; read history and outcomes; and stop an agent. It does not require a separate task-start call for a new task.

This setup temporarily targets staging. Use `https://5t4g1ng.19pine.ai` for HTTP, Socket.IO, and linked API documentation, overriding any production host in the contracts or SDK defaults. With the SDK, set `base_url` to this origin.

## Choose a transport

Prefer Socket.IO when the execution environment can keep a client connected and receive callbacks. Use the Pine SDK if available, or a compatible Socket.IO client following the client AsyncAPI; Python is not required. Wait for `ready`, join the selected session, and recover its history before sending or interpreting new work. Use REST for session management and other HTTP operations as needed.

If the environment only supports HTTP requests, use REST directly. If Socket.IO fails, attempt bounded connection recovery, then use REST with the same account and session after reconciling history. Do not replay writes when switching transports. Authentication, billing, account restrictions, and rejected actions require their own resolution; switching transports does not bypass them.

## Authentication

Use a normal user bearer token, not a partner API key or MCP OAuth token. If the API definition is unavailable, report that setup is incomplete rather than inventing a request.

1. Ask for the user's Pine account email. Explain that verifying an email without an existing account creates one; use their existing account when intended.
2. Send `POST /v2/auth/email/request` with `{"email":"..."}` and retain `data.request_token`. Ask for the emailed code, then send `POST /v2/auth/email/verify` with `email`, `code`, and `request_token`.
3. Save `data.access_token` in secure credential storage and use `Authorization: Bearer <access_token>` for authenticated HTTP requests and the protocol's token handshake for Socket.IO. Send it only to Pine, not in prompts or logs. Do not print it or save it in the Skill. If secure storage is unavailable, ask the user to configure the credential securely instead.
4. Verify with `GET /v2/auth/me` and retain `data.user_id`. Setup does not create sessions, send task instructions, or make calls.

Successful authentication renews the token's idle expiry (30 days in the production configuration). On a 401 for an expired or invalid token, repeat email login with the same account and verify its user ID before continuing existing tasks. Do not automatically replay a failed write after signing in; follow the recovery rules below.

## Start or continue work

1. When the user is following up, or an existing task may match, list sessions with `ensure_copilot=false` and inspect plausible candidates. Reuse the matching session. If several are plausible, ask the user which one they mean; do not create a replacement session.
2. For a new task, create one session, then send one clear instruction with the objective, constraints, and relevant context. Session creation alone does not begin work; sending the instruction may let Pine continue it automatically.
3. Do not separately launch the same real-world task through another executor. Keep execution of the delegated task with Pine; include any useful research you have already gathered.

Do not claim an external action happened or the task is complete from a session state, timestamp, or a `received`/`delivered` message receipt. A delivered message only confirms dispatch. A `delivery_failed` receipt does not establish that nothing was saved; inspect history before sending again. A session can remain processing after Pine has replied.

## Read updates and respond

After sending an instruction or answer, follow live events when Socket.IO is available and read the same session's history for questions, forms, state, task-finished content, connection or payment requests, and Pine's replies. Read outcomes as well when they are relevant. Present Pine's actual reply and result facts faithfully; explain them if useful, but do not manufacture a summary, outcome, or terminal status.

Fetch bounded history pages with `GET /v2/sessions/{session_id}/messages`. Messages are newest-first; parse each message's `data` JSON string. Follow `next_message_id` as `before_message_id` only while an older page is needed; an empty cursor ends pagination. Refresh messages with the same ID from the latest history response; `version` describes the encoding, not the edit revision. Use live-event revisions only as defined by the protocol. Replies appear only after persistence, so missing text after a state change calls for a later read, not resending the instruction.

With Socket.IO, consume live replies, forms, and task events; assemble text parts by message ID. Rebuild history after joining, reconnecting, or unexplained silence, since missed live events are not replayed. A chat turn ending is not task completion; keep following the session while work runs.

With REST, Pine does not send you updates automatically, so you must actively read history to follow progress. After sending a message and after each history read, decide when to check again based on Pine’s latest reply and the task’s expected progress. If timing is uncertain, prefer checking sooner rather than stopping follow-up; avoid tight polling loops. An acknowledgement is not completion. Defer checks for scheduled work, pause for required user input, and stop when the task finishes.

If you have a wait tool, use it before another bounded query; do not promise updates after you stop checking. A later conversation can recover by listing sessions and reading the chosen session's history/outcomes.

For an ordinary Pine question, collect the user's answer in this conversation and send it to the same session. Keep sensitive form values, especially L2/L3 values, out of unnecessary narration.

## Forms and web actions

Prefer a verified typed form submission when the client supports it, such as the SDK's `submit_form_response`: recover the original form from authenticated history, retain its message/request IDs and field privacy levels, and submit the user's actual answers. Follow the documented event contract for other supported structured responses; do not assume the SDK implements every authorization or location interaction.

When typed submission is unavailable, including over REST, send ordinary answers to form questions and choices as text in the same session. This includes normal contact details the user authorizes sharing. Read Pine's response and answer further clarification questions as needed; a structured event alone is not a reason to send the user to the web. Do not claim the typed form was completed merely because text was delivered. For sensitive fields requiring the protected form path, or when Pine still requires an unsupported interaction, use the Pine task page.

Actual OAuth/account connection, payment, phone verification, and other actions requiring the web must be completed at `https://5t4g1ng.19pine.ai/app/chat/{session_id}` with the actual session ID. Text answers do not complete those actions. After the user finishes, query the original session again; do not resend the task instruction or create a new session.

If your safety checks require user confirmation before sending information to Pine, briefly state what would be sent and ask the user to confirm. Avoid security jargon unless asked, and proceed after confirmation where permitted.

## Stop a task

When the user asks to stop a Pine task, identify the matching session and call `POST /v2/sessions/{session_id}/stop`. A clear request is sufficient; clarify only when the target is ambiguous. Pine decides whether that task can be stopped. If it rejects the request, explain the returned reason rather than creating a replacement task or trying another execution path.

Stop pauses a processing task; in chat, it prevents new actions while an in-flight action may finish. It does not end the session or prove the objective succeeded. Query the same session and history to report the resulting state rather than claiming an external action has already stopped. Do not stop other tasks merely to free a task slot unless the user has authorized stopping those tasks. Resume a stopped task in the Pine task page; sending a message alone does not resume it.

If stopping the task times out or returns an unknown result, query the same session and its history before deciding what to do next. The state may have changed before notification to the agent failed; do not blindly repeat the write or claim that downstream execution has stopped.

## Uncertain writes

Creating sessions and sending messages or form responses are non-idempotent. Do not blindly retry when a write times out, has an unknown result, or reports a transport/server failure. For an uncertain creation, first list recent sessions; for an uncertain message or form response, first read that session's history and reconcile the result. Ask for direction when recovery still cannot establish what happened.
