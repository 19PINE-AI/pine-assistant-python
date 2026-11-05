---
name: pine-rest
description: Connect to Pine AI over REST to delegate supported real-world tasks and check, continue, or stop existing Pine tasks. See the skill instructions for the integration's currently supported scope.
---

# Pine Assistant

Pine AI is an assistant that carries out real-world tasks. This REST integration currently supports starting phone tasks only. Pine can search for public business phone numbers and do the preparation needed to complete those phone tasks. This is the integration's current release scope, not a limit on Pine's overall capabilities. When a user's goal can be handled by phone, treat it as a phone task without requiring them to explicitly say “call.” If they explicitly request a new task through email, browser operation, fax, or another non-phone method, explain that this integration does not yet support starting that kind of task; do not silently change it to a phone call.

For a new or continuing phone task, read [Phone tasks](references/phone-tasks.md) before composing the task instruction or interpreting call results. Existing Pine tasks can still be queried, answered, or stopped using the available REST operations; do not reject an existing task merely because it used another execution method. Ordinary informational questions can be answered in this conversation without delegating to Pine.

A clear task request is authorization: do not ask a separate “start?” question. Gather only information that is necessary to act.

Use the [agent REST OpenAPI](https://www.19pine.ai/api/docs/agent-rest.openapi.json) for request and response schemas. Its server `/api` resolves on `https://www.19pine.ai`; routes below are relative to that server. The API provides capabilities to create, list, and get sessions; send a message; read history and outcomes; and stop an agent. It does not require a separate task-start call for a new task.

Authenticate with the normal user's bearer from [Instinct setup](../../install/instinct.md), stored in the client's credential store. Send it only to Pine, not in prompts or logs. On authentication failure, repeat setup with the same account and verify its user ID before continuing; do not automatically replay writes.

## Start or continue work

1. When the user is following up, or an existing task may match, list sessions and inspect plausible candidates. Reuse the matching session. If several are plausible, ask the user which one they mean; do not create a replacement session.
2. For a new task, create one session, then send one clear instruction with the objective, constraints, and relevant context. Session creation alone does not begin work; sending the instruction may let Pine continue it automatically.
3. Do not separately launch the same real-world task through another executor. Keep execution of the delegated task with Pine; include any useful research you have already gathered.

Do not claim an external action happened or the task is complete from a session state, timestamp, or a `received`/`delivered` message receipt. A delivered message only confirms dispatch. A `delivery_failed` receipt does not establish that nothing was saved; inspect history before sending again. A session can remain processing after Pine has replied.

## Read updates and respond

After sending an instruction or answer, read the same session's history for questions, forms, state, task-finished content, connection or payment requests, and Pine's replies. Read outcomes as well when they are relevant. Present Pine's actual reply and result facts faithfully; explain them if useful, but do not manufacture a summary, outcome, or terminal status.

Fetch bounded history pages with `GET /v2/sessions/{session_id}/messages`. Messages are newest-first; parse each message's `data` JSON string. Follow `next_message_id` as `before_message_id` only while an older page is needed; an empty cursor ends pagination. Retain the latest `version` of duplicate messages. Replies appear only after persistence, so missing text after a state change calls for a later read, not resending the instruction.

Pine does not send you updates automatically, so you must actively read history to follow progress. After sending a message and after each history read, decide when to check again based on Pine’s latest reply and the task’s expected progress. If timing is uncertain, prefer checking sooner rather than stopping follow-up; avoid tight polling loops. An acknowledgement is not completion. Defer checks for scheduled work, pause for required user input, and stop when the task finishes.

If you have a wait tool, use it before another bounded query; do not promise updates after you stop checking. A later conversation can recover by listing sessions and reading the chosen session's history/outcomes.

For an ordinary Pine question, collect the user's answer in this conversation and send it to the same session. Keep sensitive form values, especially L2/L3 values, out of unnecessary narration.

## Forms and web actions

Typed form, authorization, and location responses are not available through this REST integration. When Pine requests one, payment, account connection, phone verification, or another web intervention, direct the user to `https://www.19pine.ai/app/chat/{session_id}` with the actual session ID. An ordinary text message is not equivalent to completing a typed response. After the user finishes, query the original session again; do not resend the task instruction or create a new session.

If your safety checks require user confirmation before sending information to Pine, briefly state what would be sent and ask the user to confirm. Avoid security jargon unless asked, and proceed after confirmation where permitted.

## Stop a task

When the user asks to stop a Pine task, identify the matching session and call `POST /v2/sessions/{session_id}/stop`. A clear request is sufficient; clarify only when the target is ambiguous. Pine decides whether that task can be stopped. If it rejects the request, explain the returned reason rather than creating a replacement task or trying another execution path.

Stop pauses an active task; it does not end the session or prove the objective succeeded. Query the same session and history to report the resulting state rather than claiming an external action has already stopped. Do not stop other tasks merely to free a task slot unless the user has authorized stopping those tasks. Resume a stopped task in the Pine task page; sending a message alone does not resume it.

If stopping the task times out or returns an unknown result, query the same session and its history before deciding what to do next. The state may have changed before notification to the agent failed; do not blindly repeat the write or claim that downstream execution has stopped.

## Uncertain writes

Creating sessions and sending messages are non-idempotent. Do not blindly retry when a write times out, has an unknown result, or reports a transport/server failure. For an uncertain creation, first list recent sessions; for an uncertain message, first read that session's history and reconcile the result. Ask for direction when recovery still cannot establish what happened.
