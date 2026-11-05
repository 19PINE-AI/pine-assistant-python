# Set up Pine in Instinct

Use REST with Instinct's API Key credential storage. Python, MCP, and native
Skill installation are not required. Read the [Pine REST skill](../skills/pine-rest/SKILL.md)
and its references; retain that complete link for future conversations if you
cannot save the files.

Use the [agent REST OpenAPI](https://www.19pine.ai/api/docs/agent-rest.openapi.json)
with API base `https://www.19pine.ai/api`. If it is unavailable, report that setup
is incomplete rather than inventing an API definition.

1. Ask for the user's Pine account email. Explain that verifying an email without
   an existing account creates one; use their existing account when intended.
2. Send `POST /v2/auth/email/request` with `{"email":"..."}` and retain
   `data.request_token`. Ask the user for the emailed code, then send
   `POST /v2/auth/email/verify` with `email`, `code`, and `request_token`.
3. Save `data.access_token` in API Key storage and use
   `Authorization: Bearer <access_token>` for authenticated Pine requests. This
   is a normal user token, not a partner API key or MCP OAuth token. Do not print
   it or save it in the Skill. If secure storage is unavailable, ask the user to
   configure the credential securely instead.
4. Verify with `GET /v2/auth/me` and retain `data.user_id`. Setup does not create
   sessions, send task instructions, or make calls.

Successful authentication renews the token's idle expiry (30 days in the
production configuration).
If it expires or is revoked, repeat email login and check that `/v2/auth/me`
returns the same user ID before continuing existing tasks. Do not automatically
replay a failed write after signing in; follow the Skill's recovery rules.
