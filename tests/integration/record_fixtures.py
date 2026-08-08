"""Record protocol fixtures from a live session.

The fixtures under tests/protocol/fixtures are the baseline the contract tests
run against. Recording them from a running server is what makes them evidence
rather than a restatement of the documentation — and re-recording is how server
drift gets noticed, since nothing else checks that the protocol scope and the
implementation still agree.

    PINE_ACCESS_TOKEN=... PINE_USER_ID=... python -m tests.integration.record_fixtures

Every envelope seen is written to --raw-dir; only supported events replace a
fixture, and each replacement flips its provenance entry to "recorded".
Unsupported events are reported but never recorded — we do not maintain them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import re
from datetime import datetime, timezone
from typing import Any

from pine_assistant import AsyncPineAI, S2CEvent, is_supported_event
from tests.protocol.fake import SESSION_ID as PLACEHOLDER_SESSION_ID

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "protocol" / "fixtures"
DEFAULT_PROMPT = "Please call +1 415-555-0199 and ask what time Saturday's dinner starts. Make exactly one attempt — if it does not connect, stop and tell me. Do not retry."

# Values that identify a person or an account never reach a checked-in fixture.
# The phone pattern requires a leading "+" or separators: Pine's identifiers are
# long digit runs, and a looser pattern rewrites them into a fake phone number.
REDACTIONS = (
    (re.compile(r"\+\d[\d\-\s().]{7,}\d"), "+15555550100"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "+15555550100"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "someone@example.com"),
)

# Fields whose value is the user's own data. A form carries the account details
# it is asking for — names, addresses, account numbers, PINs — and the server
# builds its placeholders from them, so both sides are replaced wholesale rather
# than pattern-matched. A form's submitted `content` goes too — it is a mapping
# of answers, unlike the `content` string on a text or state event.
# The key stays, the shape stays, the value does not.
# `options` belongs here for a reason that is easy to miss: the server composes
# choice labels in the user's voice, so an option reads "I (Charles) prefer ..."
# and carries their name even though nothing about the key says user data.
USER_DATA_KEYS = frozenset({"prefilled", "placeholder", "options"})
USER_DATA_PLACEHOLDER = "[redacted]"

# The account and session a recording ran under are not part of the shape being
# recorded, and a fixture carrying a real session id cannot be replayed into a
# flow test — the client would filter it out as belonging elsewhere.
PLACEHOLDER_USER_ID = "100000000000000001"

# Identifier fields are never redacted — they are opaque numbers, and rewriting
# one destroys the shape the fixture exists to record.
OPAQUE_KEYS = frozenset({
    "id", "event_id", "message_id", "session_id", "request_id", "operation_id",
    "quoted_message_id", "thinking_id", "turn_id", "revision", "next_message_id",
    "max_message_revision", "since_revision", "device_id", "user_id",
})


def redact(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, replacement in REDACTIONS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in OPAQUE_KEYS:
                out[k] = v
            elif k in USER_DATA_KEYS and isinstance(v, str) and v:
                out[k] = USER_DATA_PLACEHOLDER
            elif k in USER_DATA_KEYS and isinstance(v, list):
                out[k] = [USER_DATA_PLACEHOLDER for _ in v]
            elif k == "content" and isinstance(v, dict) and v:
                out[k] = {key: USER_DATA_PLACEHOLDER for key in v}
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def anonymize(envelope: dict[str, Any]) -> dict[str, Any]:
    """Replace the identities the recording ran under. Redaction cannot reach
    them: an account id is an opaque number, exempt from pattern matching so it
    does not get rewritten into a fake phone number."""
    source = envelope.get("metadata", {}).get("source")
    if isinstance(source, dict) and source.get("user_id"):
        source["user_id"] = PLACEHOLDER_USER_ID
    payload = envelope.get("payload")
    if isinstance(payload, dict) and payload.get("session_id"):
        payload["session_id"] = PLACEHOLDER_SESSION_ID
    return envelope


async def record(
    prompt: str, raw_dir: pathlib.Path | None, follow_seconds: float = 0.0,
) -> dict[str, dict[str, Any]]:
    token = os.environ.get("PINE_ACCESS_TOKEN", "")
    user_id = os.environ.get("PINE_USER_ID", "")
    if not token or not user_id:
        raise SystemExit("PINE_ACCESS_TOKEN and PINE_USER_ID are required.")

    client = AsyncPineAI(
        access_token=token, user_id=user_id,
        base_url=os.environ.get("PINE_BASE_URL", "https://www.19pine.ai"),
    )
    seen: dict[str, dict[str, Any]] = {}
    everything: list[dict[str, Any]] = []

    await client.connect()
    session = await client.sessions.create()
    sid = session["id"]

    def capture(event_type: str, raw: dict[str, Any]) -> None:
        everything.append(raw)
        seen[event_type] = raw  # a live-updating card is completed by its last envelope

    remove = client._sio.add_event_handler(capture)  # type: ignore[union-attr]
    try:
        await client.join_session(sid)
        await client.rebuild(sid)
        async for event in client.chat(sid, prompt):
            print(f"  {event.type}{'' if is_supported_event(event.type) else '  (unsupported)'}")
        if follow_seconds:
            # A task runs after the turn ends. Its call reports through
            # session:tool_status, which no turn-scoped listener would see.
            print(f"  ... following for {follow_seconds:.0f}s")
            before = len(everything)
            await asyncio.sleep(follow_seconds)
            for env in everything[before:]:
                print(f"  {env['type']}")
    finally:
        remove()
        client.leave_session(sid)
        with contextlib.suppress(Exception):
            await client.sessions.delete(sid)
        await client.disconnect()

    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.joinpath("session.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in everything)
        )
        print(f"\n{len(everything)} envelopes -> {raw_dir / 'session.jsonl'}")

    return seen


def replay(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Rebuild fixtures from a captured log.

    A recording costs a live session and its credits. Re-deriving from what was
    already captured costs nothing, which matters when the fault is in how a
    fixture was written rather than in what the server sent.
    """
    seen: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            envelope = json.loads(line)
            seen[envelope["type"]] = envelope
    print(f"replayed {path}: {len(seen)} distinct event types")
    return seen


def write_fixtures(seen: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recorded, skipped = [], []

    for event_type, envelope in sorted(seen.items()):
        if event_type == S2CEvent.READY.value or not is_supported_event(event_type):
            skipped.append(event_type)
            continue
        name = event_type.replace("session:", "")
        FIXTURES.joinpath(f"{name}.json").write_text(
            json.dumps(anonymize(redact(envelope)), indent=2) + "\n"
        )
        provenance[event_type] = {
            "source": "recorded", "derived_from": None, "recorded_at": now,
        }
        recorded.append(event_type)

    FIXTURES.joinpath("provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return recorded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--follow", type=float, default=0.0, metavar="SECONDS",
                        help="keep capturing after the turn ends, for events a "
                             "running task emits later (session:tool_status)")
    parser.add_argument("--from-raw", type=pathlib.Path, default=None, metavar="JSONL",
                        help="rebuild fixtures from a previously captured log "
                             "instead of running a session")
    parser.add_argument("--raw-dir", type=pathlib.Path, default=None,
                        help="also dump every envelope seen, for inspection")
    args = parser.parse_args()

    seen = (replay(args.from_raw) if args.from_raw
            else asyncio.run(record(args.prompt, args.raw_dir, args.follow)))
    recorded, skipped = write_fixtures(seen)

    print(f"\nrecorded {len(recorded)}: {', '.join(recorded) or '(none)'}")
    print(f"skipped {len(skipped)}: {', '.join(skipped) or '(none)'}")
    still_derived = [
        event for event, entry in json.loads((FIXTURES / "provenance.json").read_text()).items()
        if entry["source"] == "derived"
    ]
    if still_derived:
        print(f"\nstill derived from the protocol source, never observed: "
              f"{', '.join(sorted(still_derived))}")
        print("These need a session that reaches the condition they describe.")


if __name__ == "__main__":
    main()
