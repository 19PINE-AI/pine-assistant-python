"""Live integration test — Python SDK against real Pine AI API."""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pine_assistant import AsyncPineAI, S2CEvent

ACCESS_TOKEN = os.environ.get("PINE_ACCESS_TOKEN", "")
USER_ID = os.environ.get("PINE_USER_ID", "")

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1


async def main():
    client = AsyncPineAI(access_token=ACCESS_TOKEN, user_id=USER_ID)

    # T2: Connection
    print("\n=== T2: Socket.IO Connection ===")
    await client.connect()
    check(client.connected, "Connected to Pine AI")

    # T3: Session CRUD
    print("\n=== T3: Session CRUD ===")
    session = await client.sessions.create()
    session_id = session["id"]
    check(session_id, f"Session created: {session_id}")
    check(session["state"] == "init", f"State is init: {session['state']}")

    result = await client.sessions.list(limit=10)
    check(result["total"] > 0, f"List sessions: {result['total']} total")

    got = await client.sessions.get(session_id)
    check(got["id"] == session_id, f"Get session: {got['id']}")

    # T4: Join + Chat
    print("\n=== T4: Join + Chat ===")
    join_data = await client.join_session(session_id)
    check(join_data is not None, f"Joined session")

    print("\n=== T5: Chat with Stream Buffering ===")
    got_text = False
    got_form = False
    got_text_part = False
    events = []

    async for event in client.chat(session_id, "Help me negotiate my Comcast bill. Account number 12345."):
        events.append(event)
        detail = ""
        if event.type == S2CEvent.SESSION_TEXT:
            got_text = True
            content = event.data.get("content", "") if isinstance(event.data, dict) else ""
            detail = f"({len(content)} chars)"
        elif event.type == S2CEvent.SESSION_FORM_TO_USER:
            got_form = True
            d = event.data if isinstance(event.data, dict) else {}
            fields = d.get("form", {}).get("fields", [])
            detail = f"({len(fields)} fields)"
            msg = d.get("message_to_user", "")
            print(f"\n  Pine AI form: \"{msg[:200]}\"")
            for f in fields:
                print(f"    - {f.get('name')} ({f.get('type')})")
            print()
        elif event.type == S2CEvent.SESSION_TEXT_PART:
            got_text_part = True
        print(f"  Event: {event.type} {detail}")

    check(got_text or got_form, "Received substantive response (text or form)")
    check(not got_text_part, "No raw text_part leaked")
    print(f"  Event types: {', '.join(set(e.type for e in events))}")

    # T9: History
    print("\n=== T9: History ===")
    history = await client.get_history(session_id, max_messages=10)
    check("messages" in history, f"History has messages ({len(history.get('messages', []))})")

    # T7: State
    print("\n=== T7: State ===")
    updated = await client.sessions.get(session_id)
    check(updated["state"] in ("chat", "init"), f"State: {updated['state']}")

    # T10: Error
    print("\n=== T10: Error ===")
    try:
        await client.sessions.get("999999999999999")
        check(False, "Should have thrown")
    except Exception as e:
        check(True, f"Non-existent session throws: {str(e)[:80]}")

    # Cleanup
    print("\n=== Cleanup ===")
    client.leave_session(session_id)
    await client.sessions.delete(session_id)
    print(f"  Session {session_id} deleted")
    await client.disconnect()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 50)
    sys.exit(1 if failed > 0 else 0)


asyncio.run(main())
