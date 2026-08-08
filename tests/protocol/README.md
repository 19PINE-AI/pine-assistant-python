# Protocol fixtures and tests

These pin the SDK to the supported protocol scope — the subset of the
task-session Socket.IO protocol that carries a compatibility guarantee.

## Layout

| Path | What it is |
|---|---|
| `fixtures/*.json` | One envelope per supported event. Not tests — the material the tests run on |
| `fixtures/provenance.json` | Where each fixture came from, and when |
| `test_contract.py` | One envelope at a time: does the SDK read it |
| `test_flow.py` | A sequence of envelopes: does the SDK behave as the scope requires |
| `fake.py` | A scripted stand-in for `socketio.AsyncClient` |

The two test files fail for different reasons, which is why they are separate:
a contract failure means a payload shape moved, a flow failure means the SDK's
logic is wrong.

Events reach the flow tests through the real transport — envelope parsing,
`SocketIOManager`, `ChatEngine`. Only the wire is fake.

## Provenance

`recorded` fixtures came off a running server. `derived` ones were written from
the protocol source because no ordinary session reaches the condition they
describe — a blocked account, a task waiting on credits. A derived fixture is
weaker evidence: it says the SDK matches what the protocol declares, not what
the server sends.

Re-record after a backend release:

```bash
PINE_ACCESS_TOKEN=... PINE_USER_ID=... python -m tests.integration.record_fixtures --raw-dir /tmp/pine-raw
git diff tests/protocol/fixtures
```

A non-empty diff is server drift, and re-recording is the only thing that
surfaces it — the scope is a commitment about future behaviour, which no check
in this repository can verify.

## Fixtures are the supported surface only

Recording never writes a fixture for an event outside the scope, and
`test_contract.py` fails if one appears. Unsupported events still reach callers
untouched — `test_flow.py` pins that — they are simply not something the SDK
promises anything about.
