"""CLI: pine chat, pine send"""

import json
from typing import Optional

import click
from rich.console import Console

from pine_assistant.models.events import S2CEvent

console = Console()


def _get_client():
    from pine_assistant.cli.main import _get_client
    return _get_client()


def _run(coro):
    from pine_assistant.cli.main import _run
    return _run(coro)


@click.command("chat")
@click.argument("session_id", required=False)
def chat_cmd(session_id: Optional[str]):
    """Interactive chat with Pine AI."""

    async def _chat():
        client = _get_client()
        await client.connect()
        sid = session_id
        if not sid:
            with console.status("Creating session..."):
                s = await client.sessions.create()
                sid = s["id"]
            console.print(f"[dim]Session: {sid}[/dim]")
        await client.join_session(sid)
        console.print("[cyan]Type your message (Ctrl+C to exit)[/cyan]\n")
        try:
            while True:
                msg = click.prompt("You", prompt_suffix=": ")
                if msg.lower() in ("/quit", "/exit"):
                    break
                async for event in client.chat(sid, msg):
                    if event.type == S2CEvent.SESSION_TEXT:
                        data = event.data if isinstance(event.data, dict) else {}
                        console.print(f"[green]Pine AI:[/green] {data.get('content', '')}")
                    elif event.type == S2CEvent.SESSION_FORM_TO_USER:
                        console.print(f"[yellow]Form needed:[/yellow] {event.data}")
                    elif event.type == S2CEvent.SESSION_STATE:
                        data = event.data if isinstance(event.data, dict) else {}
                        console.print(f"[dim][state: {data.get('content', '')}][/dim]")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            client.leave_session(sid)
            await client.disconnect()

    _run(_chat())


@click.command("send")
@click.argument("message")
@click.option("-s", "--session", "session_id", default=None)
@click.option("--json-output", "--json", is_flag=True)
def send_cmd(message: str, session_id: Optional[str], json_output: bool):
    """Send a one-shot message."""

    async def _send():
        client = _get_client()
        await client.connect()
        sid = session_id
        if not sid:
            s = await client.sessions.create()
            sid = s["id"]
            if not json_output:
                console.print(f"[dim]Session: {sid}[/dim]")
        await client.join_session(sid)
        async for event in client.chat(sid, message):
            if json_output:
                click.echo(json.dumps({"type": event.type, "data": event.data}))
            elif event.type == S2CEvent.SESSION_TEXT:
                data = event.data if isinstance(event.data, dict) else {}
                console.print(f"[green]Pine AI:[/green] {data.get('content', '')}")
        client.leave_session(sid)
        await client.disconnect()

    _run(_send())
