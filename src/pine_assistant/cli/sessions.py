"""CLI: pine sessions list|create|delete"""

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _load_config() -> dict:
    from pine_assistant.cli.main import _load_config
    return _load_config()


def _get_client():
    from pine_assistant.cli.main import _get_client
    return _get_client()


def _run(coro):
    from pine_assistant.cli.main import _run
    return _run(coro)


@click.group()
def sessions():
    """Session management."""


@sessions.command("list")
@click.option("--state", default=None)
@click.option("--limit", default=10, type=int)
@click.option("--json-output", "--json", is_flag=True)
def sessions_list(state, limit, json_output):
    """List sessions."""

    async def _list():
        client = _get_client()
        result = await client.sessions.list(state=state, limit=limit)
        if json_output:
            click.echo(json.dumps(result, indent=2))
            return
        table = Table(title=f"Sessions ({result['total']} total)")
        table.add_column("ID", style="bold")
        table.add_column("State")
        table.add_column("Title")
        table.add_column("Updated")
        for s in result["sessions"]:
            table.add_row(s["id"], s["state"], s.get("title", ""), s.get("updated_at", ""))
        console.print(table)

    _run(_list())


@sessions.command("create")
def sessions_create():
    """Create a new session."""

    async def _create():
        client = _get_client()
        with console.status("Creating session..."):
            session = await client.sessions.create()
        console.print(f"[green]Session created: {session['id']}[/green]")

    _run(_create())


@sessions.command("delete")
@click.argument("session_id")
@click.option("-f", "--force", is_flag=True)
def sessions_delete(session_id, force):
    """Delete a session."""

    async def _delete():
        client = _get_client()
        with console.status("Deleting..."):
            await client.sessions.delete(session_id, force_delete=force)
        console.print(f"[green]Session {session_id} deleted.[/green]")

    _run(_delete())
