"""CLI: pine task start|stop"""

import click
from rich.console import Console

console = Console()


def _get_client():
    from pine_assistant.cli.main import _get_client
    return _get_client()


def _run(coro):
    from pine_assistant.cli.main import _run
    return _run(coro)


@click.group()
def task():
    """Task lifecycle commands."""


@task.command("start")
@click.argument("session_id")
def task_start(session_id):
    """Start task execution (requires task_ready)."""

    async def _start():
        client = _get_client()
        with console.status("Starting task..."):
            result = await client.sessions.start_task(session_id)
        console.print(f"[green]Task started: {result.get('message', 'OK')}[/green]")

    _run(_start())


@task.command("stop")
@click.argument("session_id")
def task_stop(session_id):
    """Stop a running task."""

    async def _stop():
        client = _get_client()
        with console.status("Stopping task..."):
            result = await client.sessions.stop_task(session_id)
        console.print(f"[green]Task stopped: {result.get('message', 'OK')}[/green]")

    _run(_stop())
