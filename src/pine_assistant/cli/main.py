"""
Pine AI CLI — `pine` command.

Commands:
  pine auth login          Email verification flow
  pine chat [session-id]   Interactive REPL chat
  pine send <message>      One-shot message
  pine sessions <cmd>      Session CRUD
  pine task <cmd>          Task lifecycle
"""

import asyncio
import json
from pathlib import Path

try:
    import click
    from rich.console import Console
except ImportError:
    raise SystemExit("CLI requires extras: pip install pine-assistant[cli]")

from pine_assistant.client import AsyncPineAI

console = Console()
CONFIG_FILE = Path.home() / ".pine" / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _get_client() -> AsyncPineAI:
    cfg = _load_config()
    if not cfg.get("access_token") or not cfg.get("user_id"):
        console.print("[red]Not logged in. Run `pine auth login` first.[/red]")
        raise SystemExit(1)
    return AsyncPineAI(
        access_token=cfg["access_token"],
        user_id=cfg["user_id"],
        base_url=cfg.get("base_url", "https://www.19pine.ai"),
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.version_option("0.1.0")
def main():
    """Pine AI CLI — Let Pine AI handle your digital chores."""


# Register subcommands from separate modules
from pine_assistant.cli.auth import auth
from pine_assistant.cli.chat import chat_cmd, send_cmd
from pine_assistant.cli.sessions import sessions
from pine_assistant.cli.tasks import task

main.add_command(auth)
main.add_command(chat_cmd)
main.add_command(send_cmd)
main.add_command(sessions)
main.add_command(task)


if __name__ == "__main__":
    main()
