"""CLI: pine auth login|status|logout"""

from typing import Optional

import click
from rich.console import Console

from pine_assistant.client import AsyncPineAI

console = Console()


def _load_config() -> dict:
    from pine_assistant.cli.main import _load_config
    return _load_config()


def _save_config(cfg: dict) -> None:
    from pine_assistant.cli.main import _save_config
    _save_config(cfg)


def _run(coro):
    from pine_assistant.cli.main import _run
    return _run(coro)


@click.group()
def auth():
    """Authentication commands."""


@auth.command("login")
@click.option("--base-url", default=None, help="Pine AI base URL")
def auth_login(base_url: Optional[str]):
    """Log in with email verification."""

    async def _login():
        cfg = _load_config()
        url = base_url or cfg.get("base_url", "https://www.19pine.ai")
        client = AsyncPineAI(base_url=url)

        email = click.prompt("Email")
        with console.status("Sending verification code..."):
            result = await client.auth.request_code(email)
        console.print("[green]Code sent! Check your email.[/green]")

        code = click.prompt("Verification code")
        with console.status("Verifying..."):
            verify = await client.auth.verify_code(email, code, result["request_token"])
        console.print(f"[green]Logged in as {verify['email']} (ID: {verify['id']})[/green]")

        _save_config({**cfg, "access_token": verify["access_token"], "user_id": verify["id"],
                       "email": verify["email"], "base_url": url})
        console.print("[dim]Token saved to ~/.pine/config.json[/dim]")

    _run(_login())


@auth.command("status")
def auth_status():
    """Show current auth status."""
    cfg = _load_config()
    if cfg.get("access_token"):
        console.print(f"[green]Logged in[/green] as {cfg.get('email', 'unknown')} (ID: {cfg.get('user_id')})")
    else:
        console.print("[yellow]Not logged in. Run `pine auth login`.[/yellow]")


@auth.command("logout")
def auth_logout():
    """Clear saved credentials."""
    _save_config({})
    console.print("[green]Logged out.[/green]")
