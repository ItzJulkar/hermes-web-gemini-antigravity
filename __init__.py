"""Gemini 3.8 / Antigravity grounded web provider.

Registers three things:

* the ``gemini-antigravity`` web search + extract provider (what ``web_search`` uses),
* ``/webgemini`` — in-session status / mode switch / live test,
* ``hermes webgemini`` — the same from a shell, plus on/off for the web backends.
"""
from __future__ import annotations

from . import cli
from .provider import (
    DISPLAY,
    GeminiAntigravityWebSearchProvider,
    format_live_test,
    live_test,
    set_transport,
    status_text,
)

_USAGE = "usage: /webgemini [status | auto | api | cli | test <query>]"


def _slash_command(raw_args: str) -> str:
    arg = (raw_args or "").strip()
    word, _, rest = arg.partition(" ")
    word = word.lower()

    if word in ("", "status"):
        return status_text() + "\n\n" + _USAGE
    if word in ("auto", "api", "cli"):
        return set_transport(word)
    if word == "test":
        return format_live_test(live_test(rest.strip() or "latest AI news"))
    return f"unknown argument {word!r}\n{_USAGE}"


def register(ctx) -> None:
    ctx.register_web_search_provider(GeminiAntigravityWebSearchProvider())
    ctx.register_command(
        "webgemini",
        _slash_command,
        description="Gemini/Antigravity web provider — status, transport (auto|api|cli), live test",
        args_hint="[status|auto|api|cli|test <query>]",
    )
    ctx.register_cli_command(
        name="webgemini",
        help="Gemini/Antigravity web search provider (status, mode, on, off, test)",
        setup_fn=cli.register_cli,
        handler_fn=cli.webgemini_command,
        description=(
            f"Operate the {DISPLAY} web provider: show status, switch between the Antigravity CLI "
            "and the Gemini API, point web_search at it or restore the previous backend, and run a "
            "live grounded search."
        ),
    )