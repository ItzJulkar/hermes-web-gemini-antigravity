"""``hermes webgemini ...`` — operator commands for the Gemini/Antigravity web provider."""

from __future__ import annotations

import argparse

from . import provider as p


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes webgemini`` argparse tree."""
    subs = subparser.add_subparsers(dest="webgemini_command")
    subs.add_parser("status", help="show mode, model, credentials and whether search/extract are active")
    mode = subs.add_parser("mode", help="choose the transport: auto | api | cli")
    mode.add_argument("value", choices=p.MODES)
    subs.add_parser("on", help="point web_search + web_extract at this provider")
    subs.add_parser("off", help="restore the backend that was active before `on`")
    test = subs.add_parser("test", help="run one real grounded search and print timings")
    test.add_argument("query", nargs="*", default=None, help="query to test with")
    subparser.set_defaults(func=webgemini_command)


def webgemini_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "webgemini_command", None)
    if not cmd:
        print(p.status_text())
        print("\nusage: hermes webgemini {status|mode <auto|api|cli>|on|off|test [query]}")
        return 0

    if cmd == "status":
        print(p.status_text())
        return 0
    if cmd == "mode":
        print(p.set_transport(args.value))
        return 0
    if cmd == "on":
        print(p.enable_web_search())
        return 0
    if cmd == "off":
        print(p.disable_web_search())
        return 0
    if cmd == "test":
        query = " ".join(args.query or []) or "latest AI news"
        print(p.format_live_test(p.live_test(query)))
        return 0
    return 2