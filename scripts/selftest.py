#!/usr/bin/env python
"""One-shot live check for the gemini-antigravity provider.

Prints the resolved settings, then runs two real searches and one extract:

    python scripts/selftest.py
    python scripts/selftest.py "your query"

Exit code 0 only when every step succeeded — safe to use in a demo recording.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("HERMES_REPO", PLUGIN_DIR.parents[1] / "hermes-agent"))
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("ga_provider", PLUGIN_DIR / "provider.py")
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)

QUERIES = [arg for arg in sys.argv[1:]] or [
    "current ethereum staking yield",
    "latest US CPI inflation number",
]


def main() -> int:
    print(provider.status_text(), "\n")
    failures = 0

    for query in QUERIES:
        result = provider.live_test(query)
        print(provider.format_live_test(result), "\n")
        failures += 0 if result["ok"] else 1

    extract = provider.GeminiAntigravityWebSearchProvider().extract(["https://example.com"])
    page = extract[0]
    ok = bool(page.get("content")) and not page.get("error")
    print(f"{'OK' if ok else 'FAILED'} extract: {page.get('title')!r} "
          f"{len(page.get('content') or '')} chars {page.get('error') or ''}")
    failures += 0 if ok else 1

    print(f"\n{'all checks passed' if not failures else str(failures) + ' check(s) failed'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())