"""Offline tests for the gemini-antigravity web provider.

No network, no config writes, no Antigravity CLI. Run with:

    "…/hermes-agent/venv/Scripts/python.exe" -m pytest tests -q

The plugin directory name contains a hyphen, so the provider is imported by file
path rather than as a package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("HERMES_REPO", Path(__file__).resolve().parents[3] / "hermes-agent"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ga_{name}", PLUGIN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_package():
    """Import the plugin the way Hermes does — as a package, so `from . import cli` works."""
    name = "ga_plugin_pkg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def p():
    return _load("provider")


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("text", [
    '```json\n{"results":[{"title":"a","url":"https://a.com","description":"d"}]}\n```',
    '{"results":[{"title":"a","url":"https://a.com","description":"d"}]}',
    'Here you go: {"results":[{"title":"a","url":"https://a.com","description":"d"}]} — enjoy',
    '"{\\"results\\":[{\\"title\\":\\"a\\",\\"url\\":\\"https://a.com\\",\\"description\\":\\"d\\"}]}"',
])
def test_loads_accepts_every_reply_shape(p, text):
    rows = p._to_rows(p._loads(text), 5)
    assert len(rows) == 1
    assert rows[0]["url"] == "https://a.com"


def test_loads_salvages_a_truncated_array(p):
    broken = '[{"title":"c","url":"https://c.com","description":"d"},{"title":"e","url":"https://e.com","de'
    rows = p._to_rows(p._loads(broken), 5)
    assert [r["url"] for r in rows] == ["https://c.com"]


def test_loads_returns_none_on_garbage(p):
    assert p._loads("no json at all") is None


# --------------------------------------------------------------------------- rows
def test_rows_dedupe_reject_non_http_and_number_positions(p):
    payload = {"results": [
        {"title": "one", "url": "https://a.com/x", "description": "d1"},
        {"title": "dupe", "url": "https://a.com/x/", "description": "d2"},
        {"title": "bad", "url": "javascript:alert(1)", "description": "d3"},
        {"title": "two", "url": "http://b.com", "description": ""},
    ]}
    rows = p._to_rows(payload, 10)
    assert [r["url"] for r in rows] == ["https://a.com/x", "http://b.com"]
    assert [r["position"] for r in rows] == [1, 2]
    assert rows[1]["title"] == "two"          # missing description is tolerated


def test_rows_respect_limit(p):
    payload = {"results": [{"title": str(i), "url": f"https://e{i}.com", "description": "d"} for i in range(9)]}
    assert len(p._to_rows(payload, 3)) == 3


def test_rows_accept_flat_list_and_web_key(p):
    assert p._to_rows([{"title": "t", "url": "https://a.com", "description": "d"}], 5)
    assert p._to_rows({"web": [{"title": "t", "url": "https://a.com", "description": "d"}]}, 5)


def test_unresolvable_grounding_rows_are_dropped(p, monkeypatch):
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc=="
    monkeypatch.setattr(p, "_resolve_one", lambda url: url)   # token no longer resolves
    rows = [{"title": "real", "url": "https://a.com", "description": "d", "position": 1},
            {"title": "dead", "url": redirect, "description": "d", "position": 2}]
    assert [r["url"] for r in p._resolve_rows(rows)] == ["https://a.com"]


def test_all_redirect_rows_are_kept_when_nothing_else_is_left(p, monkeypatch):
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc=="
    monkeypatch.setattr(p, "_resolve_one", lambda url: url)
    rows = [{"title": "dead", "url": redirect, "description": "d", "position": 1}]
    assert len(p._resolve_rows(rows)) == 1


# --------------------------------------------------------------------------- prompt
def test_prompt_forbids_tool_use(p):
    text = p._prompt("anything", 4)
    assert "Do not run any shell commands" in text
    assert "4 results" in text


# --------------------------------------------------------------------------- transport resolution
def test_transport_auto_prefers_api_when_key_present(p, monkeypatch):
    monkeypatch.setattr(p, "_cfg", lambda: {"transport": "auto"})
    monkeypatch.setattr(p, "_api_key", lambda: "k")
    assert p._transport() == "api"


def test_transport_auto_falls_back_to_cli(p, monkeypatch):
    monkeypatch.setattr(p, "_cfg", lambda: {"transport": "auto"})
    monkeypatch.setattr(p, "_api_key", lambda: "")
    assert p._transport() == "cli"


def test_transport_explicit_wins(p, monkeypatch):
    monkeypatch.setattr(p, "_cfg", lambda: {"transport": "cli"})
    monkeypatch.setattr(p, "_api_key", lambda: "k")
    assert p._transport() == "cli"


def test_timeout_is_clamped(p, monkeypatch):
    monkeypatch.setattr(p, "_cfg", lambda: {"timeout": "9999"})
    assert p._timeout() == 600
    monkeypatch.setattr(p, "_cfg", lambda: {"timeout": "nonsense"})
    assert p._timeout() == p.DEFAULT_TIMEOUT
    monkeypatch.setattr(p, "_cfg", lambda: {"timeout": 1})
    assert p._timeout() == 10


def test_set_transport_rejects_unknown_mode(p):
    with pytest.raises(ValueError):
        p.set_transport("carrier-pigeon")


# --------------------------------------------------------------------------- availability
def test_available_with_only_a_cli(p, monkeypatch):
    monkeypatch.setattr(p, "_api_key", lambda: "")
    monkeypatch.setattr(p, "_cli_path", lambda: "/usr/local/bin/agy")
    assert p.GeminiAntigravityWebSearchProvider().is_available() is True


def test_unavailable_with_neither(p, monkeypatch):
    monkeypatch.setattr(p, "_api_key", lambda: "")
    monkeypatch.setattr(p, "_cli_path", lambda: "")
    assert p.GeminiAntigravityWebSearchProvider().is_available() is False


def test_search_explains_missing_credentials(p, monkeypatch):
    monkeypatch.setattr(p, "_api_key", lambda: "")
    monkeypatch.setattr(p, "_cli_path", lambda: "")
    monkeypatch.setattr(p, "_transport", lambda: "cli")
    out = p.GeminiAntigravityWebSearchProvider().search("x")
    assert out["success"] is False
    assert "Antigravity CLI not found" in out["error"]


def test_search_api_without_key_is_an_actionable_error(p, monkeypatch):
    monkeypatch.setattr(p, "_transport", lambda: "api")
    monkeypatch.setattr(p, "_api_key", lambda: "")
    out = p.GeminiAntigravityWebSearchProvider().search("x")
    assert out["success"] is False
    assert "webgemini mode cli" in out["error"]


# --------------------------------------------------------------------------- extract
def test_html_to_text_drops_scripts_and_keeps_title(p):
    html = "<html><head><title>Hi</title><style>x{}</style></head><body><script>bad()</script><p>Hello  world</p></body></html>"
    title, text = p.html_to_text(html)
    assert title == "Hi"
    assert "Hello world" in text
    assert "bad()" not in text


def test_extract_shape(p, monkeypatch):
    class FakeResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><head><title>T</title></head><body><p>body text</p></body></html>"
        url = "https://a.com"
    class FakeClient:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return FakeResp()
    monkeypatch.setattr(p.httpx, "Client", FakeClient)

    out = p.GeminiAntigravityWebSearchProvider().extract(["https://a.com"], char_limit=5)
    assert out[0]["title"] == "T"
    assert out[0]["content"] == "body text"      # provider clamps to a 1000-char floor
    assert out[0]["raw_content"] == "body text"
    assert out[0]["metadata"]["sourceURL"] == "https://a.com"


def test_extract_truncates_at_max_chars(p, monkeypatch):
    long_body = "<html><head><title>T</title></head><body><p>" + ("x" * 5000) + "</p></body></html>"

    class FakeResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = long_body
        url = "https://a.com"

    class FakeClient:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return FakeResp()

    monkeypatch.setattr(p.httpx, "Client", FakeClient)
    out = p._one_extract("https://a.com", 100)
    assert len(out["content"]) == 100


# --------------------------------------------------------------------------- operator surface
def test_status_text_reports_the_essentials(p, monkeypatch):
    monkeypatch.setattr(p, "current_settings", lambda: {
        "version": p.VERSION, "configured_transport": "auto", "active_transport": "cli",
        "model": "gemini-3.8-flash-low", "api_model": "gemini-3.8-flash", "timeout": 90,
        "cli_path": "/x/agy", "cli_version": "1.5.3", "api_key": "missing",
        "search_backend": p.NAME, "extract_backend": p.NAME,
        "search_active": True, "extract_active": True,
    })
    text = p.status_text()
    assert "auto → cli" in text
    assert "ACTIVE" in text
    assert "1.5.3" in text


def test_format_live_test_handles_failure_and_success(p):
    assert "FAILED" in p.format_live_test(
        {"query": "q", "ok": False, "seconds": 1.0, "transport": "cli", "rows": [], "error": "boom"})
    ok = p.format_live_test({"query": "q", "ok": True, "seconds": 2.0, "transport": "cli",
                             "rows": [{"position": 1, "title": "t", "url": "https://a.com"}], "error": ""})
    assert "OK (cli, 2.0s)" in ok and "https://a.com" in ok


def test_slash_command_router(p, monkeypatch):
    init = _load_package()
    monkeypatch.setattr(init, "status_text", lambda: "STATUS")
    monkeypatch.setattr(init, "set_transport", lambda m: f"switched:{m}")
    monkeypatch.setattr(init, "format_live_test", lambda r: f"test:{r['query']}")
    monkeypatch.setattr(init, "live_test", lambda q: {"query": q})
    assert init._slash_command("").startswith("STATUS")
    assert init._slash_command("api") == "switched:api"
    assert init._slash_command("test dogecoin") == "test:dogecoin"
    assert "unknown argument" in init._slash_command("bogus")


def test_manifest_declares_what_it_registers():
    try:
        import yaml
    except ImportError:  # Hermes vendors its own loader
        import hermes_yaml as yaml
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["provides_web_providers"] == ["gemini-antigravity"]
    assert manifest["kind"] == "backend"
    assert manifest["requires_hermes"].startswith(">=")
    assert any(k.startswith("web.gemini_antigravity.") for k in manifest["config_keys"])