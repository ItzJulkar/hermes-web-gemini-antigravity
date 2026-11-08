"""Gemini 3.8 grounded web search + extract for Hermes Agent.

One provider, two transports, no coupling to the chat model:

``api``
    ``generativelanguage.googleapis.com/v1beta/models/<model>:generateContent``
    with ``tools=[{"google_search": {}}]``. Needs ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY``. 2–6 s.

``cli``
    The Google Antigravity CLI (``agy``) in print mode, running
    ``gemini-3.8-flash`` with its own Google Search grounding. Uses an existing
    Google Pro subscription — no API key. 20–60 s.

``auto`` (default) — ``api`` when a key exists, else ``cli``.

Result shapes are the legacy provider contract (the tool wrapper does not
translate)::

    search:  {"success": True, "data": {"web": [{"title", "url", "description", "position"}]}}
    extract: [{"url", "title", "content", "raw_content", "metadata"}]
    failure: {"success": False, "error": str}

Everything below the provider class is the operator surface used by
``/webgemini`` and ``hermes webgemini``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any

import httpx

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

NAME = "gemini-antigravity"
DISPLAY = "Gemini 3.8 (Antigravity / Google Search)"
VERSION = "1.1.0"

DEFAULT_CLI_MODEL = "gemini-3.8-flash-low"
DEFAULT_API_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT = 90
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
USER_AGENT = "Mozilla/5.0 (compatible; HermesAgent/1.0; +https://hermes-agent.nousresearch.com)"
LIMIT_CAP = 20
MODES = ("auto", "api", "cli")

# JSON schema handed to both transports so a free-text reply never has to be
# regex-scraped. `results` is the only required field.
RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "url", "description"],
            },
        }
    },
    "required": ["results"],
}

_API_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "url": {"type": "STRING"},
            "description": {"type": "STRING"},
        },
        "required": ["title", "url", "description"],
    },
}


# --------------------------------------------------------------------------- config
def _cfg() -> dict[str, Any]:
    """``web.gemini_antigravity`` from config.yaml (empty dict on any miss)."""
    try:
        from hermes_cli.config import load_config

        cur: Any = load_config()
        for key in ("web", "gemini_antigravity"):
            cur = cur.get(key) if isinstance(cur, dict) else None
        return cur if isinstance(cur, dict) else {}
    except Exception as exc:  # noqa: BLE001 — config must never break a search
        logger.debug("gemini-antigravity: config read failed: %s", exc)
        return {}


def _write_config(mutate) -> None:
    """Mutate config.yaml through Hermes' own loader/saver (atomic, profile-aware)."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    mutate(cfg)
    save_config(cfg, merge_existing=True)


def _api_key() -> str:
    return get_provider_env("GEMINI_API_KEY") or get_provider_env("GOOGLE_API_KEY")


def _cli_path() -> str:
    """Resolve the Antigravity CLI: config → env → PATH → known install spots."""
    candidates = [
        str(_cfg().get("cli_path") or "").strip(),
        os.environ.get("GEMINI_ANTIGRAVITY_CLI", "").strip(),
        shutil.which("agy") or "",
        shutil.which("agy.exe") or "",
        os.path.expanduser("~/.local/bin/agy.exe"),
        os.path.expanduser("~/.local/bin/agy"),
        os.path.expanduser("~/AppData/Local/Programs/antigravity/agy.exe"),
        "/usr/local/bin/agy",
        "/opt/homebrew/bin/agy",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def _transport() -> str:
    want = str(_cfg().get("transport") or "auto").strip().lower()
    if want in ("api", "cli"):
        return want
    return "api" if _api_key() else "cli"


def _timeout() -> int:
    try:
        return max(10, min(int(_cfg().get("timeout") or DEFAULT_TIMEOUT), 600))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _cli_model() -> str:
    return str(_cfg().get("model") or DEFAULT_CLI_MODEL).strip()


def _api_model() -> str:
    return str(_cfg().get("api_model") or DEFAULT_API_MODEL).strip()


# --------------------------------------------------------------------------- rows
def _prompt(query: str, limit: int) -> str:
    """Prompt shape matters: asking it to "search carefully" makes Gemini reach
    for RunCommand, which headless mode auto-denies and then returns an empty turn."""
    return (
        f'Use Google Search to answer, then return JSON only: {limit} results for '
        f'"{query}", each with title, url and a one-sentence description that quotes '
        "the useful numbers. Give the publisher's own URL (the address a browser shows) — "
        "never a vertexaisearch or google redirect link. "
        "Do not run any shell commands or tools other than search. "
        "Only real pages you saw in the search results, most relevant first."
    )


_URL_RE = re.compile(r"^https?://", re.I)


def _is_grounding_url(url: str) -> bool:
    """Gemini/Antigravity hand back vertexaisearch redirect links, not the real page."""
    low = url.lower()
    return "vertexaisearch.cloud.google.com" in low or "grounding-api-redirect" in low


def _resolve_one(url: str) -> str:
    try:
        with httpx.Client(timeout=12, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            final = str(client.head(url).url)
        return final if _URL_RE.match(final) and not _is_grounding_url(final) else url
    except Exception:  # noqa: BLE001 — an unresolved redirect keeps the working link
        return url


def _resolve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace grounding-redirect URLs with their real destinations (parallel, bounded)."""
    targets = [r for r in rows if _is_grounding_url(str(r.get("url") or ""))]
    if not targets:
        return rows
    urls = [str(r["url"]) for r in targets]
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
            resolved = list(pool.map(_resolve_one, urls))
    except Exception as exc:  # noqa: BLE001
        logger.debug("gemini-antigravity: redirect resolve failed: %s", exc)
        return rows
    for row, new_url in zip(targets, resolved):
        if new_url != row["url"]:
            row["url"] = new_url
    # A grounding token that no longer resolves (they expire) is a 404 for the
    # agent; drop those rows unless nothing else is left.
    kept = [r for r in rows if not _is_grounding_url(str(r.get("url") or ""))]
    return kept or rows


def _to_rows(payload: Any, limit: int) -> list[dict[str, Any]]:
    """Normalise whatever the transport produced into Hermes web rows."""
    if isinstance(payload, dict):
        raw = payload.get("results")
        if raw is None:
            raw = payload.get("web") or payload.get("data")
        if raw is None:
            # A single result object (salvaged from a broken array, or a model
            # that returned one hit without the wrapper) is still a valid row.
            raw = [payload] if (payload.get("url") or payload.get("link")) else []
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("web") or []
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "").strip()
        if not _URL_RE.match(url):
            continue
        key = url.split("#", 1)[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        title = str(item.get("title") or item.get("name") or url).strip()
        desc = str(item.get("description") or item.get("snippet") or item.get("content") or "").strip()
        rows.append({"title": title, "url": url, "description": desc, "position": len(rows) + 1})
        if len(rows) >= limit:
            break
    return rows


def _loads(text: str) -> Any:
    """Parse JSON that may arrive fenced, wrapped in prose, doubled-encoded, or
    truncated mid-way through an array (models do all four)."""
    text = (text or "").strip()
    if not text:
        return None
    for marker in ("```json", "```JSON", "```"):
        if text.startswith(marker):
            text = text[len(marker):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    candidates = [text]
    if "{" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    if "[" in text:
        candidates.append(text[text.find("["): text.rfind("]") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, str):  # JSON-inside-JSON
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data
        return data

    # Last resort: salvage individual objects out of a broken array.
    salvaged: list[Any] = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            salvaged.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return salvaged or None


# --------------------------------------------------------------------------- api transport
def _search_api(query: str, limit: int, key: str) -> dict[str, Any]:
    body = {
        "contents": [{"role": "user", "parts": [{"text": _prompt(query, limit)}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": _API_SCHEMA,
        },
    }
    url = f"{API_BASE}/{_api_model()}:generateContent"
    try:
        with httpx.Client(timeout=_timeout()) as client:
            resp = client.post(url, json=body, headers={
                "x-goog-api-key": key, "Content-Type": "application/json"})
    except httpx.HTTPError as exc:
        return {"success": False, "error": f"Gemini API request failed: {exc}"}

    if resp.status_code != 200:
        detail = resp.text[:300].replace("\n", " ")
        return {"success": False, "error": f"Gemini API {resp.status_code}: {detail}"}
    try:
        data = resp.json()
    except ValueError:
        return {"success": False, "error": "Gemini API returned non-JSON"}

    candidates = data.get("candidates") or []
    if not candidates:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        return {"success": False, "error": f"Gemini returned no candidates{f' ({blocked})' if blocked else ''}"}

    cand = candidates[0]
    text = "".join(str(p.get("text") or "") for p in ((cand.get("content") or {}).get("parts") or [])
                   if isinstance(p, dict))
    rows = _to_rows(_loads(text), limit)

    # Grounding metadata is the authoritative source of real URLs; use it to
    # backfill anything the JSON reply left out and to drop invented links.
    grounded = []
    for chunk in ((cand.get("groundingMetadata") or {}).get("groundingChunks") or []):
        web = (chunk or {}).get("web") or {}
        uri, title = str(web.get("uri") or "").strip(), str(web.get("title") or "").strip()
        if _URL_RE.match(uri):
            grounded.append({"title": title or uri, "url": uri, "description": ""})

    if grounded:
        known = {r["url"].split("#", 1)[0].rstrip("/").lower() for r in rows}
        for g in grounded:
            if len(rows) >= limit:
                break
            if g["url"].split("#", 1)[0].rstrip("/").lower() not in known:
                rows.append({"title": g["title"], "url": g["url"], "description": "", "position": len(rows) + 1})
        real = {g["url"].split("#", 1)[0].rstrip("/").lower() for g in grounded}
        rows = [r for r in rows if r["url"].split("#", 1)[0].rstrip("/").lower() in real] or rows

    if not rows:
        return {"success": False,
                "error": f"Gemini answered without usable results: {text[:300].strip() or 'empty response'}"}
    return {"success": True, "data": {"web": _resolve_rows(rows)[:limit]}}


# --------------------------------------------------------------------------- cli transport
def _search_cli(query: str, limit: int, exe: str) -> dict[str, Any]:
    """Run the Antigravity CLI, retrying once when it returns an empty SUCCESS."""
    last: dict[str, Any] = {"success": False, "error": "Antigravity CLI produced no output"}
    for attempt in range(2):
        last = _search_cli_once(query, limit, exe)
        if last.get("success") or "no usable results" not in str(last.get("error") or ""):
            return last
        logger.info("gemini-antigravity: empty CLI reply (attempt %d/2), retrying", attempt + 1)
    return last


def _search_cli_once(query: str, limit: int, exe: str) -> dict[str, Any]:
    timeout = _timeout()
    args = [
        exe,
        "-p", _prompt(query, limit),
        "--model", _cli_model(),
        "--output-format", "json",
        "--json-schema", json.dumps(RESULT_SCHEMA),
        "--print-timeout", f"{timeout}s",
    ]
    # Off by default: with auto-approval the CLI starts running shell tools and
    # blows past the print timeout. Its built-in search needs no permission.
    if str(_cfg().get("skip_permissions") or "").strip().lower() in ("1", "true", "yes"):
        args.append("--dangerously-skip-permissions")

    # Run in a throwaway cwd so the CLI never adopts (or writes into) the
    # agent's working directory.
    workdir = tempfile.mkdtemp(prefix="hermes-agy-")
    try:
        proc = subprocess.run(args, cwd=workdir, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout + 20,
                              **_win_hide())
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Antigravity CLI timed out after {timeout + 20}s"}
    except OSError as exc:
        return {"success": False, "error": f"Antigravity CLI could not start: {exc}"}

    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return {"success": False,
                "error": f"Antigravity CLI exit {proc.returncode}: {(proc.stderr or '')[:300]}"}

    envelope = _loads(proc.stdout)
    payload: Any = None
    if isinstance(envelope, dict):
        status = str(envelope.get("status") or "").upper()
        if status and status != "SUCCESS":
            detail = str(envelope.get("response") or "").strip() or (proc.stderr or "").strip()
            return {"success": False, "error": f"Antigravity CLI status {status}: {detail[:300]}"}
        payload = envelope.get("structured_output")
        if payload in (None, {}, []):
            payload = _loads(str(envelope.get("response") or ""))
    if payload is None:
        payload = envelope

    rows = _to_rows(payload, limit)
    if not rows:
        return {"success": False,
                "error": f"Antigravity CLI returned no usable results: {str(proc.stdout)[:300]}"}
    return {"success": True, "data": {"web": _resolve_rows(rows)[:limit]}}


# --------------------------------------------------------------------------- extraction
class _TextExtractor(HTMLParser):
    """Tiny deterministic HTML → text converter (no readability dependency)."""

    _SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: D102
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:  # noqa: D102
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:  # noqa: D102
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        if data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t\u00a0]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(body: str) -> tuple[str, str]:
    """``(title, text)`` from an HTML document; never raises on bad markup."""
    parser = _TextExtractor()
    try:
        parser.feed(body)
    except Exception as exc:  # noqa: BLE001 — malformed markup must not kill the page
        logger.debug("gemini-antigravity: html parse issue: %s", exc)
    return parser.title.strip(), parser.text()


def _one_extract(url: str, max_chars: int) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=30, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
        if resp.status_code >= 400:
            return {"url": url, "title": "", "content": "", "error": f"HTTP {resp.status_code} from origin"}
        ctype = (resp.headers.get("content-type") or "").lower()
        body = resp.text
        if "html" in ctype or body.lstrip().lower().startswith(("<!doctype html", "<html")):
            title, content = html_to_text(body)
        elif "json" in ctype or "text" in ctype or not ctype:
            title, content = url.rsplit("/", 1)[-1] or url, body
        else:
            return {"url": url, "title": "", "content": "", "error": f"unsupported content type: {ctype}"}
        if not content:
            return {"url": url, "title": title, "content": "", "error": "no extractable text"}
        if len(content) > max_chars:
            content = content[:max_chars]
        return {
            "url": url, "title": title or url, "content": content, "raw_content": content,
            "metadata": {"sourceURL": str(resp.url), "title": title or url, "status_code": resp.status_code},
        }
    except httpx.HTTPError as exc:
        return {"url": url, "title": "", "content": "", "error": f"fetch failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "title": "", "content": "", "error": f"extract failed: {exc}"}


# --------------------------------------------------------------------------- provider
class GeminiAntigravityWebSearchProvider(WebSearchProvider):
    """Google-grounded search (Gemini 3.8) + plain HTTP extraction."""

    name = NAME                     # type: ignore[assignment]
    display_name = DISPLAY          # type: ignore[assignment]

    def is_available(self) -> bool:
        """Cheap, no network: a Gemini key or an installed Antigravity CLI."""
        return bool(_api_key() or _cli_path())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        try:
            limit = max(1, min(int(limit), LIMIT_CAP))
        except (TypeError, ValueError):
            limit = 5
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "empty query"}

        transport = _transport()
        if transport == "api":
            key = _api_key()
            if not key:
                return {"success": False, "error":
                        "GEMINI_API_KEY / GOOGLE_API_KEY is not set (run `hermes webgemini mode cli` "
                        "to use the Antigravity CLI instead)"}
            result = _search_api(query, limit, key)
        else:
            exe = _cli_path()
            if not exe:
                return {"success": False, "error":
                        "Antigravity CLI not found — set web.gemini_antigravity.cli_path, install "
                        "`agy`, or run `hermes webgemini mode api` with a GEMINI_API_KEY"}
            result = _search_cli(query, limit, exe)

        if result.get("success"):
            logger.info("Gemini/Antigravity [%s] '%s': %d results",
                        transport, query[:80], len(result["data"]["web"]))
        else:
            logger.warning("Gemini/Antigravity [%s] '%s' failed: %s", transport, query[:80], result.get("error"))
        return result

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        try:
            max_chars = int(kwargs.get("char_limit") or kwargs.get("max_chars") or 15000)
        except (TypeError, ValueError):
            max_chars = 15000
        max_chars = max(1000, min(max_chars, 200000))
        out = [_one_extract(str(url), max_chars) for url in (urls or [])]
        logger.info("Gemini/Antigravity extract: %d/%d pages",
                    sum(1 for r in out if r.get("content")), len(out))
        return out

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": DISPLAY,
            "badge": "google",
            "tag": "Gemini 3.8 with Google Search grounding — Antigravity CLI (Pro sub) or GEMINI_API_KEY.",
            "env_vars": [
                {"key": "GEMINI_API_KEY",
                 "prompt": "Gemini API key (optional — the Antigravity CLI needs no key)",
                 "url": "https://aistudio.google.com/apikey"},
            ],
        }


# --------------------------------------------------------------------------- operator surface
def _backend_names() -> tuple[str, str]:
    """``(search_backend, extract_backend)`` straight from config.yaml."""
    try:
        from hermes_cli.config import load_config

        web = (load_config() or {}).get("web") or {}
        search = str(web.get("search_backend") or web.get("backend") or "")
        return search, str(web.get("extract_backend") or "")
    except Exception:  # noqa: BLE001
        return "", ""


def _win_hide() -> dict[str, Any]:
    """Windows: never flash a console window for our child processes.

    The Hermes gateway/desktop runs without a console, so every ``agy.exe``
    launch would pop a conhost window and close it again a second later. The
    provider spawns that CLI on every search, so this matters far more here than
    in a terminal session.
    """
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _cli_version(exe: str) -> str:
    try:
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=15, **_win_hide())
        return (proc.stdout or proc.stderr or "").strip().splitlines()[0][:40]
    except Exception:  # noqa: BLE001
        return "unknown"


def current_settings() -> dict[str, Any]:
    """Everything the status command shows, resolved (not raw config)."""
    exe, key = _cli_path(), _api_key()
    configured = str(_cfg().get("transport") or "auto").strip().lower()
    search_backend, extract_backend = _backend_names()
    return {
        "version": VERSION,
        "configured_transport": configured if configured in MODES else "auto",
        "active_transport": _transport(),
        "model": _cli_model(),
        "api_model": _api_model(),
        "timeout": _timeout(),
        "cli_path": exe,
        "cli_version": _cli_version(exe) if exe else "",
        "api_key": "set" if key else "missing",
        "search_backend": search_backend,
        "extract_backend": extract_backend,
        "search_active": search_backend == NAME,
        "extract_active": extract_backend == NAME,
    }


def status_text() -> str:
    s = current_settings()
    mode = f"{s['configured_transport']} → {s['active_transport']}"
    if s["active_transport"] == "cli" and s["api_key"] == "missing":
        mode += "  (no API key, so the CLI is used)"
    lines = [
        f"Gemini Antigravity web provider  v{s['version']}",
        f"  mode       : {mode}",
        f"  cli model  : {s['model']}",
        f"  api model  : {s['api_model']}   key: {s['api_key']}",
        f"  cli        : {s['cli_path'] or 'not found'}"
        + (f"  ({s['cli_version']})" if s["cli_version"] else ""),
        f"  timeout    : {s['timeout']}s",
        f"  web_search : {'ACTIVE' if s['search_active'] else 'not selected'}  "
        f"(backend={s['search_backend'] or 'unset'})",
        f"  web_extract: {'ACTIVE' if s['extract_active'] else 'not selected'}  "
        f"(backend={s['extract_backend'] or 'unset'})",
    ]
    if not s["cli_path"] and s["api_key"] == "missing":
        lines.append("  ! neither GEMINI_API_KEY nor the Antigravity CLI was found — searches will fail")
    return "\n".join(lines)


def set_transport(mode: str) -> str:
    """Persist ``web.gemini_antigravity.transport`` and report the result."""
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} — got {mode!r}")

    def _mutate(cfg: dict[str, Any]) -> None:
        web = cfg.setdefault("web", {})
        web.setdefault("gemini_antigravity", {})["transport"] = mode

    _write_config(_mutate)
    s = current_settings()
    if s["active_transport"] == "api" and s["api_key"] == "missing":
        return (f"transport set to {mode}, but no GEMINI_API_KEY / GOOGLE_API_KEY is set — "
                "searches will fail until you add one or switch back to cli")
    picked = s["model"] if s["active_transport"] == "cli" else s["api_model"]
    return f"transport set to {mode} (resolved: {s['active_transport']}, model: {picked})"


def enable_web_search() -> str:
    """Point web.search_backend / web.extract_backend at this provider."""
    def _mutate(cfg: dict[str, Any]) -> None:
        web = cfg.setdefault("web", {})
        section = web.setdefault("gemini_antigravity", {})
        previous = web.get("search_backend") or web.get("backend")
        if previous and previous != NAME:
            section.setdefault("fallback_backend", previous)
        web["search_backend"] = NAME
        web["extract_backend"] = NAME

    _write_config(_mutate)
    return (f"web_search + web_extract now use {NAME}. "
            "A running session picks it up on its next call.")


def disable_web_search() -> str:
    """Restore the backend that was active before ``enable``."""
    fallback = str(_cfg().get("fallback_backend") or "ddgs")

    def _mutate(cfg: dict[str, Any]) -> None:
        web = cfg.setdefault("web", {})
        web["search_backend"] = fallback
        web["extract_backend"] = fallback

    _write_config(_mutate)
    return (f"web_search + web_extract restored to {fallback!r}. "
            "Remove the plugin itself with `hermes plugins disable web-gemini-antigravity`.")


def live_test(query: str = "latest AI news") -> dict[str, Any]:
    """One real grounded search with timings — used by `/webgemini test`."""
    import time

    started = time.time()
    result = GeminiAntigravityWebSearchProvider().search(query, 3)
    return {
        "query": query,
        "ok": bool(result.get("success")),
        "seconds": round(time.time() - started, 1),
        "transport": _transport(),
        "rows": result.get("data", {}).get("web") or [],
        "error": result.get("error") or "",
    }


def format_live_test(result: dict[str, Any]) -> str:
    if not result["ok"]:
        return f"FAILED ({result['transport']}, {result['seconds']}s): {result['error']}"
    head = (f"OK ({result['transport']}, {result['seconds']}s) — "
            f"{len(result['rows'])} results for {result['query']!r}")
    body = [f"  {r['position']}. {r['title'][:70]}\n     {r['url']}" for r in result["rows"]]
    return "\n".join([head, *body])