# web-gemini-antigravity

**Every `web_search` in Hermes becomes a Google-grounded Gemini 3.8 search — whatever model you chat with.**

Hermes routes `web_search` through a pluggable provider registry, not through the model. This
plugin registers itself as that provider, so a model with no native search (DeepSeek V4.1 on
Surplus, a local Llama, an OpenRouter endpoint, GPT, Claude) still gets live, grounded, cited
results. The chat model never sees Google; it just gets rows.

```
you → hermes (any model) → web_search tool → [ gemini-antigravity provider ] → Google Search grounding
```

## Two transports, one command to switch

| mode | what runs | needs | latency |
|---|---|---|---|
| `cli` | Google Antigravity CLI (`agy`), `gemini-3.8-flash`, its own Google Search tool | an installed + signed-in `agy` (a Google Pro plan) — **no API key** | 20–60 s |
| `api` | `generativelanguage.googleapis.com` `:generateContent` with `tools:[{"google_search":{}}]` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | 2–6 s |
| `auto` (default) | `api` when a key exists, else `cli` | — | — |

```bash
hermes webgemini status          # mode, model, credentials, whether search/extract are active
hermes webgemini mode api        # switch transport (auto | api | cli) — persisted to config.yaml
hermes webgemini test "query"    # one real grounded search with timings
hermes webgemini on | off        # point web_search/web_extract here, or restore the previous backend
```

Inside a session the same thing is `/webgemini [status|auto|api|cli|test <query>]`.

## Install

```bash
# 1. put the directory in the user plugin folder
#    Windows  %LOCALAPPDATA%\hermes\plugins\gemini_antigravity\
#    Linux    ~/.hermes/plugins/gemini_antigravity/
# 2. enable it — the manifest name, not the folder name
hermes plugins enable web-gemini-antigravity
# 3. point the web tools at it
hermes webgemini on
# 4. prove it
hermes webgemini test "latest CPI print"
```

Disable without deleting: `hermes plugins disable web-gemini-antigravity` (and `hermes webgemini off`
to give `web_search` back to the previous backend).

## What it returns

The legacy provider contract, unchanged — the tool wrapper does not translate:

```json
{"success": true, "data": {"web": [
  {"title": "…", "url": "https://real-page", "description": "…", "position": 1}
]}}
```

`web_extract` is served too: plain HTTP fetch (30 s timeout, redirects followed), HTML → text with
`html.parser`, `raw_content` mirrored, per-URL `error` instead of an exception, `char_limit`
respected.

## Why it doesn't break (the parts that took the longest)

- **Grounding URLs are redirects.** Gemini hands back
  `vertexaisearch.cloud.google.com/grounding-api-redirect/…`, not the article. Every such URL is
  resolved with a parallel `HEAD` (follow redirects) so the agent gets `bls.gov/news.release/…`
  instead of an opaque token. Unresolvable ones keep the working link.
- **Empty `SUCCESS` turns.** Asked to "search carefully", the Antigravity agent reaches for a shell
  tool; headless mode auto-denies it and returns an empty turn. The prompt forbids non-search tools
  and one retry covers the rest. `--dangerously-skip-permissions` is *not* used (it makes the CLI
  run commands and blow the print timeout).
- **Effort changes wall time.** `gemini-3.8-flash-low` ≈ 20 s; `-medium` ≈ 90 s; `-high` routinely
  exceeds a 120 s print timeout. Search doesn't need deep reasoning, so `-low` is the default and
  every model/effort value is config-overridable.
- **Model replies are not JSON.** Code fences, prose wrappers, JSON-inside-JSON, single objects and
  half-written arrays are all handled (`_loads`), and a bare result object still becomes a row.
- **Config writes go through Hermes.** `hermes config set`-equivalent (`load_config` +
  `save_config(merge_existing=True)`), never a hand-edited YAML, so other settings survive.

## Configuration

| key | default | meaning |
|---|---|---|
| `web.search_backend` | — | `gemini-antigravity` makes this provider serve `web_search` |
| `web.extract_backend` | — | same provider serves `web_extract` |
| `web.gemini_antigravity.transport` | `auto` | `auto` \| `api` \| `cli` |
| `web.gemini_antigravity.model` | `gemini-3.8-flash-low` | Antigravity CLI model |
| `web.gemini_antigravity.api_model` | `gemini-3.8-flash` | Gemini API model |
| `web.gemini_antigravity.timeout` | `90` | seconds (CLI gets +20 s subprocess headroom) |
| `web.gemini_antigravity.cli_path` | auto | explicit path to `agy` |
| `web.gemini_antigravity.skip_permissions` | off | adds `--dangerously-skip-permissions`; leave off |
| `web.gemini_antigravity.fallback_backend` | `ddgs` | backend restored by `hermes webgemini off` |

`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) lives in `.env`, read through Hermes' provider helper.

## Tests

```bash
cd gemini_antigravity
python -m pytest tests -q      # 28 offline checks: parsing, rows, transport, extract, commands
hermes plugins validate .      # the same gate catalog CI runs — passes with no warnings
python scripts/selftest.py     # live: status + two grounded searches + one extract
```

The pytest suite and `validate` are offline: no network, no config writes, no `agy`.

## Disclosure (catalog rule 13)

- Network calls: `generativelanguage.googleapis.com` (api mode), `vertexaisearch.cloud.google.com`
  redirect resolution, and the pages `web_extract` fetches.
- Runs a local executable: the Antigravity CLI (`agy`) in print mode, in a throwaway temp cwd, with
  a hard timeout. It is **not** run with permission auto-approval.
- No telemetry, no self-updater, no credential files read, no writes outside Hermes' own config.

## License

MIT.