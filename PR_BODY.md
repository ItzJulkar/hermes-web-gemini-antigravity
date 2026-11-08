# PR title

Add `web-gemini-antigravity` to the plugin catalog

# PR body

**Plugin:** `web-gemini-antigravity` (`gemini-antigravity` provider)
**Repo:** https://github.com/ItzJulkar/hermes-web-gemini-antigravity
**Pin:** `<sha>` — `hermes plugins validate .` passes locally with no warnings.

## What it does

Registers a `WebSearchProvider` that serves both `web_search` and `web_extract`, so **every search
is Google-grounded Gemini 3.8 regardless of which model the agent is running**. That is the point:
`web_search` is resolved by `agent/web_search_registry.py` from `web.search_backend`, not by the
model, so a DeepSeek / Llama / GPT chat model with no native search still gets cited, live results.

Two transports behind one config key:

- `cli` — the Google Antigravity CLI (`agy`) in print mode, `gemini-3.8-flash` with its own Google
  Search tool. Uses the user's Google Pro subscription; no API key. 20–60 s.
- `api` — `generativelanguage.googleapis.com` `:generateContent` with
  `tools: [{"google_search": {}}]`, `GEMINI_API_KEY`. 2–6 s.
- `auto` (default) picks `api` when a key exists, else `cli`.

Plus an operator surface so it is not a black box: `hermes webgemini status|mode|on|off|test`, and
the same as `/webgemini` in a session.

## Hermes surfaces used

- `ctx.register_web_search_provider()` (search + extract capabilities)
- `ctx.register_command()` → `/webgemini`
- `ctx.register_cli_command()` → `hermes webgemini`
- Reads config through `hermes_cli.config.load_config`; writes through `load_config()` +
  `save_config(merge_existing=True)` (same path as `hermes config set`, no hand-edited YAML).
- Secrets through `get_provider_env()` (`GEMINI_API_KEY` / `GOOGLE_API_KEY` from `.env`).

No core rebinds, no monkeypatching, no bundled binaries.

## Testing

- `python -m pytest tests -q` → **26 passed** (offline: parsing, row normalisation, transport
  resolution, timeout clamps, extract shapes, slash-command routing, manifest declarations).
- `hermes plugins validate .` → every check ✓, no warnings.
- Live: `hermes webgemini test "current ethereum staking yield"` returns real URLs
  (`stakingrewards.com`, `beaconcha.in`) in ~30 s; `web_extract` verified on a real page.
- Model-agnostic proof: the session that produced those results runs on Surplus
  `deepseek-v4.1-flash-1017` — a model with no native search.

## Disclosure (rule 13)

- **Network:** `generativelanguage.googleapis.com` (api mode), `vertexaisearch.cloud.google.com`
  redirect resolution (HEAD, no body), and the pages `web_extract` fetches on the user's behalf.
- **Local process:** in `cli` mode it runs the user's installed `agy` binary in print mode, in a
  throwaway temp cwd, with `--print-timeout` and a subprocess timeout. It does **not** pass
  `--dangerously-skip-permissions` by default, so the CLI cannot approve its own shell actions.
- **No credentials are read from files.** The CLI's own login is used by the CLI, not by the plugin.
- No telemetry, no self-updater, no background processes.

## Demo

Under 90 s: `hermes webgemini status` → `hermes webgemini test "…"` → the same query through the
`web_search` tool while the chat model is DeepSeek → `/webgemini api` switch.