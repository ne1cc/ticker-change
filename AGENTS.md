# AGENTS.md

Workflow rules for agents: read [`.github/AGENT-WORKFLOW.md`](.github/AGENT-WORKFLOW.md) before your first commit.

## Commands

- Install: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-test.txt`
- Test (offline; fetchers are mocked): `python -m pytest tests/ -q`
- Lint / format: none configured
- Run (dev): `python3 app.py` → http://127.0.0.1:5001

## Architecture

- Single-process Flask app. `app.py` holds routes and page builders (`get_*`/`compute_*` return a dict for a Jinja template); satellites: `db.py` (SQLite cache), `providers.py`, `ai.py` (LLM reports), `options/` (greeks, GEX, vol), `glossary.py`, `brief*.py`.
- No frontend build: Tailwind CDN, ECharts JSON mounts in `templates/`.
- Data layers degrade instead of raising: in-process memo → SQLite `daily_prices`/`api_cache` → stale rows → `None`. Option chains: S3 → SQLite → Yahoo → Cboe.
- Page sections run inside `_guard_section(label, fn, default, timeout, status)` so one failure never blanks a page.

## Conventions

- Always render something: a missing or failed source becomes a declared missing-state (null plus a reason), never a crash, a blank or a fake zero.
- Evidence-page tooltips come from `glossary.py` via the `metric(label, key)` macro in `templates/_macros.html`; brief metrics come from `brief_contracts.py`.
- Reuse existing math (`options/`, `derivatives_alpha.py`); never add another Black-Scholes or GEX implementation.
- Tests are unittest/pytest style in `tests/` and must not touch the network.
- Conventional Commits; commit subjects end with `(#N)`.

## Do not touch

- `deploy/lightsail/`, `fly.toml`, `render.yaml`, `.github/workflows/` — merging to `main` with green CI is the deploy.
- The S3 key layout in `s3_cache.py` without the same change in `warm_s3_cache.py`.
- `warehouse.py` (unwired experiment). Never commit `.env`, `stocks.db` or `model.pkl`.
