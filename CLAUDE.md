# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ticker-change` is a single-process Flask equity-research dashboard. The
front door is the daily brief (`/` universe dashboard, `/brief/<ticker>` per
ticker); the seven connected views are its drill-down evidence pages — price
table (`/stock`), risk analytics (`/analytics`), market positioning
(`/positioning`), live microstructure (`/live`), strategies/momentum
(`/strategies`, with `/momentum` as an alias), options terminal (`/options`),
and an AI analyst report (`/ai-summary`) — built from free-tier data sources
(`yfinance`, Finnhub, FMP, SEC EDGAR) that all degrade gracefully when
unconfigured or rate-limited.

There is no frontend build step — Tailwind is loaded from the CDN and charts
are native ECharts JSON mounts embedded in Jinja templates; stock candles use
TradingView Lightweight Charts.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-test.txt   # test-only deps (pytest, duckdb)

# Run (dev, auto-reload) — http://127.0.0.1:5001
python3 app.py

# Run (prod-like)
gunicorn app:app --bind 0.0.0.0:5001 --workers 2

# Tests — unittest-style classes run under pytest; no network required
# (tests mock fetchers and skip when cached data is unavailable)
python -m pytest tests/ -q

# Train the ML signal model (optional; without it the ML section is omitted)
python ml.py train AAPL MSFT NVDA SPY ...

# Train on the whole S&P 500 -- backfill first, since train only reads
# price history already cached in SQLite, never fetching from yfinance itself
python ml.py backfill --sp500
python ml.py train --sp500

# Warm the S3 option-chain cache manually (same script the hourly GH Action runs)
python warm_s3_cache.py
```

Verify changes by running the app and hitting the relevant route/`/api/*`
endpoint, plus the test suite.

Deploying: the live Lightsail demo self-deploys — merging to `main` with green
CI *is* the deploy (see Deployment below). Never deploy manually. Fly.io
(`fly deploy --ha=false`) remains an alternate target; confirm with the user
before running it.

## Architecture

### Core: `app.py` (~4,230 lines) plus ~15 satellite modules

Routes, page data-assembly functions, chart data assembly (ECharts), GEX/dealer
positioning math, and page orchestration live in `app.py`. Each page route has
a `get_*`/`compute_*` builder that assembles a data dict passed straight to
the Jinja template; three pages (`/stock`, `/analytics`, `/positioning`)
also have a JSON twin under `/api/*` returning the same data minus embedded
chart HTML (`/options`, `/strategies`, `/live`, and `/ai-summary` have no
twin; `/stock`'s twin omits fundamentals).

Supporting modules:

- `db.py` — SQLite layer (`stocks.db`): `daily_prices` cache, generic
  `api_cache` (provider/AI response cache with TTL), `app_settings`
  (user-entered API keys from `/settings`), `gex_snapshots` (schema exists;
  no production writer yet — wired candidate, see Known gaps), and a
  `try_claim_lock` helper for cross-worker locking.
- `providers.py` — Finnhub / FMP / SEC EDGAR clients plus AI-provider key
  resolution. Every provider call returns `None` on failure instead of
  raising. NOTE: key-resolution order is asymmetric — Finnhub/FMP prefer env
  vars and fall back to `/settings` values; AI and SEC prefer `/settings`.
  CLAUDE.md previously claimed the opposite; check `providers.py:60-130`
  before relying on precedence.
- `options/` — package: Black-Scholes greeks (`options.greeks`), GEX, vol
  cones, terminal computation. app.py still carries a *second*, older BS
  implementation for the GEX profile (shadowing import) — consolidation is
  pending; don't add a third.
- `momentum_engine.py`, `backtest_engine.py` — strategy scoring and
  walk-forward backtests (incl. Deflated Sharpe). `backtest_engine` reuses
  `signals.py`'s private factor builders.
- `signals.py` — deterministic multi-factor composite (0-100). NOT imported
  by app.py at runtime; consumed via backtest_engine only.
- `ml.py` — gradient-boosted Buy/Hold/Sell model trained offline via
  `python ml.py train ...` on triple-barrier labels; serialized to `model.pkl`
  (gitignored). Gracefully absent when untrained/stale.
- `ai.py` — LLM analyst report generation (standard/comprehensive/options
  variants), multi-provider fallback chain (Anthropic → OpenAI → Gemini →
  OpenRouter), results cached in `api_cache` (12h).
- `brief.py` — deterministic daily-brief builder: cache-only context →
  sections → contract validation (CPU only, no network).
- `brief_contracts.py` — machine-checked definition/unit/window/source/
  missing-state for every brief metric; version bump on label/definition
  change, never silently.
- `snapshot_store.py` — generic publication store (payload + `current`
  pointer + two-edition retention in `api_cache`); pointer moves last so a
  failed publish leaves the last-good snapshot serving.
- `glossary.py` — single source of truth for evidence-page metric tooltips
  and `/glossary` page entries (injected as a Jinja global); brief surfaces
  take definitions from `brief_contracts.py` instead — see the tooltip note
  below.
- `s3_cache.py` — optional S3-backed option-chain cache; no-ops to SQLite
  `api_cache` when `S3_CACHE_BUCKET` is unset. `warm_s3_cache.py` is the
  standalone GH-Action script (duplicates the S3 key layout — keep both in
  sync when changing keys).
- `sp500.py` — live S&P 500 constituents + price backfill. Kept independent
  of `app.py` (importing app.py starts background warmer threads).
- `corporate_actions.py`, `decide.py` (pre-trade checklist),
  `derivatives_alpha.py`, `event_study.py`, `macro_engine.py`,
  `microstructure.py`, `sec_8k.py`, `api_docs.py` — analytics helpers behind
  `/api/corporate-actions`, `/api/checklist`, `/api/institutional`.
- `warehouse.py` — DuckDB warehouse experiment. Exists with tests but is
  UNWIRED (not imported by app.py; duckdb is a test-only dependency).

### Caching, layered for a tight free-tier budget

1. **In-process memo** (`_PRICE_MEMO` in `app.py`) — collapses duplicate
   price look-ups within/across nearby requests (45s TTL; unbounded per-symbol
   entries — bounded candidate, see Known gaps).
2. **SQLite `daily_prices`** — 1h freshness (`db.is_fresh`); refreshed from
   `yfinance` with retry/backoff on a miss.
3. **Stale-while-error** — if a refresh fails but older rows exist, those are
   served instead of an error.
4. **`api_cache`** — generic 24h (provider) / 12h (AI) TTL cache keyed by
   `(provider, key)`. Expired rows are never purged (unbounded growth).
5. **Option chains get their own path**: `get_cached_chain`/
   `get_cached_expirations` read S3 first (4h TTL), then SQLite `api_cache`,
   then live `yfinance` (written back to both), falling back to stale S3 →
   stale SQLite → `None`. A GitHub Action (`.github/workflows/warm-cache.yml`)
   runs `warm_s3_cache.py` hourly during market hours.

When changing fetch/cache behavior, preserve the "always render something"
contract — every layer that can fail should fall through to the next rather
than raising. Analytics builders (`compute_analytics`, `compute_momentum`)
guard themselves and return `None`/`{}` on internal failure.

### UI mode plumbing (light / dark / excel)

Three-state theme stored in `localStorage` (pre-paint, anti-flash) and
mirrored to a `ui_mode` cookie so Flask can branch structurally. The
`inject_ui_mode` context processor exposes `excel_mode` to every template.
CSS-only "Excel" paint lives in `static/excel-mode.css` (keyed off
`html.excel`); structural conversion (cards → spreadsheet-style grids) uses
Jinja macros in `templates/_excel.html` — currently only `/analytics`
imports it; other pages get CSS paint only. The `xlfmt` Jinja filter renders
negatives in accounting-style parens.

`templates/_macros.html` provides the `metric(label, key)` macro used
throughout for the `(i)` tooltip that links every stat to its `/glossary`
entry — reuse it instead of hand-rolling label markup for new metrics.

Tooltip/definition authority is deliberately split by surface: evidence
pages take tooltips from `glossary.py` via that macro; the brief surfaces
(`/`, `/brief/*`) take definitions and missing-states from
`brief_contracts.py`, whose provenance is machine-validated per metric.
Adding a brief metric means updating the brief contracts, not `glossary.py`
— and adding an evidence-page metric means `glossary.py`, not the contracts.

## Known gaps (deliberate, tracked — don't "fix" silently)

- `api_cache` expired rows are never purged; no schema migration mechanism.
- `/api/config` returns only `has_finnhub` (key leak fixed); the live page's
  browser-direct Finnhub stream is therefore degraded to simulation until a
  server-side proxy replaces it.
- `/api/institutional` requires `WARM_CACHE_TOKEN` (503 when unset).
- `gex_snapshots` table exists with no production writer.
- `warehouse.py` unwired; `docs/ORCHESTRATION.md` / `PROJECT_CONTEXT.md`
  describe the planned (not-built) DuckDB/dbt/Evidence extension.
- Three coexisting "Excel-ish" visual systems; consolidation pending.

## Deployment

The live demo runs on AWS Lightsail (Nginx, HTTPS, persistent SQLite) and
self-deploys: a systemd timer on the host runs
`deploy/lightsail/deploy.sh` every 5 minutes, building the newest `main`
commit whose Tests workflow passed and swapping containers only once the new
one answers `/health` (otherwise the previous container is restored). There is
no deploy command to run from the repo — merge to `main` with green CI. After
changing anything in `deploy/lightsail/`, reinstall it on the host (commands
in the README's Deploy section).

Fly.io remains a configured alternate target (`fly.toml`):
`fly deploy --ha=false` — single-instance only, because SQLite has no
multi-writer story. `fly.toml` sets `DB_PATH=/data/stocks.db` and
`MODEL_PATH=/data/model.pkl`; `db.py`/`ml.py` read these from the
environment, so the volume mount makes both survive redeploys. `model.pkl` is
excluded from the Docker build context; train it once in place with
`fly ssh console -C "python ml.py train ..."`.

`render.yaml` exists as a further alternative but has no persistent disk —
SQLite and `model.pkl` are ephemeral there. API keys are optional everywhere
and can be set via environment, `.env`, or the `/settings` page (which
requires `SETTINGS_PASSWORD`, HTTP Basic). See `.env.example` for the full
list.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues on ne1cc/ticker-change via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels are used verbatim (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
