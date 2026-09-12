# ticker/change

**An equity research terminal for retail traders and investors.**

Enter a ticker, understand the context, sanity-check a trade before you take it.
ticker/change pulls free market data into seven connected views — prices,
fundamentals, options positioning, strategy backtests, and risk — so you can
decide whether a name is worth your attention. It explains what the data
shows; it does not predict the market or execute trades.

- **Live demo:** https://ticker-change.fly.dev

---

## What it is

| | |
| --- | --- |
| **Category** | Equity research terminal / pre-trade context cockpit |
| **Built for** | Retail traders and investors researching their own ideas |
| **Data** | Free-tier sources (`yfinance`, Finnhub, FMP, SEC EDGAR) |
| **Architecture** | Single-process Flask app, SQLite cache, no frontend build step |

**Not:** a broker, a validated alpha engine, financial advice, or a replacement
for Bloomberg. Buyer Signals and ML outputs are structured summaries — not
forecasts. See [What we don't claim](#what-we-dont-claim).

---

## How you'll use it

One mode, one app: **Pre-trade** — is this position worth taking? Enter a
ticker, deep dive, go or pass.

```
Enter a ticker  →  Seven views  →  Checklist on /analytics
```

**Workflow:** enter a symbol → work through the views → pre-trade checklist on
`/analytics`. Your ticker follows you across every view for the length of the
browser session, and the homepage clears it.

---

## Three layers

Everything maps to one of three layers:

```
┌─────────────────────────────────────────┐
│  DECIDE  — pre-trade checklist          │  built
├─────────────────────────────────────────┤
│  UNDERSTAND — GEX, IV, momentum, risk   │  built ← current strength
├─────────────────────────────────────────┤
│  OBSERVE — prices, fundamentals, flow   │  built
└─────────────────────────────────────────┘
```

---

## Views

Enter a ticker and navigate seven connected views, grouped into four
navigation pillars — Markets, Options, Analytics, Strategies. The sub-navbar
carries your symbol across every page.

| View | Pillar | Route | Purpose |
| --- | --- | --- | --- |
| **Price Table** | Markets | `/stock` | Multi-period change, candlestick chart, fundamentals glance |
| **Live Microstructure** | Markets | `/live` | L2 depth, trades feed, options chain + Greeks, payoff simulator |
| **Options Terminal** | Options | `/options` | Dollar GEX by strike, IV term structure, 3D vol surface, Greek matrix |
| **Risk Analytics** | Analytics | `/analytics` | Volatility, VaR, GEX, Monte Carlo, **pre-trade checklist**, Buyer Signals, institutional suite |
| **Market Positioning** | Analytics | `/positioning` | Valuation, analyst ratings, insider activity, 13F holders |
| **Strategy Backtests** | Strategies | `/strategies` | Three selectable strategies, leaderboard, screener, single-ticker backtest |
| **AI Report** | Strategies | `/ai-summary` | LLM desk note synthesising metrics (optional; needs AI key) |

The Options pillar also links **Options Chain** straight to `/live?tab=greeks`
— the chain is a tab of the Live view, not a route of its own.

Also: [`/glossary`](https://ticker-change.fly.dev/glossary) (metric reference),
[`/settings`](https://ticker-change.fly.dev/settings) (API keys),
[`/api/docs`](https://ticker-change.fly.dev/api/docs) (OpenAPI explorer).

---

## Features by layer

### Observe — aggregate public data

**`/stock`**
- 18-period price change (1D–5D through 5Y + YTD), percentage and net dollar
- Custom lookback: *N* trading days or calendar days ago
- ATR (30d), 52-week and all-time ranges
- Fundamentals mini-panel (P/E, EV/EBITDA, short % float, …)
- Interactive candlestick + volume chart (Plotly)

**`/positioning`**
- Valuation and fundamentals (Finnhub)
- Analyst recommendation breakdown
- Insider sentiment and transactions (Finnhub + SEC EDGAR fallback)
- Top institutional 13F holders (FMP + SEC fallback)

SEC EDGAR panels work with **no API keys**.

### Understand — compute context

**`/analytics`**
- Risk stats: vol, max drawdown, beta, VaR (95/99), Sharpe, skew, kurtosis
- Rolling vol, drawdown, and VaR time series
- **Dealer Gamma Exposure (GEX)** — modelled from OI + in-house Black-Scholes
- Options vol smile, cumulative return vs SPY/QQQ
- Monte Carlo forward paths (3m / 6m / 1y)
- **Buyer Signals** — transparent multi-factor tally (`signals.py`), not a black box
- ML Buy/Hold/Sell signal (optional; train with `ml.py`)
- **Institutional Quantitative Analytics** — microstructure (VPIN toxicity,
  Corwin-Schultz spread, Amihud illiquidity, squeeze-risk gauge), macro regime
  and asymmetric bull/bear betas, higher-order Greeks (Vanna, Charm, Vomma),
  variance risk premium, and SEC Form 8-K material events

**`/live`** — three tabs: L2 microstructure, option Greeks, options AI analyst
- Streaming trades (Finnhub WebSocket when keyed; simulated fallback)
- Simulated Level 2 order book with depth chart and liquidity analytics
- **Options chain**: bid/ask, volume, open interest, IV and full Greeks per
  strike, calls and puts mirrored around a centred strike column. Pick 20/40/60
  strikes nearest spot or the whole chain; the header stays pinned while you
  scroll. Deep-linkable at `/live?tab=greeks`
- IV rank / percentile, expected move, max pain, put-call ratios
- Payoff simulator — click any contract to load it
- Strategy posture suggestions based on IV vs HV

**`/options`** — server-rendered options terminal
- Strike-by-strike dollar gamma exposure ($M per 1% move)
- ATM implied-volatility term structure and 3D volatility surface mesh
- Multi-window realised volatility vs current ATM IV
- Strike Greek matrix with Vanna and Charm columns

**`/strategies`** — three selectable strategies over one shared engine
  (`momentum_engine.py`)

| Strategy | Signal |
| --- | --- |
| 12-1 Relative Strength | Cross-sectional rank, top-5 monthly rotation |
| Dual Momentum | Relative rank plus an absolute cash hurdle; failing names sit in cash |
| SMA Trend Following | 50/200 golden cross, ranked by `(SMA50 - SMA200) / SMA200` |

- Universe leaderboard with rotation backtest vs SPY/QQQ, turnover-scaled costs
- Cross-sectional screener with momentum and cash-hurdle filters
- Single-ticker backtest with alpha/beta vs SPY, CSV export on both tables

**`/ai-summary`**
- LLM analyst note grounded in computed metrics (multi-provider fallback)
- Raw data tables rendered alongside for verification

Every metric has an `(i)` tooltip linking to [`/glossary`](/glossary).

**Three display modes** — light, dark, and an Excel mode that re-renders cards
as spreadsheet-style grids with accounting-style negatives. Stored in
`localStorage` and mirrored to a cookie so the server can branch structurally.

### Decide — go/no-go support

**`/analytics` (checklist card)**
- Six pass / warn / fail checks: trend, momentum, IV, earnings, GEX, extension
- One-line verdict summary

**API:** `GET /api/checklist/<ticker>`

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
# → http://127.0.0.1:5001
```

Production-like:

```bash
gunicorn app:app --bind 0.0.0.0:5001 --workers 2
```

Works immediately with **no configuration** — price data and SEC EDGAR panels
need no keys. Add keys via [`.env.example`](.env.example) or `/settings` to
unlock Finnhub/FMP panels, live WebSocket, and AI reports.

**Optional — train the ML model:**

```bash
python ml.py train AAPL MSFT NVDA SPY ...
```

Without `model.pkl`, the ML section is omitted. Predictions older than 30 days
are suppressed.

### Running tests

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v
```

122 tests, run on every push and pull request via
`.github/workflows/tests.yml`. A few of them shell out to `node` to parse and
execute the inline navigation JavaScript in `templates/base.html`; those skip
cleanly if node is absent.

---

## Configuration

All keys optional; every layer degrades gracefully. Settings-page values
override environment variables.

**`SETTINGS_PASSWORD`** gates the `/settings` page itself (HTTP Basic
Auth) — required, since every key saved there is global to the app
instance, not scoped per visitor. Without it, `/settings` refuses to
serve rather than falling back to open access.

### Market data

| Env var | Provider | Powers |
| --- | --- | --- |
| _(none)_ | [SEC EDGAR](https://www.sec.gov/edgar) | Insider + 13F filings — **no key needed** |
| `FINNHUB_API_KEY` | [Finnhub](https://finnhub.io) | Fundamentals, insider, analyst data, live trades |
| `FMP_API_KEY` | [FMP](https://financialmodelingprep.com) | Institutional 13F holders |
| `SEC_USER_AGENT` | — | Required contact string for SEC, e.g. `"Name you@email.com"` |

### AI reports *(any one provider is enough)*

| Env var | Provider |
| --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |

Providers are tried in order; with none configured, AI sections simply don't render.

---

## API

JSON twins of every view, plus health and config:

| Endpoint | Returns |
| --- | --- |
| `GET /api/checklist/<ticker>` | Pre-trade checklist (`?trade_type=long_stock`) |
| `GET /api/stock/<ticker>` | Price, period changes, chart metadata |
| `GET /api/analytics/<ticker>` | Risk stats and metrics |
| `GET /api/positioning/<ticker>` | Valuation, insider, institutional data |
| `GET /api/options-greeks/<ticker>` | Option chain + Greeks (`?expiration=`, `?strikes=N\|all`) |
| `GET /api/options-analysis/<ticker>` | IV rank, expected move, GEX, strategy posture |
| `GET /api/options-terminal/<ticker>` | Dollar GEX, term structure, vol surface, Greek matrix |
| `GET /api/options-ai-report/<ticker>` | LLM options note (needs an AI key) |
| `GET /api/institutional/<ticker>` | Microstructure, macro regime, signals backtest, 8-K events |
| `GET /api/corporate-actions/<ticker>` | Point-in-time identity and split/dividend history |
| `GET /api/raw-sec-filings/<ticker>` | Unfiltered SEC EDGAR submissions |
| `GET /api/chart-data/<ticker>` | OHLCV history |
| `GET /api/active-tickers` | Symbols currently cached (used by the cache warmer) |
| `GET /health` | `{"status": "ok"}` |

Full schema: [`/api/docs`](/api/docs)

---

## How it works

```
  Browser ──►  Flask (routes, Plotly, ML, AI)
                  │
                  ├── yfinance ──► SQLite (daily_prices, api_cache, settings)
                  ├── Finnhub / FMP / SEC EDGAR
                  └── S3 (optional option-chain cache)
```

- **Caching:** 1h price freshness, 24h provider cache, 45s in-process memo
- **Option chains:** S3 first (4h TTL, survives redeploys) when
  `S3_CACHE_BUCKET` is set, then SQLite, then a live fetch, then stale reads of
  either store. With no bucket configured the S3 layer no-ops and SQLite on the
  mounted volume carries the cache on its own
- **Resilience:** stale-while-error — serves cached data when refresh fails
- **ML:** triple-barrier labels, walk-forward validation, LightGBM/sklearn

More detail: [`CLAUDE.md`](CLAUDE.md)

---

## Deploy

**Live:** https://ticker-change.fly.dev

```bash
fly deploy --ha=false   # single instance — SQLite has no multi-writer story
```

Fly.io mounts a persistent volume at `/data` for `stocks.db`. Also ships with
`Dockerfile`, `Procfile`, and `render.yaml` for Render.

---

## Project structure

```
app.py               Routes, charts, GEX/options pricing, page data assembly
db.py                SQLite: daily_prices, api_cache, app_settings, locks
providers.py         Finnhub / FMP / SEC EDGAR / AI clients (fail gracefully)

decide.py            Pre-trade checklist
signals.py           Buyer Signals factor tally
momentum_engine.py   Shared scoring + backtests for all three strategies
backtest_engine.py   Signals walk-forward simulation
event_study.py       Event-study / CAR engine

microstructure.py    VPIN, Corwin-Schultz, Amihud, squeeze risk
macro_engine.py      Macro regime, dual betas, capture ratios
derivatives_alpha.py Higher-order Greeks, variance risk premium
sec_8k.py            SEC 8-K material-event parsing and classification
corporate_actions.py Splits, dividends, point-in-time ticker identity

options/             Options terminal engine (gex, greeks, vol, cones, storage)
s3_cache.py          Optional S3 option-chain cache
warm_s3_cache.py     Cache warmer run by the hourly GitHub Action

ml.py                Offline ML training and inference
ai.py                LLM analyst reports
glossary.py          Metric definitions (tooltips + /glossary)
api_docs.py          OpenAPI spec behind /api/docs
warehouse.py         DuckDB warehouse experiment (not wired into the app)
templates/           Jinja2 + Tailwind CDN + Plotly
```

---

## What we don't claim

- Market-beating alpha or validated edge (unless backtested and published in-app)
- Real-time institutional order flow or measured dealer positioning
- Financial advice or trade recommendations
- Replacement for a broker or professional risk system

GEX is modelled from open interest, not observed flow. Buyer Signals is a
descriptive heuristic tally. ML and AI outputs are educational tools on
delayed data.

---

## Docs

| Doc | Contents |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | Architecture guide for contributors |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Engineering roadmap and sprint plan |
| [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) | Planned DuckDB / dbt ELT pipeline — design notes, not built |

---

## Tech stack

Flask · yfinance · pandas · numpy · Plotly · LightGBM/sklearn · Tailwind (CDN) ·
Jinja2 · Gunicorn · Fly.io

Do not commit secrets — use environment variables or `/settings`.
