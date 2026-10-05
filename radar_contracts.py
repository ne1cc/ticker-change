"""Radar metric contracts: definition, unit, window, source, missing state.

Every metric a radar builder emits must have an entry here; the Audit preset
and security drawer render these verbatim. Version bump = label/definition
change, never silently.
"""
from __future__ import annotations

METRIC_CONTRACT_VERSION = 2
VALUE_SCORE_VERSION = 1

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))


def _c(term, definition, unit, window, source, missing_state):
    return {"term": term, "definition": definition, "unit": unit,
            "window": window, "source": source, "missing_state": missing_state}


_CONTRACTS = {
    # --- identity ---
    "symbol": _c("Symbol", "Normalized ticker at snapshot build time.", "text",
                 "snapshot", "universe", "never"),
    "sector": _c("Sector", "GICS sector from the warmed fundamentals payload.",
                 "text", "fundamentals payload as-of", "yfinance .info",
                 "unknown — payload missing"),
    "mkt_cap": _c("Market cap", "Shares outstanding x price, as reported by the "
                  "fundamentals payload.", "USD", "fundamentals payload as-of",
                  "yfinance .info", "unknown — payload missing"),
    # --- drift: reaction ---
    "print_date": _c("Announcement date", "Earnings announcement date as published "
                     "by yfinance. Session alignment is unverified (no before-open/"
                     "after-close flag at source).", "date", "event", "yfinance",
                     "row absent without a date"),
    "alignment": _c("Session alignment", "Reaction session = first trading session "
                    "on/after the announcement date. Source lacks an amc/bmo flag, "
                    "so alignment is unverified in v1.", "state", "event", "derived",
                    "always 'unverified' in v1"),
    "sessions_since": _c("Sessions since reaction", "Trading sessions elapsed since "
                         "the reaction session. Weekends and holidays do not advance "
                         "it.", "sessions", "event", "derived from price calendar",
                         "absent before the reaction session"),
    "sessions_to_20d": _c("Sessions to 20d", "Remaining sessions in the 20-session "
                          "drift window: max(0, 20 - sessions since reaction).",
                          "sessions", "event", "derived",
                          "0 once the window completes"),
    "eps_estimate": _c("EPS estimate", "Consensus EPS for the reported quarter.",
                       "USD/share", "event", "yfinance", "unknown estimate"),
    "eps_actual": _c("EPS actual", "Reported EPS for the quarter.", "USD/share",
                     "event", "yfinance", "absent for upcoming prints"),
    "surprise_pct": _c("Surprise %", "(actual - estimate) / |estimate| x 100. "
                       "|estimate| < 0.05 renders NM (low-denominator).", "%",
                       "event", "yfinance", "NM when estimate near zero or absent"),
    "surprise_nm": _c("Surprise NM", "True when surprise % is not meaningful: "
                      "|estimate| < 0.05 or the estimate is absent. Renders NM in "
                      "place of the percentage.", "state", "event", "derived",
                      "always False when surprise is defined"),
    "gap_pct": _c("Print-day gap", "open(reaction) / close(reaction-1) - 1 — the "
                  "opening gap only, not the full session.", "%", "reaction session",
                  "derived from daily_prices", "absent without prior close"),
    "reaction_ret": _c("Reaction-day return", "close(reaction) / close(reaction-1) "
                       "- 1 — the full first-session move.", "%", "reaction session",
                       "derived from daily_prices", "absent without prior close"),
    "runup_5d": _c("Pre-print 5d run-up", "close(reaction-1) / close(reaction-6) "
                   "- 1 — anticipation before the print.", "%", "5 sessions pre",
                   "derived from daily_prices", "absent without history"),
    "rel_volume": _c("Print-day relative volume", "volume(reaction) / mean volume of "
                     "the 20 sessions ending reaction-1 (reaction day excluded).",
                     "x", "reaction session", "derived from daily_prices",
                     "absent without history"),
    # --- drift: continuation ---
    "drift_5d": _c("Drift 5d", "close(t0+5 sessions) / close(t0) - 1. Completed "
                   "windows only; excludes the reaction day.", "%", "5 sessions",
                   "derived from daily_prices", "incomplete until 5 sessions elapse"),
    "drift_10d": _c("Drift 10d", "close(t0+10 sessions) / close(t0) - 1. Completed "
                    "windows only.", "%", "10 sessions", "derived from daily_prices",
                    "incomplete until 10 sessions elapse"),
    "drift_20d": _c("Drift 20d", "close(t0+20 sessions) / close(t0) - 1. Completed "
                    "windows only.", "%", "20 sessions", "derived from daily_prices",
                    "incomplete until 20 sessions elapse"),
    "drift_to_date": _c("Drift to-date", "close(latest) / close(t0) - 1 over the "
                        "elapsed sessions (n shown alongside). Not comparable "
                        "across rows.", "%", "elapsed sessions", "derived",
                        "absent on the reaction day itself"),
    "drift_to_date_sessions": _c("Drift to-date sessions", "Elapsed sessions n used "
                                 "by drift to-date: sessions between the reaction "
                                 "session and the latest close.", "sessions",
                                 "elapsed sessions", "derived",
                                 "0 on the reaction day itself"),
    "excess_10d_sector": _c("Excess 10d vs sector", "drift_10d minus the sector ETF "
                            "return over the same sessions. Sector = GICS map.",
                            "%", "10 sessions", "derived (sector ETF)",
                            "absent without sector ETF history"),
    "car_30": _c("CAR (-10/+30)", "Market-model cumulative abnormal return; "
                 "estimation -120/-21 sessions vs SPY. Audit/drawer only — not a "
                 "screening metric in v1.", "%", "-10/+30 sessions",
                 "event_study engine", "absent under 30 estimation bars"),
    "hist_beat_median": _c("Own-name beat history", "Median 10-session post-print "
                           "drift after beats, this name only, completed windows "
                           "only.", "%", "per event, 10 sessions",
                           "cached earnings history", "limited history < 3 events"),
    "hist_miss_median": _c("Own-name miss history", "Median 10-session post-print "
                           "drift after misses, this name only, completed windows "
                           "only.", "%", "per event, 10 sessions",
                           "cached earnings history", "limited history < 3 events"),
    "next_print": _c("Next print", "Earliest upcoming announcement date; "
                     "confirmation state not available at source.", "date",
                     "event", "yfinance", "absent when unscheduled"),
    # --- value ---
    "fwd_pe": _c("Forward P/E", "Price / forward EPS as published. Non-positive "
                 "renders NM and is excluded from percentile pools.", "x",
                 "fundamentals payload as-of", "yfinance .info", "NM when <= 0"),
    "trail_pe": _c("Trailing P/E", "Price / trailing EPS. Non-positive renders NM.",
                   "x", "fundamentals payload as-of", "yfinance .info", "NM when <= 0"),
    "ps": _c("Price / Sales", "Market cap / trailing revenue (as published). "
             "Non-positive renders NM.", "x", "fundamentals payload as-of",
             "yfinance .info", "NM when <= 0"),
    "ev_ebitda": _c("EV / EBITDA", "Enterprise value / EBITDA as published. "
                    "Non-positive renders NM.", "x", "fundamentals payload as-of",
                    "yfinance .info", "NM when <= 0"),
    "peg": _c("PEG", "Trailing PEG as published. Optional column: the earnings "
              "growth horizon is not reliable at source.", "x",
              "fundamentals payload as-of", "yfinance .info", "unknown"),
    "div_yield": _c("Dividend yield", "Trailing yield as published. Context only — "
                    "not a safety measure.", "%", "fundamentals payload as-of",
                    "yfinance .info", "unknown — non-payer or missing"),
    "fcf_yield": _c("FCF yield", "Free cash flow / market cap, both as published "
                    "in the fundamentals payload.", "%", "fundamentals payload as-of",
                    "yfinance .info", "unknown when either input missing"),
    "rev_growth": _c("Revenue growth", "Year-over-year revenue growth as published.",
                     "%", "fundamentals payload as-of", "yfinance .info", "unknown"),
    "eps_growth": _c("EPS growth", "Year-over-year earnings growth as published.",
                     "%", "fundamentals payload as-of", "yfinance .info", "unknown"),
    "mom_12_1": _c("Momentum 12-1", "12-month return excluding the most recent "
                   "month.", "%", "12 months - 1", "daily_prices closes",
                   "absent under 273 bars"),
    "mom_6m": _c("Momentum 6m", "6-month return.", "%", "6 months",
                 "daily_prices closes", "absent under 273 bars"),
    "mom_3m": _c("Momentum 3m", "3-month return.", "%", "3 months",
                 "daily_prices closes", "absent under 273 bars"),
    "mom_1m": _c("Momentum 1m", "1-month return.", "%", "1 month",
                 "daily_prices closes", "absent under 273 bars"),
    "ann_vol": _c("Annualized volatility", "Standard deviation of daily returns, "
                  "annualized, trailing year.", "%", "1 year", "daily_prices closes",
                  "absent under 273 bars"),
    "value_score": _c("Value score", "Weighted mean of available component "
                      "percentiles: 50% cheapness (fwd P/E, P/S), 25% growth "
                      "(revenue, EPS), 25% momentum percentile. Higher = cheaper/"
                      "stronger. Weights and states per VALUE_SCORE_VERSION.",
                      "0-100", "snapshot", "derived",
                      "low-coverage flag under 50% inputs"),
    "peer_percentile": _c("Peer percentile", "Percentile within the peer group "
                          "(industry, min 3 valid peers; sector fallback). Higher = "
                          "cheaper for multiples (inverted).", "0-100", "snapshot",
                          "derived", "insufficient-peers below the minimum"),
    "quadrant": _c("Quadrant", "Valuation cheap/expensive (vs peer median) x "
                   "improving/deteriorating (12-1 momentum sign in v1).", "state",
                   "snapshot", "derived", "needs valuation and momentum"),
    "short_pct_float": _c("Short % float", "Short interest as % of float at the SI "
                          "settlement date. A dated snapshot, not live.", "%",
                          "SI settlement date", "yfinance .info", "unknown — feed absent"),
    # --- squeeze watch ---
    "si_settlement": _c("SI settlement date", "Settlement date of the short-interest "
                        "snapshot.", "date", "report", "yfinance .info",
                        "absent — age badge hidden"),
    "shares_short": _c("Shares short", "Outstanding short shares at settlement.",
                       "shares", "SI settlement date", "yfinance .info", "unknown"),
    "prior_shares_short": _c("Prior-report shares short", "Shares short at the "
                             "previous settlement.", "shares", "prior report",
                             "yfinance .info", "unknown — no prior"),
    "si_change": _c("SI change (report/report)", "Shares short vs the prior "
                    "settlement, same units.", "%", "report-over-report",
                    "yfinance .info", "unknown without both reports"),
    "dtc_vendor": _c("Days to cover (vendor)", "Vendor-reported days to cover.",
                     "days", "SI settlement date", "yfinance .info", "unknown"),
    "dtc_calc": _c("Days to cover (calc)", "Shares short / 20-session average daily "
                   "volume.", "days", "20 sessions to latest close", "derived",
                   "absent without history"),
    "dist_52w_high": _c("Distance below 52w high", "(52w high - price) / 52w high; "
                        "0 = at the high; positive = below.", "%", "trailing year",
                        "daily_prices closes", "absent under history"),
    "price_5d": _c("5d price change", "close(latest) / close(5 sessions ago) - 1.",
                   "%", "5 sessions", "daily_prices closes", "absent under history"),
    "price_20d": _c("20d price change", "close(latest) / close(20 sessions ago) - 1.",
                    "%", "20 sessions", "daily_prices closes", "absent under history"),
    "dollar_volume": _c("Dollar volume", "Median close x volume over the trailing 20 "
                        "sessions.", "USD", "20 sessions", "daily_prices",
                        "absent under history"),
    "squeeze_score": _c("Crowding score", "Heuristic 0-100 composite of crowding, "
                        "activation, and tradability sub-scores (components "
                        "inspectable in the drawer). Descriptive level label, not a "
                        "probability.", "0-100", "snapshot", "derived",
                        "defaults flagged 'assumed' without real SI"),
}

METRIC_KEYS_BY_TAB = {
    "drift": ["symbol", "sector", "mkt_cap", "print_date", "alignment",
              "sessions_since", "sessions_to_20d", "eps_estimate", "eps_actual",
              "surprise_pct", "surprise_nm", "gap_pct", "reaction_ret",
              "runup_5d", "rel_volume", "drift_5d", "drift_10d", "drift_20d",
              "drift_to_date", "drift_to_date_sessions", "excess_10d_sector",
              "car_30", "hist_beat_median", "hist_miss_median", "next_print"],
    "value": ["symbol", "sector", "mkt_cap", "fwd_pe", "trail_pe", "ps", "peg",
              "ev_ebitda", "div_yield", "fcf_yield", "rev_growth", "eps_growth",
              "mom_12_1", "mom_6m", "mom_3m", "mom_1m", "ann_vol",
              "short_pct_float", "value_score", "peer_percentile", "quadrant"],
    "squeeze": ["symbol", "sector", "mkt_cap", "short_pct_float", "si_settlement",
                "shares_short", "prior_shares_short", "si_change", "dtc_vendor",
                "dtc_calc", "squeeze_score", "dist_52w_high", "price_5d",
                "price_20d", "dollar_volume"],
}


def contract(key: str) -> dict:
    return _CONTRACTS[key]
