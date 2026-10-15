"""Brief metric contracts: definition, unit, window, source, missing state.

Every metric the brief builder emits must have an entry here. A contract's
source is exactly one of SOURCE_KEYS, and meta.receipts in a built brief
carries the matching as-of read from the cache-age path — so a number's
provenance is machine-checkable. Version bump = label/definition change,
never silently.
"""
from __future__ import annotations

BRIEF_CONTRACT_VERSION = 1

SOURCE_KEYS = ("prices", "earnings", "fundamentals", "short")


def _c(term, definition, unit, window, source, missing_state):
    return {"term": term, "definition": definition, "unit": unit,
            "window": window, "source": source, "missing_state": missing_state}


_CONTRACTS = {
    "ticker": _c("Ticker", "Normalized ticker the brief was built for.",
                 "text", "snapshot", "context", "never"),
    # --- price: the last completed session, and whether it matters ---
    "session_date": _c("Session date", "Date of the last completed session in "
                       "the daily_prices cache.", "date", "last session",
                       "prices", "absent without cached prices"),
    "session_return": _c("Session return", "close(last) / close(prior session) "
                         "- 1. Sessions are cache rows: weekends and holidays "
                         "never add a step.", "%", "1 session", "prices",
                         "absent under 2 cached bars"),
    "gap_pct": _c("Session gap", "open(last) / close(prior session) - 1 — the "
                  "opening gap only.", "%", "1 session", "prices",
                  "absent without an open and prior close"),
    "volume_ratio": _c("Volume vs baseline", "volume(last) / mean volume of the "
                       "20 completed sessions before it. Full baseline or "
                       "nothing — never silently short.", "x", "20 sessions "
                       "before last", "prices",
                       "incomplete until 20 sessions precede the last"),
    "vol_window_sessions": _c("Vol window sessions", "Completed session returns "
                              "available in the trailing window before the last "
                              "session (max 20). Reads alongside the "
                              "significance state.", "sessions",
                              "trailing window", "prices",
                              "absent under 2 cached bars"),
    "z_score": _c("Move z-score", "|z| of the session return against its "
                  "trailing completed 20-session return window (mean/std, "
                  "sample std). The 'does it matter' measure: the move vs the "
                  "ticker's own recent volatility.", "σ",
                  "trailing 20 sessions", "prices",
                  "window incomplete until 20 returns precede the session"),
    "significance": _c("Significance", "Deterministic label from the z-score: "
                       "|z| >= 2 notable, >= 1 elevated, else within noise. "
                       "No LLM, no network.", "state", "trailing 20 sessions",
                       "prices", "None until the z-score exists"),
    "significance_state": _c("Significance state", "Declared state of the "
                             "significance computation: ok, window incomplete, "
                             "or degenerate window (zero-variance baseline).",
                             "state", "trailing 20 sessions", "prices",
                             "declares why the z-score is absent"),
    # --- earnings: upcoming + recent, session-aligned ---
    "next_print_date": _c("Next print date", "Earliest upcoming announcement "
                          "date as published by yfinance.", "date", "event",
                          "earnings", "absent when unscheduled"),
    "next_print_in_days": _c("Next print in days", "Calendar days to the next "
                             "announcement. Calendar, not sessions: a countdown "
                             "to a known date.", "days", "event", "earnings",
                             "absent when unscheduled"),
    "last_print_date": _c("Last print date", "Most recent announcement date as "
                          "published by yfinance. Source lacks a before-open/"
                          "after-close flag, so alignment is unverified.", "date",
                          "event", "earnings", "absent without a cached print"),
    "sessions_since_print": _c("Sessions since print", "Trading sessions elapsed "
                               "since the reaction session (first session on/"
                               "after the print date). Weekends and holidays do "
                               "not advance it.", "sessions", "event", "prices",
                               "absent without an aligned cached session"),
    "eps_surprise_pct": _c("EPS surprise %", "(actual - estimate) / |estimate| "
                           "x 100 as published. |estimate| < 0.05 renders NM.",
                           "%", "event", "earnings",
                           "NM when estimate near zero or absent"),
    "eps_surprise_nm": _c("Surprise NM", "True when the surprise is not "
                          "meaningful: |estimate| < 0.05 or absent.", "state",
                          "event", "earnings",
                          "None without a cached print"),
    "reaction_ret": _c("Reaction return", "close(reaction) / close(reaction-1) "
                       "- 1 — the full first-session move.", "%",
                       "reaction session", "prices",
                       "absent without an aligned prior close"),
    "print_gap_pct": _c("Print gap", "open(reaction) / close(reaction-1) - 1.",
                        "%", "reaction session", "prices",
                        "absent without an aligned prior close"),
    "print_rel_volume": _c("Print relative volume", "volume(reaction) / mean "
                           "volume of the 20 sessions ending reaction-1 "
                           "(reaction day excluded).", "x", "reaction session",
                           "prices", "absent without aligned history"),
    "post_drift_5d": _c("Post-print drift 5d", "close(t0+5 sessions) / "
                        "close(t0) - 1. Completed windows only; excludes the "
                        "reaction day.", "%", "5 sessions", "prices",
                        "incomplete until 5 sessions elapse"),
    "post_drift_10d": _c("Post-print drift 10d", "close(t0+10 sessions) / "
                         "close(t0) - 1. Completed windows only.", "%",
                         "10 sessions", "prices",
                         "incomplete until 10 sessions elapse"),
    "post_drift_20d": _c("Post-print drift 20d", "close(t0+20 sessions) / "
                         "close(t0) - 1. Completed windows only.", "%",
                         "20 sessions", "prices",
                         "incomplete until 20 sessions elapse"),
    # --- context: static, demoted below the deltas ---
    "sector": _c("Sector", "GICS sector from the cached fundamentals payload.",
                 "text", "fundamentals as-of", "fundamentals",
                 "unknown — payload missing"),
    "mkt_cap": _c("Market cap", "Shares outstanding x price as published in the "
                  "cached fundamentals payload.", "USD", "fundamentals as-of",
                  "fundamentals", "unknown — payload missing"),
    "fwd_pe": _c("Forward P/E", "Price / forward EPS as published. Non-positive "
                 "renders NM.", "x", "fundamentals as-of", "fundamentals",
                 "NM when <= 0 or payload missing"),
    "ps": _c("Price / Sales", "Market cap / trailing revenue as published. "
             "Non-positive renders NM.", "x", "fundamentals as-of",
             "fundamentals", "NM when <= 0 or payload missing"),
    "div_yield": _c("Dividend yield", "Trailing yield as published. Context "
                    "only — not a safety measure.", "%", "fundamentals as-of",
                    "fundamentals", "unknown — non-payer or missing"),
    "short_pct_float": _c("Short % of float", "Short interest as a fraction of "
                          "float at the SI settlement date. A dated snapshot, "
                          "not live.", "fraction of float", "SI settlement date",
                          "short", "unknown — feed absent"),
    "days_to_cover": _c("Days to cover", "Vendor-reported days to cover at the "
                        "SI settlement date.", "days", "SI settlement date",
                        "short", "unknown — feed absent"),
    "si_change_pct": _c("SI change (report/report)", "Shares short vs the prior "
                        "settlement, same units.", "%", "report-over-report",
                        "short", "unknown without both reports"),
}

METRIC_KEYS_BY_SECTION = {
    "price": ["session_date", "session_return", "gap_pct", "volume_ratio",
              "vol_window_sessions", "z_score", "significance",
              "significance_state"],
    "earnings": ["next_print_date", "next_print_in_days", "last_print_date",
                 "sessions_since_print", "eps_surprise_pct", "eps_surprise_nm",
                 "reaction_ret", "print_gap_pct", "print_rel_volume",
                 "post_drift_5d", "post_drift_10d", "post_drift_20d"],
    "context": ["sector", "mkt_cap", "fwd_pe", "ps", "div_yield",
                "short_pct_float", "days_to_cover", "si_change_pct"],
}

# Section-level plumbing: the declared presence state of the section's
# primary source, uniform across sections, not a metric.
NON_METRIC_SECTION_KEYS = frozenset({"state"})

CONTRACT_FIELDS = ("term", "definition", "unit", "window", "source",
                   "missing_state")


def contract(key: str) -> dict:
    return _CONTRACTS[key]


def incomplete_contracts() -> list[str]:
    """Contract entries missing any required field, as "<key>: empty <field>";
    section metrics whose source is not a receipt key, as
    "<key>: unknown source"."""
    problems = [f"{key}: empty {field}"
                for key, entry in _CONTRACTS.items()
                for field in CONTRACT_FIELDS
                if not str(entry.get(field) or "").strip()]
    for keys in METRIC_KEYS_BY_SECTION.values():
        problems.extend(f"{key}: unknown source"
                        for key in keys
                        if contract(key)["source"] not in SOURCE_KEYS)
    return problems


def uncontracted_section_keys(section: dict) -> list[str]:
    """Section keys with neither a contract entry nor a non-metric listing."""
    return sorted(k for k in section
                  if k not in _CONTRACTS and k not in NON_METRIC_SECTION_KEYS)
