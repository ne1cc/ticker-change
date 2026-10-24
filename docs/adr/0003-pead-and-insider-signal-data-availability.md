# ADR 0003: Data Availability and Descoping for PEAD and Insider Net Buying Signals

## Status
Accepted

## Date
2026-10-24

## Context
Issue #137 (Epic 5: Signal Lab) requires evaluating Post-Earnings Announcement Drift (PEAD) and Insider Net Buying as cross-sectional alpha candidates alongside price-based signals.

A rigorous cross-sectional signal evaluation across the S&P 500 universe requires:
1. Point-in-time historical EPS surprises (consensus estimate, reported EPS, and exact announcement timestamp before market open vs. after market close) across all 500 constituents spanning multiple years.
2. Point-in-time SEC EDGAR Form 4 filings (transaction date, filing date, insider role, net purchase/sale shares) across all 500 constituents spanning multiple years.

## Findings from Data Audit
1. **yfinance Limitations:**
   - `.earnings_dates` only provides the last 4 to 8 quarters on-demand per ticker, frequently drops estimate values, and does not provide point-in-time cross-sectional synchronization across 500 constituents.
   - `insider_transactions` returns only the most recent ~15 transactions without historical depth or point-in-time filing date alignment across the universe.
   - Live querying across 500 tickers during panel generation triggers HTTP 429 rate limits.
2. **SEC EDGAR Form 4 Bulk Limitations:**
   - Ingesting multi-year Form 4 bulk data requires parsing quarterly XML archives (hundreds of gigabytes of unstructured filings) and entity CIK mapping, which is outside the lightweight embedded architecture of this application.

## Decision
In accordance with Issue #137 ("First check data availability... If unavailable at sufficient depth, record that as an ADR and descope... Insufficient history is declared, not extrapolated"):
- We formally descope multi-year cross-sectional backtesting of PEAD and Insider Net Buying from the active historical panel.
- Both signals are recorded in the Signal Lab catalog with status `"insufficient_history"`.
- The UI explicitly renders them as unvalidated with declared state `"insufficient_history"`, never extrapolating or simulating fictitious historical edge.
- Pure-function interfaces and models are provided so that when institutional point-in-time data sources are connected, they can be evaluated through the same Rank IC and decile spread pipeline.

## Consequences
- No artificial or look-ahead contaminated alpha numbers are presented for PEAD or insider trading.
- Institutional desk credibility is maintained: missing data is declared with transparency rather than hand-waved or extrapolated.
