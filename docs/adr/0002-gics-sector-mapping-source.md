# ADR 0002: S&P 500 GICS Sector & Sub-Industry Taxonomy Mapping Source

## Status
Accepted

## Context
Cross-sectional quant comparisons, peer percentile rankings, and sector attribution require an objective, institutional sector taxonomy.
Previously, ticker categorization relied on Yahoo Finance's `info['sector']`, which is a proprietary taxonomy defined by Yahoo/Refinitiv rather than standard GICS (Global Industry Classification Standard). Differences between Yahoo's categories and standard S&P GICS create classification drift and inconsistent sector groupings.

To compute valid peer percentiles vs. a constituent's actual GICS sector and sub-industry, the application requires an authoritative mapping of S&P 500 constituents to their official GICS Sector and GICS Sub-Industry.

## Decision
1. **Source:** We source the S&P 500 constituents table from the curated, automated public repository maintained by `frictionlessdata/datasets` (`https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv`).
2. **Licensing:** The underlying dataset is dedicated to the public domain under the **Open Data Commons Public Domain Dedication and License (PDDL)** and **Creative Commons CC0 1.0 Universal**. It permits free commercial and non-commercial reproduction, distribution, and computation without proprietary API license lock-in.
3. **Caching and Refresh Cadence:**
   - The S&P 500 index reconstitutes periodically throughout the year. The mapping is cached in SQLite (`api_cache` table) with a 7-day (168-hour) TTL under provider `gics_taxonomy`.
   - A bundled fallback snapshot fixture is included to guarantee resilient, deterministic offline operation and fast test suite execution without external network dependency.
4. **Symbol Normalization:**
   - Dual-class share notation is normalized to standard yfinance hyphenated convention (`.` -> `-`, e.g. `BRK.B` -> `BRK-B`, `BF.B` -> `BF-B`), matching the existing convention in `sp500.py`.
5. **Degradation Policy:**
   - If a ticker is not in the S&P 500 mapping, `sector_of(ticker)` and `sub_industry_of(ticker)` return `None`.
   - In cross-sectional percentile calculations, an unmapped sector degrades cleanly to universe-only percentiles, with no fabricated sector assignments.

## Consequences
- Every S&P 500 constituent maps reliably to one of the 11 standard GICS sectors and its specific GICS sub-industry.
- Sector peer percentiles operate on verified, standardized peer groups.
- Zero external API costs or proprietary licensing encumbrances.
