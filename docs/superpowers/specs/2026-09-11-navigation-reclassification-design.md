# Navigation Reclassification & Hierarchical Taxonomy Design Spec

**Date:** 2026-09-11  
**Status:** Approved  
**Topic:** Topbar and Sub-Navbar Reclassification & Redundancy Elimination  
**Branch:** `feat/nav-hierarchy`

---

## 1. Problem Statement & Motivation
Currently, both the top navigation bar and the secondary ticker sub-navbar duplicate the exact same 7 functional destinations:
1. Overview / Price Table (`/stock`)
2. Options / Options Terminal (`/options`)
3. Analytics / Risk Analytics (`/analytics`)
4. Positioning / Market Positioning (`/positioning`)
5. Live Microstructure (`/live`)
6. Strategies / Strategy Backtests (`/strategies` or `/momentum`)
7. AI Strategy / AI Strategy Report (`/ai-summary`)

This creates direct visual redundancy, wastes screen real estate, causes horizontal crowding on laptop screens, and fails to establish an intuitive information hierarchy.

---

## 2. Information Architecture & Four Trading Pillars

The application navigation is restructured into **4 top-level domain pillars** on the desktop topbar, folding all underlying tools into contextual sub-views:

```
[Top Nav]
ticker/change  |  Markets    Options    Analytics    Strategies  |  Glossary  API Docs  [Settings]  [Theme]
---------------------------------------------------------------------------------------------------------
[Sub-Nav (Contextual to Active Pillar)]
When Markets active:    [ MU ]  |  Price Table    Live Microstructure     |  [Search Ticker...]
When Options active:    [ MU ]  |  Options Terminal                       |  [Search Ticker...]
When Analytics active:  [ MU ]  |  Risk Analytics    Market Positioning   |  [Search Ticker...]
When Strategies active: [ MU ]  |  Strategy Backtests    AI Strategy Report |  [Search Ticker...]
```

### Detailed Pillar Taxonomy

| Pillar Name | Default Target Route | Child Sub-Views | Sub-View Route | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Markets** | `/stock?ticker=<T>` | `Price Table`<br>`Live Microstructure` | `/stock?ticker=<T>`<br>`/live?ticker=<T>` | Core equity pricing, performance metrics, and real-time tick/order book microstructure. |
| **Options** | `/options?ticker=<T>` | `Options Terminal` | `/options?ticker=<T>` | Institutional options analytics, GEX, IV surfaces, volatility cones, and chain data. |
| **Analytics** | `/analytics?ticker=<T>` | `Risk Analytics`<br>`Market Positioning` | `/analytics?ticker=<T>`<br>`/positioning?ticker=<T>` | Quantitative risk models (VaR, Monte Carlo, Beta) and 13F institutional/insider positioning. |
| **Strategies** | `/strategies?ticker=<T>` | `Strategy Backtests`<br>`AI Strategy Report` | `/strategies?ticker=<T>`<br>`/ai-summary?ticker=<T>` | Quantitative strategy backtests and deep LLM multi-factor strategy synthesis. |

### Global Utilities
- `Glossary` (`/glossary`)
- `API Docs` (`/api/docs`)
- `Settings` (`/settings`)
- `Theme Switcher` (cycles `light` → `dark` → `excel`)

---

## 3. Interaction & Routing Behavior

1. **Top Nav Navigation:**
   - Clicking a pillar in the topbar directly opens that pillar's default landing view (`/stock`, `/options`, `/analytics`, `/strategies`) while preserving the active `?ticker=` parameter.
   - The active pillar receives prominent visual styling (amber accent in dark/light mode; active chrome styling in Excel mode).
2. **Contextual Sub-Navbar:**
   - Renders only the child sub-views corresponding to the currently active pillar.
   - Highlights the currently active child view with solid accent fill (`bg-amber-500 text-zinc-950` in dark/light mode; white tab with top green border in Excel mode).
   - Shows active ticker badge (`[TICKER]`) and the quick ticker search form.
   - On utility pages (`/glossary`, `/api/docs`, `/settings`), if an active ticker exists, the sub-nav displays the ticker badge and search box without pillar child tabs, allowing fast return to market analysis.
3. **Mobile Drawer (`#mobile-nav-menu`):**
   - Restructured into the same 4 clear categories:
     - **Markets:** Price Table, Live Microstructure
     - **Options:** Options Terminal
     - **Analytics:** Risk Analytics, Market Positioning
     - **Strategies:** Strategy Backtests, AI Strategy Report
     - **Reference & Tools:** Glossary, API Docs, Settings
4. **Excel Mode Compatibility:**
   - Topbar remains the Excel title bar (green chrome `#217346` with white text).
   - Sub-navbar retains sheet-tab styling for only the active pillar's child tools, matching the spreadsheet workbook mental model.

---

## 5. Testing & Verification

1. **Unit & Integration Tests (`tests/test_navigation.py`):**
   - Verify all 4 pillar top-nav elements (`#nav-markets`, `#nav-options`, `#nav-analytics`, `#nav-strategies`) are present and route correctly.
   - Verify utility links (`#nav-glossary`, `#nav-apidocs`, `#nav-settings`) remain present and functional.
   - Verify that all child views are mapped in the sub-navbar structure.
   - Verify mobile drawer links match the new 4-pillar structure.
   - Verify ticker persistence (`?ticker=...`) is maintained across all pillar transitions and sub-nav searches.
2. **Regression Check:**
   - Full test suite passes cleanly with SQLite test isolation.
