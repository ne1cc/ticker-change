# Navigation Reclassification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclassify topbar navigation into 4 core trading pillars (Markets, Options, Analytics, Strategies) with a dynamic contextual sub-navbar, eliminating visual redundancy across desktop and mobile.

**Architecture:** The desktop topbar displays 4 high-level domain pillars (`Markets`, `Options`, `Analytics`, `Strategies`) plus global utilities. The secondary ticker sub-navbar dynamically renders only the child sub-views for the currently active pillar (e.g. Markets: Price Table & Live Microstructure; Analytics: Risk Analytics & Market Positioning), preserving the active ticker across transitions. The mobile menu mirrors this exact 4-pillar hierarchy, and Excel mode renders the sub-views as spreadsheet workbook sheet tabs.

**Tech Stack:** Python 3.9+, Flask, Jinja2, Tailwind CSS, JavaScript (Vanilla ES6), pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-navigation-reclassification-design.md`

## Global Constraints

- Topbar desktop navigation MUST contain exactly 4 primary trading pillars: `Markets` (`#nav-markets`), `Options` (`#nav-options`), `Analytics` (`#nav-analytics`), and `Strategies` (`#nav-strategies`).
- Topbar desktop navigation MUST retain global utilities: `Glossary` (`#nav-glossary`), `API Docs` (`#nav-apidocs`), `Settings` (`#nav-settings`), and the theme toggle (`#theme-toggle`).
- The sub-navbar (`#ticker-sub-nav`) MUST display active ticker context (`#sub-nav-ticker`), sub-nav search (`#sub-nav-search-input`), and only the child views of the active pillar.
- Active child view IDs in the sub-navbar MUST be:
  - Under Markets: `#sub-link-stock` (Price Table) and `#sub-link-live` (Live Microstructure).
  - Under Options: `#sub-link-options` (Options Terminal).
  - Under Analytics: `#sub-link-analytics` (Risk Analytics) and `#sub-link-positioning` (Market Positioning).
  - Under Strategies: `#sub-link-strategies` (Strategy Backtests) and `#sub-link-ai-summary` (AI Strategy Report).
- Mobile navigation menu (`#mobile-nav-menu`) MUST group links under 4 pillar section headers plus Reference & Tools.
- Ticker query parameter (`?ticker=<SYMBOL>`) MUST persist across top-level pillar links, sub-navbar links, and sub-nav search submissions.
- Excel mode (`html.excel`) styling MUST apply seamlessly without broken borders, misaligned tabs, or missing contrast.
- All tests in `tests/test_navigation.py` MUST pass cleanly with 0 failures and no warnings.

---

### Task 1: Navigation Test Suite Update

**Files:**
- Modify: `tests/test_navigation.py`

**Interfaces:**
- Consumes: Existing Flask test client in `TestNavigation` class.
- Produces: Updated assertions for 4 topbar pillar IDs, contextual sub-navbar link IDs, and mobile drawer IDs.

- [ ] **Step 1: Write the updated navigation test assertions**

Update `tests/test_navigation.py` to assert the 4 new pillar IDs (`nav-markets`, `nav-options`, `nav-analytics`, `nav-strategies`), the contextual sub-link elements, and the updated mobile link IDs (`mobile-nav-markets`, etc.). Specifically, update `test_desktop_navigation_links_present`, `test_ticker_sub_navbar_links_present`, and `test_mobile_navigation_links_present`.

```python
    def test_desktop_navigation_links_present(self):
        """Verify that desktop navigation contains all 4 pillar links and utilities."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_ids = [
            'id="nav-markets"',
            'id="nav-options"',
            'id="nav-analytics"',
            'id="nav-strategies"',
            'id="nav-glossary"',
            'id="nav-apidocs"',
            'id="nav-settings"',
        ]
        for link_id in expected_ids:
            self.assertIn(
                link_id,
                html,
                f"Missing desktop navigation element: {link_id}",
            )

    def test_ticker_sub_navbar_links_present(self):
        """Verify that the ticker sub-navbar structure contains all child tool link anchors."""
        resp = self.client.get("/stock?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_sub_ids = [
            'id="sub-link-stock"',
            'id="sub-link-live"',
            'id="sub-link-options"',
            'id="sub-link-analytics"',
            'id="sub-link-positioning"',
            'id="sub-link-strategies"',
            'id="sub-link-ai-summary"',
        ]
        for sub_id in expected_sub_ids:
            self.assertIn(
                sub_id,
                html,
                f"Missing ticker sub-navbar element: {sub_id}",
            )

    def test_mobile_navigation_links_present(self):
        """Verify that mobile navigation drawer contains all core view links organized by pillar."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_mobile_ids = [
            'id="mobile-nav-stock"',
            'id="mobile-nav-live"',
            'id="mobile-nav-options"',
            'id="mobile-nav-analytics"',
            'id="mobile-nav-positioning"',
            'id="mobile-nav-strategies"',
            'id="mobile-nav-ai-summary"',
            'id="mobile-nav-glossary"',
            'id="mobile-nav-apidocs"',
            'id="mobile-nav-settings"',
        ]
        for m_id in expected_mobile_ids:
            self.assertIn(
                m_id,
                html,
                f"Missing mobile navigation element: {m_id}",
            )
```

- [ ] **Step 2: Run test to verify failure (RED)**

Run: `.venv/bin/pytest tests/test_navigation.py`
Expected: FAIL with missing desktop navigation element `id="nav-markets"`.

- [ ] **Step 3: Commit test changes**

```bash
git add tests/test_navigation.py
git commit -m "test(nav): update test assertions for 4-pillar navigation architecture"
```

---

### Task 2: Topbar & Mobile Drawer Markup Refactor

**Files:**
- Modify: `templates/base.html:227-302`

**Interfaces:**
- Consumes: Flask request context and URLs.
- Produces: Clean HTML structure for 4 topbar links (`#nav-markets`, `#nav-options`, `#nav-analytics`, `#nav-strategies`) and 4-pillar mobile drawer structure.

- [ ] **Step 1: Replace desktop navigation links with the 4 pillars**

In `templates/base.html`, replace lines 227-236:
```html
            <div class="hidden md:flex items-center gap-5 lg:gap-6">
              <a id="nav-markets" href="/stock" class="text-sm text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors font-medium border-b-2 border-transparent pb-0.5">Markets</a>
              <a id="nav-options" href="/options" class="text-sm text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors font-medium border-b-2 border-transparent pb-0.5">Options</a>
              <a id="nav-analytics" href="/analytics" class="text-sm text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors font-medium border-b-2 border-transparent pb-0.5">Analytics</a>
              <a id="nav-strategies" href="/strategies" class="text-sm text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors font-medium border-b-2 border-transparent pb-0.5">Strategies</a>
            </div>
```

- [ ] **Step 2: Restructure mobile navigation drawer into 4 pillars**

In `templates/base.html`, update lines 274-301 to:
```html
        <div id="mobile-nav-menu" class="hidden md:hidden border-t border-zinc-200 dark:border-zinc-800 py-3 space-y-3">
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 px-2 mb-1">Markets</div>
            <div class="grid grid-cols-2 gap-1.5">
              <a id="mobile-nav-stock" href="/stock" class="mobile-nav-link">Price Table</a>
              <a id="mobile-nav-live" href="/live" class="mobile-nav-link">Live Microstructure</a>
            </div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 px-2 mb-1">Options</div>
            <div class="grid grid-cols-1 gap-1.5">
              <a id="mobile-nav-options" href="/options" class="mobile-nav-link">Options Terminal</a>
            </div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 px-2 mb-1">Analytics</div>
            <div class="grid grid-cols-2 gap-1.5">
              <a id="mobile-nav-analytics" href="/analytics" class="mobile-nav-link">Risk Analytics</a>
              <a id="mobile-nav-positioning" href="/positioning" class="mobile-nav-link">Market Positioning</a>
            </div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 px-2 mb-1">Strategies</div>
            <div class="grid grid-cols-2 gap-1.5">
              <a id="mobile-nav-strategies" href="/strategies" class="mobile-nav-link">Strategy Backtests</a>
              <a id="mobile-nav-ai-summary" href="/ai-summary" class="mobile-nav-link">AI Strategy Report</a>
            </div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 px-2 mb-1">Reference &amp; Tools</div>
            <div class="grid grid-cols-2 gap-1.5">
              <a id="mobile-nav-glossary" href="/glossary" class="mobile-nav-link">Glossary</a>
              <a id="mobile-nav-apidocs" href="/api/docs" class="mobile-nav-link text-amber-600 dark:text-amber-400 font-semibold">API Docs</a>
              <a id="mobile-nav-settings" href="/settings" class="mobile-nav-link">Settings</a>
            </div>
          </div>
        </div>
```

- [ ] **Step 3: Update sub-navbar anchor markup in `templates/base.html`**

Update `#sub-link-momentum` to `#sub-link-strategies`:
```html
        <div id="sub-nav-links-container" class="mobile-horizontal-scroll order-3 w-full lg:order-none lg:w-auto flex items-center gap-1 sm:gap-2">
          <a id="sub-link-stock" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Price Table</a>
          <a id="sub-link-live" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Live Microstructure</a>
          <a id="sub-link-options" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Options Terminal</a>
          <a id="sub-link-analytics" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Risk Analytics</a>
          <a id="sub-link-positioning" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Market Positioning</a>
          <a id="sub-link-strategies" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">Strategy Backtests</a>
          <a id="sub-link-ai-summary" href="#" class="sub-nav-pill text-xs font-semibold px-3 py-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors">AI Strategy Report</a>
        </div>
```

- [ ] **Step 4: Commit markup changes**

```bash
git add templates/base.html
git commit -m "feat(nav): restructure topbar into 4 pillars and align mobile navigation"
```

---

### Task 3: Contextual Sub-Navbar & Navigation JavaScript Controller

**Files:**
- Modify: `templates/base.html:424-536`

**Interfaces:**
- Consumes: `window.location.pathname`, `window.location.search`, `localStorage.getItem('last_ticker')`.
- Produces: Dynamic display of active pillar child tabs, active styling, persistent `?ticker=` URL updates, and search handler.

- [ ] **Step 1: Implement pillar-and-subview JavaScript mapping**

In `templates/base.html`, refactor the navigation script to:
1. Define pillar hierarchy mapping:
```javascript
        const PILLARS = {
          markets:    { id: 'nav-markets', defaultRoute: '/stock', children: ['stock', 'live'] },
          options:    { id: 'nav-options', defaultRoute: '/options', children: ['options'] },
          analytics:  { id: 'nav-analytics', defaultRoute: '/analytics', children: ['analytics', 'positioning'] },
          strategies: { id: 'nav-strategies', defaultRoute: '/strategies', children: ['strategies', 'ai-summary'] }
        };

        const ROUTE_MAP = {
          '/':            { pillar: 'markets', sub: 'stock' },
          '/stock':       { pillar: 'markets', sub: 'stock' },
          '/live':        { pillar: 'markets', sub: 'live' },
          '/options':     { pillar: 'options', sub: 'options' },
          '/analytics':   { pillar: 'analytics', sub: 'analytics' },
          '/positioning': { pillar: 'analytics', sub: 'positioning' },
          '/strategies':  { pillar: 'strategies', sub: 'strategies' },
          '/momentum':    { pillar: 'strategies', sub: 'strategies' },
          '/ai-summary':  { pillar: 'strategies', sub: 'ai-summary' },
          '/glossary':    { utility: 'glossary' },
          '/settings':    { utility: 'settings' }
        };
```
2. Determine active pillar and active sub-view from `ROUTE_MAP` (with `/api/docs` handling).
3. Apply active styling to the topbar pillar element (`border-amber-500 text-amber-600 dark:text-amber-400 font-semibold`).
4. Read active ticker from query param or `localStorage.last_ticker`.
5. Update core links (`#brand-link`, `#nav-markets`, `#nav-options`, `#nav-analytics`, `#nav-strategies`, mobile links) with `?ticker=` parameter.
6. When displaying `#ticker-sub-nav`:
   - Show only child links matching `PILLARS[activePillar].children` (hide others by adding `hidden` class).
   - For active child view, apply solid active styling (`bg-amber-500 text-zinc-950 font-bold`).
   - If on utility route and ticker exists, display `#ticker-sub-nav` with ticker badge and search bar, hiding pillar-specific sub-links.
7. Update `handleSubNavSearch(e)` to preserve destination if current path is a core route, or default to `/stock?ticker=<T>`.

- [ ] **Step 2: Run navigation test suite**

Run: `.venv/bin/pytest tests/test_navigation.py -v`
Expected: PASS (all 5 tests pass).

- [ ] **Step 3: Commit JavaScript controller changes**

```bash
git add templates/base.html
git commit -m "feat(nav): implement dynamic contextual sub-navbar and pillar state controller"
```

---

### Task 4: Excel Mode Styling Alignment & Full Verification

**Files:**
- Modify: `static/excel-mode.css`
- Modify: `tests/test_navigation.py`

**Interfaces:**
- Consumes: CSS classes and DOM structure from Tasks 2 & 3.
- Produces: Pixel-perfect Excel mode styling for the 4 pillars and sheet tabs, plus regression test coverage.

- [ ] **Step 1: Audit and align Excel mode CSS rules**

In `static/excel-mode.css`:
- Ensure `html.excel nav a[id^="nav-"]` carries clean Excel chrome styling (`color: #ffffff !important;` and hover states).
- Ensure active topbar pillar displays clean Excel accent styling.
- Ensure sub-navbar sheet tabs (`html.excel #ticker-sub-nav a[id^="sub-link"]`) continue to render as white sheet tabs with `#217346` top border when active, with hidden sub-links properly not taking up space (`display: none !important;` when carrying `hidden`).

- [ ] **Step 2: Add comprehensive route and sub-nav visibility tests**

Add automated tests in `tests/test_navigation.py`:
- Test that visiting `/analytics?ticker=AAPL` includes `id="nav-analytics"` with active indicator.
- Test that visiting `/stock?ticker=AAPL` renders `sub-link-stock` and `sub-link-live`.
- Test that `/strategies?ticker=AAPL` and `/momentum?ticker=AAPL` both route properly and retain query params.

- [ ] **Step 3: Run full pytest suite**

Run: `.venv/bin/pytest tests/test_navigation.py -v`
Expected: All tests pass cleanly.

- [ ] **Step 4: Commit final styling and test refinements**

```bash
git add static/excel-mode.css tests/test_navigation.py
git commit -m "feat(nav): polish excel mode sheet tabs and complete test suite"
```
