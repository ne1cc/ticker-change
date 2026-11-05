"""Mobile layout ratchet: real browsers, phone viewports, seeded offline data.

Intent: every route must satisfy four rules (overflow, tap size, text size,
sticky header). tests/mobile_known.py lists tolerated violations; a NEW
violation fails, and so does a stale entry whose violation is gone, so the
list can only shrink.
Contract: skips (never fails) when Playwright, its browsers or the CDNs
(Tailwind/ECharts) are unavailable.
Limits: data is synthetic; real-data tables/charts are covered by the manual
check in the PR test plan, not here.
"""
import os
import tempfile
import unittest.mock as mock

import pytest

import db
import news
from tests.mobile_known import KNOWN
from tests.mobile_support import AUDIT_JS, cdn_reachable, seed_universe, serve

sync_api = pytest.importorskip("playwright.sync_api")

T1 = ["/", "/brief/AAA", "/stock?ticker=AAA", "/news?ticker=AAA", "/glossary"]
T2 = ["/analytics?ticker=AAA", "/positioning?ticker=AAA", "/heatmap", "/radar",
      "/strategies?ticker=AAA", "/ai-summary?ticker=AAA", "/methodology", "/settings"]
T3 = ["/options?ticker=AAA", "/live?ticker=AAA", "/signal-lab"]
ALL = T1 + T2 + T3
PHONES = {"android-360": {"width": 360, "height": 740},
          "iphone-390": {"width": 390, "height": 844}}
RULES = ("overflow", "tap", "font", "sticky")
# Excel is the default theme; light/dark are opted into via localStorage
# (same keys the base.html theme-init script reads).
THEME_JS = {"excel": "",
            "light": "localStorage.setItem('theme','light');"
                     "localStorage.setItem('theme_default_set_excel','true');",
            "dark": "localStorage.setItem('theme','dark');"
                    "localStorage.setItem('theme_default_set_excel','true');"}


_OBSERVED: set = set()  # (route, theme, rule) seen failing in any engine/viewport
_RAN: set = set()       # cases executed this session (for the stale-entry check)


def _news_feed():
    """Healthy offline feed so /news renders its article table (no API key needed)."""
    feed = news._result("AAA", "ok")
    feed.update(listing=dict(id="AAA|NYSE|us", symbol="AAA", exchange="NYSE",
                             country="us", name="AAA Corp"),
                articles=[dict(title="AAA reports a very long headline that must wrap on a phone "
                                     f"without pushing the page wider {i}",
                               source="publisher.example", url="https://publisher.example/s",
                               published_at="2026-10-12T11:00:00Z") for i in range(8)])
    return feed


def _cases():
    cases = []
    for route in ALL:
        cases += [("chromium", "android-360", "excel", route),
                  ("chromium", "iphone-390", "excel", route),
                  ("webkit", "iphone-390", "excel", route)]
    for route in T1:
        for theme in ("light", "dark"):
            cases += [("chromium", "iphone-390", theme, route),
                      ("webkit", "iphone-390", theme, route)]
    return cases


@pytest.fixture(scope="module")
def base_url():
    if not cdn_reachable():
        pytest.skip("Tailwind/ECharts CDNs unreachable; mobile layout suite needs them")
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    orig = db.DB_PATH
    db.DB_PATH = path
    db.init_db()
    prior_pw = os.environ.get("SETTINGS_PASSWORD")
    os.environ["SETTINGS_PASSWORD"] = "mobile-test"
    import app as app_module
    patches = [mock.patch.object(app_module, "_fetch_yfinance_with_retry", return_value=None),
               mock.patch.object(app_module, "_warm_brief_universe"),
               mock.patch("news.get_feed", return_value=_news_feed())]
    for p in patches:
        p.start()
    seed_universe(app_module)
    server, url = serve(app_module)
    yield url
    server.shutdown()
    for p in patches:
        p.stop()
    db.DB_PATH = orig
    os.remove(path)
    if prior_pw is None:
        os.environ.pop("SETTINGS_PASSWORD", None)
    else:
        os.environ["SETTINGS_PASSWORD"] = prior_pw


@pytest.fixture(scope="module")
def browsers():
    """One launched browser per engine for the whole module; skip if not installed."""
    launched = {}
    with sync_api.sync_playwright() as pw:
        def get(engine):
            if engine not in launched:
                try:
                    launched[engine] = getattr(pw, engine).launch()
                except Exception as exc:  # binary not installed
                    pytest.skip(f"{engine} not installed: {exc}")
            return launched[engine]
        yield get
        for b in launched.values():
            b.close()


def _phone_page(browser, vp, theme="excel"):
    ctx = browser.new_context(viewport=PHONES[vp], is_mobile=True, has_touch=True,
                              device_scale_factor=3,
                              http_credentials={"username": "admin", "password": "mobile-test"})
    ctx.add_init_script(THEME_JS[theme])
    return ctx, ctx.new_page()


@pytest.mark.parametrize("engine,vp,theme,route", _cases())
def test_route_meets_mobile_rules(browsers, base_url, engine, vp, theme, route):
    ctx, page = _phone_page(browsers(engine), vp, theme)
    try:
        resp = page.goto(base_url + route, wait_until="networkidle")
        assert resp.status == 200, f"{route} -> {resp.status}"
        page.wait_for_timeout(400)  # let theme transition + charts settle
        result = page.evaluate(AUDIT_JS)
    finally:
        ctx.close()

    failing = {r for r in RULES if result[r]}
    _OBSERVED.update((route, theme, r) for r in failing)
    _RAN.add((engine, vp, theme, route))
    new = {r for r in failing if (route, theme, r) not in KNOWN}
    detail = {r: (result[r][:8] if isinstance(result[r], list) else result[r]) for r in failing}
    assert not new, f"new violations on {route} [{engine}/{vp}/{theme}]: {detail}"


def test_known_has_no_stale_entries():
    """KNOWN may only list violations some phone case still shows. Runs last;
    skipped on partial runs (-k, single test) where absence proves nothing."""
    if _RAN != set(_cases()):
        pytest.skip("partial run; stale check needs every case")
    stale = sorted(set(KNOWN) - _OBSERVED)
    assert not stale, f"stale KNOWN entries (fixed! delete them): {stale}"


def test_body_is_not_a_scroll_container(browsers, base_url):
    """position:sticky inside <body> breaks if body is a scroll container."""
    ctx, page = _phone_page(browsers("chromium"), "iphone-390")
    try:
        page.goto(base_url + "/glossary", wait_until="domcontentloaded")
        overflow_x = page.evaluate("getComputedStyle(document.body).overflowX")
    finally:
        ctx.close()
    assert overflow_x == "clip", overflow_x


@pytest.mark.parametrize("route", ["/brief/AAA", "/analytics?ticker=AAA"])
def test_definition_tap_opens_popover_without_navigating(browsers, base_url, route):
    ctx, page = _phone_page(browsers("chromium"), "iphone-390")
    try:
        page.goto(base_url + route, wait_until="networkidle")
        toggle = page.locator("[data-tip-toggle]:visible").first
        assert toggle.count() == 1, f"no definition toggle on {route}"
        url = page.url
        toggle.scroll_into_view_if_needed()
        toggle.tap()
        pop = page.locator("[data-tip][data-open] > [role=tooltip]").first
        assert pop.is_visible()
        box = pop.bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= PHONES["iphone-390"]["width"], box
        assert page.url == url  # did not navigate to /glossary
        page.mouse.click(5, 300)  # tap away (left gutter)
        assert page.locator("[data-tip][data-open]").count() == 0
    finally:
        ctx.close()


def test_sticky_first_column_stays_put_and_scroll_fade_appears(browsers, base_url):
    ctx, page = _phone_page(browsers("chromium"), "iphone-390")
    try:
        page.goto(base_url + "/glossary", wait_until="networkidle")
        page.evaluate("""() => {
          const w = document.createElement('div');
          w.className = 'overflow-x-auto sticky-first'; w.id = 'probe';
          w.innerHTML = '<table><tr><td>A</td>' + '<td>wide cell</td>'.repeat(40) + '</tr></table>';
          document.body.prepend(w);
        }""")
        page.wait_for_timeout(500)  # mobile.js rescans on DOM mutation after 200ms
        assert "has-more" in page.get_attribute("#probe", "class")
        page.evaluate("document.getElementById('probe').scrollLeft = 300")
        left = page.evaluate("document.querySelector('#probe td:first-child').getBoundingClientRect().left")
    finally:
        ctx.close()
    assert abs(left) < 2, left
