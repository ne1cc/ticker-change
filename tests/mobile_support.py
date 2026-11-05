"""Harness for tests/test_mobile_layout.py: offline seeded data, a live
werkzeug server on a free port, a CDN reachability probe, and the in-page
audit script.

Intent: let a real browser render real pages without touching the network
for data (Tailwind/ECharts still come from their CDNs -- see cdn_reachable).
"""
import threading
import urllib.error
import urllib.request

import pandas as pd
from werkzeug.serving import make_server

import db

TICKERS = ("AAA", "BBB")


def _store(sym, closes, end="2026-10-12"):
    idx = pd.bdate_range(end=end, periods=len(closes))
    db.store_prices(sym, pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": [1_000_000] * len(closes)}, index=idx))


def seed_universe(app_module):
    """260 bars per ticker + fundamentals/earnings caches + a published brief."""
    for n, sym in enumerate(TICKERS):
        closes = [100.0 + ((i * 7 + n) % 11) for i in range(259)] + [105.0 + n]
        _store(sym, closes)
        db.cache_set("yfinance", f"fundamentals:{sym}",
                     {"Sector": "Technology", "Market Cap": 1e9,
                      "Forward P/E": 10.0, "Price / Sales": 2.0})
        db.cache_set("yfinance", f"earnings:{sym}", [
            {"date": "2026-09-08", "eps_estimate": 1.0, "eps_actual": 1.2,
             "surprise_pct": 20.0, "is_upcoming": False}])
        app_module.publish_brief(sym)


def serve(app_module):
    """Start the Flask app on 127.0.0.1:<free port> in a daemon thread."""
    server = make_server("127.0.0.1", 0, app_module.app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def cdn_reachable():
    """Tailwind and ECharts load from CDNs; without them layout is meaningless."""
    try:
        urllib.request.urlopen("https://cdn.tailwindcss.com", timeout=5).close()
        return True
    except urllib.error.HTTPError:
        return True  # CDN answered (it 403s Python's default User-Agent); browsers get through
    except Exception:
        return False


# Returns {overflow, tap, font, sticky}; each list holds short element
# descriptors, empty == pass. Run via page.evaluate(AUDIT_JS).
AUDIT_JS = r"""
() => {
  const vw = document.documentElement.clientWidth;
  const out = {overflow: [], tap: [], font: [], sticky: null};
  const desc = el => el.tagName.toLowerCase() +
    (el.id ? '#' + el.id : '') +
    (typeof el.className === 'string' && el.className
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '');
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' &&
           s.display !== 'none' && s.opacity !== '0';
  };
  const scroller = el => {
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      const o = getComputedStyle(p).overflowX;
      if (['auto', 'scroll', 'hidden', 'clip'].includes(o)) return p;
    }
    return null;
  };

  // 1. overflow: element sticks out of the viewport and nothing legitimately scrolls it
  for (const el of document.querySelectorAll('body *')) {
    if (el.closest('svg, canvas, [data-audit-skip]') || !visible(el)) continue;
    if (getComputedStyle(el).position === 'fixed') continue;
    const r = el.getBoundingClientRect();
    if (r.right <= vw + 1 && r.left >= -1) continue;
    const sc = scroller(el);
    if (sc) {
      const sr = sc.getBoundingClientRect();
      if (sr.left >= -1 && sr.right <= vw + 1) continue;
    }
    out.overflow.push(desc(el) + ' right=' + Math.round(r.right));
  }

  // 2. tap targets
  const tapSel = 'button, input:not([type=hidden]):not([type=checkbox]):not([type=radio]), ' +
    'select, summary, [role=button], nav a, a.sub-nav-pill, .mobile-nav-link';
  for (const el of document.querySelectorAll(tapSel)) {
    if (el.closest('[data-audit-skip], [data-tip-toggle]') || !visible(el)) continue;
    const r = el.getBoundingClientRect();
    const iconOnly = !(el.textContent || '').trim() && el.tagName !== 'INPUT';
    if (r.height < 43.5 || (iconOnly && r.width < 43.5))
      out.tap.push(desc(el) + ' ' + Math.round(r.width) + 'x' + Math.round(r.height));
  }

  // 3. text floor
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  while (walker.nextNode()) {
    const n = walker.currentNode, el = n.parentElement;
    if (!n.textContent.trim() || seen.has(el)) continue;
    seen.add(el);
    if (el.closest('script, style, svg, canvas, sup, sub, option, [data-audit-skip]') || !visible(el)) continue;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 11.95) out.font.push(desc(el) + ' ' + fs + 'px');
  }

  // 4. sticky header (only meaningful on tall pages)
  const nav = document.querySelector('body > nav');
  if (nav && document.documentElement.scrollHeight > 1200) {
    window.scrollTo(0, 600);
    const top = nav.getBoundingClientRect().top;
    const sub = document.getElementById('ticker-sub-nav');
    const subH = sub && !sub.classList.contains('hidden') ? sub.getBoundingClientRect().height : 0;
    const budget = window.innerHeight * 0.25;
    if (Math.abs(top) > 1) out.sticky = 'nav not pinned (top=' + Math.round(top) + ')';
    else if (nav.getBoundingClientRect().height + subH > budget)
      out.sticky = 'header ' + Math.round(nav.getBoundingClientRect().height + subH) + 'px > ' + Math.round(budget) + 'px budget';
    window.scrollTo(0, 0);
  }
  return out;
}
"""
