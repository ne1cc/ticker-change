"""Executable checks for the inline <script> blocks in templates/base.html.

Two bugs shipped to production in one commit and neither was caught:

  SyntaxError: Identifier 'path' has already been declared
  ReferenceError: ticker is not defined

The first stops the entire script from parsing, so the topbar stops appending
?ticker= and every pillar link drops the active ticker. Flask still returns
HTTP 200 for all of it, and `test_navigation.py` only asserts status codes --
which is exactly why both slipped through.

These tests parse and then actually run the navigation block, so a dead script
fails the suite instead of the page.
"""
import json
import pathlib
import re
import shutil
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_HTML = REPO_ROOT / "templates" / "base.html"
NODE = shutil.which("node")

# Inline blocks only -- <script src=...> pulls in Tailwind/Plotly from a CDN.
INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def inline_scripts() -> list:
    """Inline script bodies from base.html, with Jinja expressions neutralised."""
    out = []
    for m in INLINE_SCRIPT.finditer(BASE_HTML.read_text()):
        body = m.group(1)
        body = re.sub(r"\{\{.*?\}\}", '"JINJA"', body, flags=re.S)
        body = re.sub(r"\{%.*?%\}", "", body, flags=re.S)
        out.append(body)
    return out


def nav_script() -> str:
    """The block that owns pillar state and ticker propagation."""
    for body in inline_scripts():
        if "PILLARS" in body and "ROUTE_MAP" in body:
            return body
    raise AssertionError("navigation block not found in base.html")


# A DOM thin enough to stay maintainable, real enough that the block runs its
# whole ticker path rather than short-circuiting on null guards. `replace` is
# recorded rather than followed, so the rebinding redirect is observable.
DOM_STUB = """
const makeEl = (id) => ({
  id, href: '', className: '', innerText: '', value: '',
  classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
});
const els = {};
globalThis.document = { getElementById: (id) => (els[id] ||= makeEl(id)) };
const mkStore = (seed) => ({ store: {...seed},
  getItem(k){ return this.store[k] ?? null; },
  setItem(k,v){ this.store[k] = String(v); },
  removeItem(k){ delete this.store[k]; } });
globalThis.localStorage = mkStore({});
globalThis.sessionStorage = mkStore(SEED);
const redirects = [];
globalThis.window = {
  location: { pathname: PATHNAME, search: SEARCH, href: '',
              replace(u){ redirects.push(u); } },
  addEventListener(){},
};
const report = () => console.log(JSON.stringify({
  redirects,
  session: sessionStorage.store,
  brandHref: els['brand-link'] ? els['brand-link'].href : null,
  analyticsHref: els['nav-analytics'] ? els['nav-analytics'].href : null,
}));
"""


@unittest.skipUnless(NODE, "node is required to parse/run the inline scripts")
class TestBaseInlineScripts(unittest.TestCase):

    def test_every_inline_script_parses(self):
        """A SyntaxError anywhere kills the whole block, silently, at HTTP 200."""
        failures = []
        for i, body in enumerate(inline_scripts()):
            proc = subprocess.run(
                [NODE, "--check", "-"], input=body,
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                first = next(
                    (ln for ln in proc.stderr.splitlines() if "Error" in ln),
                    proc.stderr.strip().splitlines()[:1] or ["unknown"],
                )
                failures.append(f"block {i}: {first if isinstance(first, str) else first[0]}")
        self.assertEqual([], failures, "\n".join(failures))

    def _run_nav(self, pathname: str, search: str, session=None):
        stub = (DOM_STUB
                .replace("PATHNAME", json.dumps(pathname))
                .replace("SEARCH", json.dumps(search))
                .replace("SEED", json.dumps(session or {})))
        return subprocess.run(
            [NODE, "--input-type=module", "-e",
             stub + "\n" + nav_script() + "\nreport();"],
            capture_output=True, text=True,
        )

    def _state(self, pathname: str, search: str, session=None):
        proc = self._run_nav(pathname, search, session)
        self.assertEqual(0, proc.returncode, proc.stderr.strip()[:600])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_nav_runs_with_a_ticker(self):
        """The path that ReferenceError'd: every consumer of `ticker` executes."""
        proc = self._run_nav("/analytics", "?ticker=MU")
        self.assertEqual(0, proc.returncode, proc.stderr.strip()[:600])

    def test_nav_runs_without_a_ticker(self):
        proc = self._run_nav("/analytics", "")
        self.assertEqual(0, proc.returncode, proc.stderr.strip()[:600])

    def test_nav_runs_on_the_homepage(self):
        """Root clears saved state and returns early."""
        proc = self._run_nav("/", "")
        self.assertEqual(0, proc.returncode, proc.stderr.strip()[:600])

    def test_nav_runs_on_the_options_chain_deep_link(self):
        """/live?tab=greeks is the one child route carrying its own query string."""
        proc = self._run_nav("/live", "?ticker=MU&tab=greeks")
        self.assertEqual(0, proc.returncode, proc.stderr.strip()[:600])

    def test_pillar_links_carry_the_active_ticker(self):
        """The user-visible symptom: nav links losing ?ticker= when this breaks."""
        st = self._state("/analytics", "?ticker=MU")
        self.assertIn("ticker=MU", st["analyticsHref"])


class TestSessionScopedTickerBinding(unittest.TestCase):
    """Binding must survive moving between views, but not the browser session."""

    _state = TestBaseInlineScripts._state
    _run_nav = TestBaseInlineScripts._run_nav

    def test_url_ticker_is_remembered_for_the_session(self):
        st = self._state("/analytics", "?ticker=MU")
        self.assertEqual("MU", st["session"].get("active_ticker"))
        self.assertEqual([], st["redirects"])

    def test_bare_route_rebinds_to_the_session_ticker(self):
        """Landing on /analytics mid-session keeps the subject you were on."""
        st = self._state("/analytics", "", session={"active_ticker": "MU"})
        self.assertEqual(["/analytics?ticker=MU"], st["redirects"])

    def test_bare_route_with_no_session_shows_the_empty_state(self):
        """A fresh visitor gets nothing resurrected."""
        st = self._state("/analytics", "")
        self.assertEqual([], st["redirects"])

    def test_rebinding_preserves_an_existing_query_string(self):
        """The options-chain deep link must keep tab=greeks through the rebind."""
        st = self._state("/live", "?tab=greeks", session={"active_ticker": "MU"})
        self.assertEqual(["/live?tab=greeks&ticker=MU"], st["redirects"])

    def test_utility_routes_never_rebind(self):
        st = self._state("/glossary", "", session={"active_ticker": "MU"})
        self.assertEqual([], st["redirects"])

    def test_homepage_clears_the_session_ticker(self):
        """'/' is the reset, and the brand logo is how it is reached."""
        st = self._state("/", "", session={"active_ticker": "MU"})
        self.assertNotIn("active_ticker", st["session"])
        self.assertEqual([], st["redirects"])

    def test_brand_link_is_not_given_a_ticker(self):
        """The logo must keep its '/' template href, or the reset is unreachable.

        None means the script never even looked the element up, which is the
        strongest form of "left alone".
        """
        st = self._state("/analytics", "?ticker=MU")
        self.assertIsNone(st["brandHref"], "nav script must not rewrite brand-link")

    def test_no_redirect_loop_once_the_ticker_is_present(self):
        st = self._state("/analytics", "?ticker=MU", session={"active_ticker": "NVDA"})
        self.assertEqual([], st["redirects"])
        self.assertEqual("MU", st["session"].get("active_ticker"),
                         "the URL wins and replaces the remembered ticker")


if __name__ == "__main__":
    unittest.main()
