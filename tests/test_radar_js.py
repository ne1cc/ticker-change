"""Syntax checks for the inline <script> blocks in templates/radar.html.

The radar client toolkit (drawer, presets, sort, filters) is ~300 lines of
inline JS across several blocks. Flask renders HTTP 200 no matter what the
script does, and a SyntaxError in one block silently kills every block that
follows it -- the exact failure mode test_base_nav_script.py guards against
for base.html. These tests parse each inline block with node --check so a
dead script fails the suite instead of the page.

Structure assertions for the same markup live in test_radar_page.py; here we
only prove the JavaScript parses. Jinja expressions are neutralised first
(like test_base_nav_script.py) so the check sees plain JS.
"""
import pathlib
import re
import shutil
import subprocess
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RADAR_HTML = REPO_ROOT / "templates" / "radar.html"
NODE = shutil.which("node")

INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def inline_scripts() -> list:
    """Inline script bodies from radar.html, with Jinja neutralised."""
    out = []
    for m in INLINE_SCRIPT.finditer(RADAR_HTML.read_text()):
        body = m.group(1)
        body = re.sub(r"\{\{.*?\}\}", '"JINJA"', body, flags=re.S)
        body = re.sub(r"\{%.*?%\}", "", body, flags=re.S)
        out.append(body)
    return out


@unittest.skipUnless(NODE, "node is required to parse the inline scripts")
class TestRadarJsSyntax(unittest.TestCase):

    def test_inline_scripts_parse(self):
        """A SyntaxError anywhere kills the whole block, silently, at HTTP 200."""
        self.assertTrue(inline_scripts(), "no inline scripts found in radar.html")
        for i, body in enumerate(inline_scripts()):
            with self.subTest(i=i):
                out = subprocess.run(
                    [NODE, "--check", "-"], input=body,
                    capture_output=True, text=True,
                )
                self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
