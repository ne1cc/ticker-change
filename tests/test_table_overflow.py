"""Wide tables must be able to scroll rather than overflow their card.

The backtest stat table on /strategies had eight columns inside a 7/12-width
card and no scroll container, so the Calmar column was drawn past the card's
right border instead of scrolling. Every other wide table in the project
already wraps in overflow-x-auto / overflow-auto; this one was the omission.

Nothing rendered wrong server-side and the page returned HTTP 200, so no
existing test could see it.
"""
import pathlib
import re
import unittest

from bs4 import BeautifulSoup


TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "templates"

# Below this, a table fits its card at any realistic width and needs no
# container. Measured against the project: every table at or above it is
# wrapped, every unwrapped one is under it.
WIDE = 6

SCROLL_CLASSES = {"overflow-x-auto", "overflow-auto", "overflow-scroll"}

# templates/_excel.html's xl-grid is the one wide table that handles overflow in
# CSS instead of a utility class: static/excel-mode.css gives html.excel
# .xl-sheet `overflow-x: auto` with a min-width under the 640px breakpoint.
CSS_HANDLED_ANCESTOR = "xl-sheet"


def _colspan(cell) -> int:
    raw = str(cell.get("colspan", 1))
    return int(raw) if raw.isdigit() else 1  # colspan can be a Jinja expression


def _widest_row(table) -> int:
    return max(
        (sum(_colspan(c) for c in tr.find_all(["th", "td"], recursive=False))
         for tr in table.find_all("tr")),
        default=0,
    )


def _scrolls(table) -> bool:
    for parent in table.parents:
        if not hasattr(parent, "get"):
            continue
        classes = set(parent.get("class") or [])
        if classes & SCROLL_CLASSES or CSS_HANDLED_ANCESTOR in classes:
            return True
    return False


class TestWideTablesCanScroll(unittest.TestCase):
    def test_every_wide_table_has_a_scroll_container(self):
        offenders = []
        checked = 0
        for path in sorted(TEMPLATES.glob("*.html")):
            # Jinja comments can contain example markup; drop them first.
            markup = re.sub(r"\{#.*?#\}", "", path.read_text(), flags=re.S)
            for table in BeautifulSoup(markup, "html.parser").find_all("table"):
                cols = _widest_row(table)
                if cols < WIDE:
                    continue
                checked += 1
                if not _scrolls(table):
                    offenders.append(
                        f"{path.name}: {cols}-column table has no "
                        f"overflow-x-auto ancestor, so it will overflow its card"
                    )

        self.assertGreater(checked, 0, "no wide tables found - has the scan broken?")
        self.assertEqual([], offenders, "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
