"""Tolerated mobile-layout violations: (route, theme, rule) -> reason.

Recorded as the pre-port baseline (2026-11). Each phase deletes the entries
it fixes (a stale entry fails test_known_has_no_stale_entries); Phase 4 ends
with this empty.
"""
KNOWN: dict[tuple[str, str, str], str] = {
    ('/ai-summary?ticker=AAA', 'excel', 'tap'): "baseline 2026-11 (phase 3)",
    ('/analytics?ticker=AAA', 'excel', 'font'): "baseline 2026-11 (phase 3)",
    ('/brief/AAA', 'dark', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/brief/AAA', 'excel', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/brief/AAA', 'light', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/heatmap', 'excel', 'overflow'): "baseline 2026-11 (phase 3)",
    ('/options?ticker=AAA', 'excel', 'font'): "baseline 2026-11 (phase 4)",
    ('/options?ticker=AAA', 'excel', 'tap'): "baseline 2026-11 (phase 4)",
    ('/stock?ticker=AAA', 'dark', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/stock?ticker=AAA', 'excel', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/stock?ticker=AAA', 'light', 'overflow'): "baseline 2026-11 (phase 2)",
    ('/strategies?ticker=AAA', 'excel', 'tap'): "baseline 2026-11 (phase 3)",
}
