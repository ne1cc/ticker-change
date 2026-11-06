"""Tolerated mobile-layout violations: (route, theme, rule) -> reason.

Recorded as the pre-port baseline (2026-11). Each phase deletes the entries
it fixes (a stale entry fails test_known_has_no_stale_entries); Phase 4 ends
with this empty.
"""
KNOWN: dict[tuple[str, str, str], str] = {}
