# ticker-change

A Flask equity-research portal that answers one question per ticker — "what
changed today, and does it matter?" — via nightly-published daily briefs with a
source-and-as-of receipt on every number.

## Language

**Brief**:
The published daily artifact for one ticker: last session's deltas plus
significance, receipts on every number, static profile as context only.
_Avoid_: report, summary, dashboard entry

**Receipt**:
The source-and-as-of annotation attached to every number in a brief.
_Avoid_: provenance stamp, attribution, tooltip

**Universe**:
The tiered set of tickers a nightly publication pass covers — watchlist
full-depth, S&P 500 deltas.
_Avoid_: watchlist (that is a tier), coverage list

**Publication**:
One atomic batch of built briefs: payload written, then pointer; last-2
retained; last-good served on failure.
_Avoid_: deploy, release, sync

**Front door**:
The root view: notable deltas across the published universe plus search,
served from publications only.
_Avoid_: home page, landing page

**Evidence page**:
One of the pre-existing deep pages (analytics, options, positioning, radar,
strategies, live, stock), kept as drill-down material behind brief claims.
_Avoid_: subpage, detail view

**Declared missing-state**:
The rendered contract state when a source is absent, failed, or stale — an
honest gap, never a blank or a crash.
_Avoid_: error, empty, fallback
