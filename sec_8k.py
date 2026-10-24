"""SEC EDGAR Form 8-K Material Event Ingestion & Classification.

Parses unstructured & structured 8-K filings from SEC EDGAR:
- Item 1.01: Entry into Material Definitive Agreement
- Item 2.01: Completion of Acquisition or Disposition of Assets (M&A)
- Item 4.02: Non-Reliance on Previously Issued Financials (Restatements)
- Item 5.02: Departure / Election of Directors or Principal Officers
- Item 7.01 / 8.01: Regulation FD Disclosure & Other Events
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import providers
import db


ITEM_DEFINITIONS = {
    "1.01": "Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "2.01": "Completion of Acquisition/Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition (Earnings)",
    "2.03": "Creation of Direct Financial Obligation",
    "3.01": "Notice of Delisting / Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements (Restatement)",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Principal Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
}

# Event-study taxonomy: each known item maps to an event type, or to
# "routine" when the item is administrative and carries no market-moving
# information (shareholder votes, charter/bylaw housekeeping, catch-all
# "Other Events" absent a content classifier). Only items listed here reach
# event studies.
ITEM_EVENT_TYPES = {
    "1.01": "m&a",
    "1.02": "m&a",
    "2.01": "m&a",
    "2.02": "earnings",
    "2.03": "financing",
    "3.01": "listing",
    "3.02": "financing",
    "4.01": "auditor",
    "4.02": "restatement",
    "5.01": "control",
    "5.02": "management",
    "5.03": "routine",
    "5.07": "routine",
    "7.01": "reg_fd",
    "8.01": "routine",
}

# Mixed-item filings resolve to the first type in this order: highest
# market-information severity wins, so a 2.02+5.02 filing is "earnings"
# material and a 2.02+7.01 pair (release + slides) stays "earnings".
EVENT_TYPE_PRIORITY = [
    "restatement", "control", "listing", "m&a", "auditor",
    "management", "earnings", "financing", "reg_fd",
]


def resolve_event_type(items: List[str]) -> Optional[str]:
    """Taxonomy type for a filing's item set, or None when routine-only.

    Deterministic by construction: the informative item earliest in
    EVENT_TYPE_PRIORITY wins regardless of item order; a filing whose items
    are all routine (or unknown) resolves to None and is excluded from
    event studies.
    """
    types = {ITEM_EVENT_TYPES.get(it) for it in (items or [])}
    for t in EVENT_TYPE_PRIORITY:
        if t in types:
            return t
    return None


@dataclass
class MaterialEvent8K:
    accession_number: str
    cik: str
    ticker: str
    filing_date: str  # YYYY-MM-DD
    report_date: str  # YYYY-MM-DD (Date of period/event)
    items: List[str]  # e.g., ["1.01", "5.02"]
    item_descriptions: List[str]
    primary_doc_url: str
    description: str = ""
    is_high_impact: bool = False
    event_type: Optional[str] = None  # None = routine-only: excluded from studies
    after_close: bool = False  # accepted at/after 16:00 ET → next session reacts


def fetch_and_parse_8k_filings(ticker: str, limit: int = 20) -> List[MaterialEvent8K]:
    """Fetch recent Form 8-K submissions from SEC EDGAR for a given ticker."""
    cik_str = providers.sec_cik_for_ticker(ticker)
    if not cik_str:
        return []

    cache_key = f"sec_8k_{ticker.upper()}"
    cached = db.cache_get("sec_edgar", cache_key, ttl_hours=6.0)
    if cached is not None:
        return [MaterialEvent8K(**item) for item in cached]

    try:
        cik_int = int(cik_str)
    except ValueError:
        return []

    url = providers.SEC_SUBMISSIONS_URL.format(cik=cik_int)
    res = providers._get_json(url, headers=providers._sec_headers())
    if not res or "filings" not in res:
        return []

    recent = res.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    items_list = recent.get("items", [])
    descriptions = recent.get("primaryDocDescription", [])
    acceptances = recent.get("acceptanceDateTime", [])

    parsed_events: List[MaterialEvent8K] = []

    for i in range(min(len(forms), 500)):
        form_name = forms[i]
        if form_name in ("8-K", "8-K/A"):
            raw_items = items_list[i] if i < len(items_list) else ""
            acc_num = accessions[i].replace("-", "") if i < len(accessions) else ""
            prim_doc = primary_docs[i] if i < len(primary_docs) else ""
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_num}/{prim_doc}" if acc_num and prim_doc else ""

            # Extract distinct items, e.g. "1.01,5.02" or semicolon separated
            found_items = []
            if raw_items:
                tokens = re.split(r"[,;\s]+", str(raw_items))
                for tok in tokens:
                    tok = tok.strip()
                    if tok in ITEM_DEFINITIONS and tok not in found_items:
                        found_items.append(tok)

            item_descs = [f"Item {it}: {ITEM_DEFINITIONS.get(it, 'Special Event')}" for it in found_items]

            # High impact categories: 4.02 (restatements), 4.01 (auditor switch), 5.01 (control change), 2.01 (M&A)
            high_impact = any(it in ("4.01", "4.02", "5.01", "2.01", "3.01") for it in found_items)

            # EDGAR acceptanceDateTime carries an ET offset, so .hour is the
            # local filing time: at/after 16:00 means the market was already
            # closed and the first reaction session is the next trading day.
            after_close = False
            acceptance = acceptances[i] if i < len(acceptances) else ""
            if acceptance:
                try:
                    after_close = datetime.fromisoformat(acceptance).hour >= 16
                except ValueError:
                    pass

            event = MaterialEvent8K(
                accession_number=accessions[i] if i < len(accessions) else "",
                cik=str(cik_str),
                ticker=ticker.upper(),
                filing_date=filing_dates[i] if i < len(filing_dates) else "",
                report_date=report_dates[i] if i < len(report_dates) and report_dates[i] else filing_dates[i],
                items=found_items,
                item_descriptions=item_descs,
                primary_doc_url=doc_url,
                description=descriptions[i] if i < len(descriptions) else "Form 8-K Current Report",
                is_high_impact=high_impact,
                event_type=resolve_event_type(found_items),
                after_close=after_close,
            )
            parsed_events.append(event)
            if len(parsed_events) >= limit:
                break

    # Cache formatted records
    db.cache_set("sec_edgar", cache_key, [e.__dict__ for e in parsed_events])
    return parsed_events
