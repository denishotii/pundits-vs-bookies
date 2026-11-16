"""Part 2 (Polymarket): tennis prices around the first set.

Question (from the TennisEdge draft): when a tennis favourite loses the first
set, does the Polymarket price fall further than history says it should?

Status: the two data calls below work. The analysis is TODO (part 2 role).

Things we learned while checking the API:
- Tennis events have tag_id 864. Each match event holds a "moneyline" market
  and often a "Set 1 winner" market. That market settles to 0 or 1 when the
  first set ends, which tells us when set 1 finished.
- The events API stops paging after about 2,100 results, so query week by week.
- Prices come from the CLOB API; fidelity=1 gives one price per minute.
- Many matches have almost no money traded; use those with volume >= 10,000.
"""

from __future__ import annotations

from . import config
from .http import PoliteSession

TENNIS_TAG_ID = 864
MIN_VOLUME = 10_000


def tennis_events(session: PoliteSession, start: str, end: str, closed: bool = True) -> list[dict]:
    """All tennis events starting between two dates ("2026-03-01"). Keep the window short."""
    events, offset = [], 0
    while True:
        params = {
            "tag_id": TENNIS_TAG_ID,
            "closed": str(closed).lower(),
            "start_date_min": start,
            "start_date_max": end,
            "limit": 100,
            "offset": offset,
        }
        page = session.get(f"{config.POLYMARKET_GAMMA}/events", params=params).json()
        events.extend(page)
        if len(page) < 100:
            return events
        offset += 100


def price_history(session: PoliteSession, token_id: str, start_ts: int, end_ts: int, fidelity: int = 1) -> list[dict]:
    """Price points [{"t": unix time, "p": price}] for one outcome token."""
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity}
    resp = session.get(f"{config.POLYMARKET_CLOB}/prices-history", params=params)
    return resp.json().get("history", [])
