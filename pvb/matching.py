"""Match each preview to its match in the odds data.

Status: TODO (odds and matching role). The plan, tested in our pilot:

1. Names differ between sources ("Brighton & Hove Albion" vs "Brighton",
   "Jannik Sinner" vs "Sinner J."), so compare normalised names with fuzzy
   matching (rapidfuzz, as on the course slides).
2. The same fixture happens every season. In the pilot, 24 of 60 matches found
   by name alone were the wrong season. So a preview only matches a game that
   kicks off 0 to 10 days after the preview was published.
3. Keep a match only if exactly one game passes both checks; log the rest.
"""

from __future__ import annotations

import re
import unicodedata

from . import config

MATCHED_CSV = config.PROCESSED / "matched.csv"
MAX_DAYS_BEFORE_KICKOFF = 10


def normalise_name(name: str) -> str:
    """Lower case, no accents, no punctuation: "Atlético Madrid" -> "atletico madrid"."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def match_football():
    """TODO: previews.csv x football odds -> one row per matched preview."""
    raise NotImplementedError("match_football: see the plan in this module's docstring")


def match_tennis():
    """TODO: same idea; tennis-data writes players as "Sinner J."."""
    raise NotImplementedError("match_tennis: see the plan in this module's docstring")
