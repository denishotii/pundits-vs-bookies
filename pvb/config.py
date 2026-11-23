"""Project-wide settings: folders, data sources and polite-scraping defaults.

Everything that might change between team members (delay, contact) can be set
with an environment variable, so nobody has to edit this file.
"""

import os
from pathlib import Path

# Folders -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("PVB_DATA", ROOT / "data"))  # e.g. point it at a shared drive
RAW = DATA / "raw"              # exactly what we downloaded, never edited
INTERIM = DATA / "interim"      # parsed tables with article text (preview_urls.csv, previews.csv)
# The tables used in the analysis. They are on GitHub, so they always stay in the repo, even with PVB_DATA set.
PROCESSED = ROOT / "data" / "processed"

SPORTSMOLE_RAW = RAW / "sportsmole"
ODDS_RAW = RAW / "odds"
POLYMARKET_RAW = RAW / "polymarket"

# Sources -------------------------------------------------------------------

SPORTSMOLE = "https://www.sportsmole.co.uk"
FOOTBALL_DATA_CSV = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
TENNIS_DATA_INDEX = "https://www.tennis-data.co.uk/alldata.php"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"

# Scope (see docs/plan.md) ----------------------------------------------------

SPORTS = ("football", "tennis")

# football-data.co.uk league codes
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Seasons written the football-data way: "2122" = 2021/22
FOOTBALL_SEASONS = ("2122", "2223", "2324", "2425", "2526")
TENNIS_YEARS = (2024, 2025, 2026)
# The last match day we study. Our odds files were downloaded on 16 Nov 2026 and the 2026 ATP
# file ends on 14 Nov, so a fixed end keeps re-runs identical when tennis-data adds new matches.
STUDY_END = "2026-11-14"

# Polite scraping -------------------------------------------------------------

REQUEST_DELAY = float(os.getenv("PVB_DELAY", "2.0"))  # seconds between requests to one site
CONTACT = os.getenv("PVB_CONTACT", "student research project, KU Eichstaett-Ingolstadt")
USER_AGENT = f"pundits-vs-bookies/0.1 ({CONTACT})"
