"""Download and load the free odds files.

- football-data.co.uk: one CSV per league and season, with results, earlier
  odds (e.g. AvgH) and closing odds (e.g. AvgCH, since 2019/20).
- tennis-data.co.uk: one Excel file per tour and year, with results, set
  scores and closing odds (e.g. AvgW / AvgL).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config
from .http import PoliteSession

log = logging.getLogger(__name__)

FOOTBALL_DIR = config.ODDS_RAW / "football"
TENNIS_DIR = config.ODDS_RAW / "tennis"
# tennis-data links end in  /2025/2025.xlsx  (men, ATP)  or  /2025w/2025.xlsx  (women, WTA)
TENNIS_FILE = re.compile(r"/(\d{4})(w?)/\d{4}\.xlsx?$", re.I)


# Football ----------------------------------------------------------------------

def download_football(session: PoliteSession, seasons=config.FOOTBALL_SEASONS, leagues=config.LEAGUES, refresh=False) -> int:
    """Save one CSV per league and season. Files already saved are kept unless refresh=True."""
    FOOTBALL_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for season in seasons:
        for league in leagues:
            path = FOOTBALL_DIR / f"{season}_{league}.csv"
            if path.exists() and not refresh:
                continue
            url = config.FOOTBALL_DATA_CSV.format(season=season, league=league)
            try:
                path.write_bytes(session.get(url).content)
                saved += 1
            except (requests.RequestException, PermissionError) as exc:
                log.warning("could not download %s: %s", url, exc)
    log.info("football odds: %d files saved to %s", saved, FOOTBALL_DIR)
    return saved


def load_football() -> pd.DataFrame:
    """All saved football files in one table, with `league`, `season` and a parsed `date`."""
    frames = []
    for path in sorted(FOOTBALL_DIR.glob("[0-9]*_*.csv")):
        season, league = path.stem.split("_", 1)
        df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
        df = df.dropna(subset=["HomeTeam"]).copy()
        df["season"], df["league"] = season, league
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no football odds in {FOOTBALL_DIR}; run `python -m pvb odds football` first")
    df = pd.concat(frames, ignore_index=True)
    # older files write years as "21", newer ones as "2021"
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df


# Tennis ------------------------------------------------------------------------

def tennis_links(session: PoliteSession) -> dict[tuple[int, str], str]:
    """{(year, "ATP" or "WTA"): file URL}, read from tennis-data's download page."""
    soup = BeautifulSoup(session.get(config.TENNIS_DATA_INDEX).text, "html.parser")
    links = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        match = TENNIS_FILE.search(href)
        if match:
            tour = "WTA" if match.group(2) else "ATP"
            links[(int(match.group(1)), tour)] = urljoin(config.TENNIS_DATA_INDEX, href)
    return links


def download_tennis(session: PoliteSession, years=config.TENNIS_YEARS, refresh=False) -> int:
    """Save the ATP and WTA files for each year. Use refresh=True to update the current year."""
    TENNIS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for (year, tour), url in sorted(tennis_links(session).items()):
        if year not in years:
            continue
        extension = url[url.rfind("."):]
        path = TENNIS_DIR / f"{year}_{tour}{extension}"
        if path.exists() and not refresh:
            continue
        try:
            path.write_bytes(session.get(url).content)
            saved += 1
        except (requests.RequestException, PermissionError) as exc:
            log.warning("could not download %s: %s", url, exc)
    log.info("tennis odds: %d files saved to %s", saved, TENNIS_DIR)
    return saved


def load_tennis() -> pd.DataFrame:
    """All saved tennis files in one table, with `year` and `tour` columns."""
    frames = []
    for path in sorted(TENNIS_DIR.glob("[0-9]*_*.xls*")):  # skips Excel lock files (~$...)
        year, tour = path.stem.split("_", 1)
        frames.append(pd.read_excel(path).assign(year=int(year), tour=tour))
    if not frames:
        raise FileNotFoundError(f"no tennis odds in {TENNIS_DIR}; run `python -m pvb odds tennis` first")
    return pd.concat(frames, ignore_index=True)


# Shared ------------------------------------------------------------------------

def implied_probabilities(*decimal_odds: float) -> list[float]:
    """Turn decimal odds into probabilities that sum to 1 (removes the bookmaker's margin).

    >>> [round(p, 3) for p in implied_probabilities(2.0, 3.5, 4.0)]
    [0.483, 0.276, 0.241]
    """
    raw = [1 / odd for odd in decimal_odds]
    total = sum(raw)
    return [p / total for p in raw]
