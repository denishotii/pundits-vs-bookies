"""Turn saved Sports Mole preview pages into one table, one row per preview.

A preview page has a short intro, then sections under <h2> headings:
"Match preview", "Team News" (football) or "Head To Head" (tennis), and
finally the verdict, e.g. "We say: Brighton & Hove Albion 2-2 Leeds United"
or "We say: Medvedev to win in three sets".
"""

from __future__ import annotations

import html
import json
import logging
import re

import pandas as pd
from bs4 import BeautifulSoup, Tag

from . import config
from .sportsmole import classify

log = logging.getLogger(__name__)

PREVIEWS_CSV = config.INTERIM / "previews.csv"

# h2 heading -> the column its paragraphs go into
SECTIONS = {"match preview": "text_preview", "team news": "text_team_news", "head to head": "text_head_to_head"}
TEXT_COLUMNS = ["text_intro", "text_preview", "text_team_news", "text_head_to_head", "text_verdict", "text_other"]
PICK_COLUMNS = ["pick_home_goals", "pick_away_goals", "pick_result", "pick_winner", "pick_sets"]

# "We say: Brighton & Hove Albion 2-2 Leeds United"
FOOTBALL_PICK = re.compile(r"^We say:\s*(?P<home>.+?)\s+(?P<hg>\d+)\s*-\s*(?P<ag>\d+)\s+(?P<away>.+)$", re.I)
# "We say: Medvedev to win in three sets" (the sets part is sometimes missing)
TENNIS_PICK = re.compile(
    r"^We say:\s*(?P<winner>.+?)\s+to win\b(?:\s+in\s+(?P<sets>straight|two|three|four|five)\s+sets)?", re.I
)
# "Preview: Arsenal vs Chelsea - prediction, ..." or, in tennis,
# "Preview: ASB Classic: Elina Svitolina vs. Sonay Kartal - prediction, ..."
TITLE = re.compile(r"^Preview:\s*(?:[^:]+:\s+)?(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)\s+-\s+", re.I)
SLUG = re.compile(r"/preview/(?P<a>.+?)-vs-(?P<b>.+?)-prediction")
# The byline and a promo link are not part of the preview text
NOISE = re.compile(r"^(By [^|]+\|.*|For data analysis of the most likely results.*)$", re.I)


def parse_preview(page: str | bytes, url: str | None = None) -> dict:
    """Parse one preview page into a flat dict. Missing values are None.

    `url` is optional: saved pages carry their own address in <link rel="canonical">.
    """
    soup = BeautifulSoup(page, "html.parser")
    body = soup.select_one("div.sm-article-body")
    if url is None:
        url = _canonical_url(soup)
    if body is None or url is None:
        raise ValueError(f"not a preview page: {url}")

    sport, article_id = classify(url) or (None, None)
    title = soup.h1.get_text(" ", strip=True) if soup.h1 else None
    meta = _article_metadata(soup)
    texts, verdict = _sections(body)
    side_a, side_b = _sides(title, url)

    record = {
        "article_id": article_id,
        "url": url,
        "sport": sport,
        "title": title,
        "published": meta["published"],
        "author": meta["author"],
        "side_a": side_a,  # home team in football
        "side_b": side_b,  # away team in football
        "pick_text": verdict,
    }
    record.update(parse_pick(verdict, sport))
    for column in TEXT_COLUMNS:
        record[column] = texts.get(column, "")
    record["word_count"] = sum(len(record[column].split()) for column in TEXT_COLUMNS)
    return record


def parse_pick(verdict: str | None, sport: str | None) -> dict:
    """Read the "We say" line. Football gives a score, tennis a winner and number of sets."""
    pick: dict[str, object] = {column: None for column in PICK_COLUMNS}
    if not verdict:
        return pick

    if sport == "football":
        match = FOOTBALL_PICK.match(verdict)
        if match:
            home_goals, away_goals = int(match["hg"]), int(match["ag"])
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"
            pick.update(pick_home_goals=home_goals, pick_away_goals=away_goals, pick_result=result)

    elif sport == "tennis":
        match = TENNIS_PICK.match(verdict)
        if match:
            pick["pick_winner"] = match["winner"]
            # "straight" means 2 or 3 sets depending on the match format, so we keep the word
            if match["sets"]:
                pick["pick_sets"] = match["sets"].lower()

    return pick


def parse_all() -> pd.DataFrame:
    """Parse every saved page in data/raw/sportsmole/ and write previews.csv."""
    rows, failed = [], 0
    for path in sorted(config.SPORTSMOLE_RAW.glob("*.html")):
        try:
            rows.append(parse_preview(path.read_bytes()))
        except ValueError as exc:
            failed += 1
            log.warning("%s: %s", path.name, exc)

    df = pd.DataFrame(rows)
    if not df.empty:
        df[["pick_home_goals", "pick_away_goals"]] = df[["pick_home_goals", "pick_away_goals"]].astype("Int64")
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREVIEWS_CSV, index=False)
    log.info("parsed %d previews (%d failed) -> %s", len(df), failed, PREVIEWS_CSV)
    return df


def _sections(body: Tag) -> tuple[dict[str, str], str | None]:
    """Group the paragraphs under their <h2> heading. Returns (texts by column, verdict line)."""
    paragraphs: dict[str, list[str]] = {}
    current = "text_intro"
    verdict = None
    for element in body.find_all(["h2", "p"]):
        text = element.get_text(" ", strip=True)
        if element.name == "h2":
            if text.lower().startswith("we say"):
                current, verdict = "text_verdict", text
            else:
                current = SECTIONS.get(text.lower(), "text_other")
        elif text and not NOISE.match(text):
            paragraphs.setdefault(current, []).append(text)
    texts = {column: "\n".join(lines) for column, lines in paragraphs.items()}
    return texts, verdict


def _article_metadata(soup: BeautifulSoup) -> dict:
    """Publish time and author from the page's JSON-LD (a NewsArticle)."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for item in items:
            if isinstance(item, dict) and item.get("datePublished"):
                return {"published": item["datePublished"], "author": _author_name(item.get("author"))}

    tag = soup.find("meta", attrs={"property": "article:published_time"})
    published = tag.get("content") if isinstance(tag, Tag) else None
    return {"published": published, "author": None}


def _author_name(author) -> str | None:
    if isinstance(author, list):
        author = author[0] if author else None
    if isinstance(author, dict):
        author = author.get("name")
    return html.unescape(author) if isinstance(author, str) else None  # "O&#039;Shea" -> "O'Shea"


def _canonical_url(soup: BeautifulSoup) -> str | None:
    tag = soup.find("link", rel="canonical")
    href = tag.get("href") if isinstance(tag, Tag) else None
    return href if isinstance(href, str) else None


def _sides(title: str | None, url: str) -> tuple[str | None, str | None]:
    """The two teams or players: from the page title if possible, else from the URL."""
    match = TITLE.match(title or "")
    if match:
        return match["a"].strip(), match["b"].strip()
    match = SLUG.search(url)
    if match:
        return match["a"].replace("-", " "), match["b"].replace("-", " ")
    return None, None
