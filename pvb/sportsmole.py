"""Find Sports Mole match previews and download them.

Sports Mole lists every article in yearly sitemaps
(sitemap-articles-2024.xml, ...). We read those, keep the football and tennis
preview pages, and save each page's raw HTML once, so parsing can be re-run
without downloading again.
"""

from __future__ import annotations

import csv
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .http import PoliteSession

log = logging.getLogger(__name__)

SITEMAP_INDEX = config.SPORTSMOLE + "/sitemap_index.xml"
ARTICLE_SITEMAP = re.compile(r"sitemap-articles-(\d{4})(?:-\d+)?\.xml$")
# e.g. /football/brighton-and-hove-albion/preview/brighton-vs-leeds-prediction-team-news-lineups_604333.html
PREVIEW_URL = re.compile(r"sportsmole\.co\.uk/(football|tennis)/(?:[^/]+/)*preview/[^/]+_(\d+)\.html$")
# Every real preview page has this element; a block or error page does not.
ARTICLE_MARKER = b"sm-article-body"

URLS_CSV = config.INTERIM / "preview_urls.csv"
MANIFEST_CSV = config.SPORTSMOLE_RAW / "manifest.csv"
MANIFEST_HEADER = ["article_id", "url", "status", "fetched_at"]


def parse_sitemap(xml_text: str) -> list[str]:
    """Return every <loc> URL in a sitemap or sitemap index."""
    root = ET.fromstring(xml_text)
    return [loc.text.strip() for loc in root.findall(".//{*}loc") if loc.text]


def classify(url: str) -> tuple[str, int] | None:
    """Return (sport, article_id) for a preview URL, or None for anything else."""
    match = PREVIEW_URL.search(url)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def article_sitemaps(session: PoliteSession, years: list[int]) -> list[tuple[str, int]]:
    """(sitemap URL, year) for the given years. Some years are split in two files."""
    found = []
    for url in parse_sitemap(session.get(SITEMAP_INDEX).text):
        match = ARTICLE_SITEMAP.search(url)
        if match and int(match.group(1)) in years:
            found.append((url, int(match.group(1))))
    return sorted(found)


def collect_preview_urls(session: PoliteSession, years: list[int], sports=config.SPORTS) -> pd.DataFrame:
    """One row per preview: url, sport, article_id, sitemap_year."""
    rows = []
    for sitemap, year in article_sitemaps(session, years):
        urls = parse_sitemap(session.get(sitemap).text)
        kept = 0
        for url in urls:
            found = classify(url)
            if found and found[0] in sports:
                rows.append({"url": url, "sport": found[0], "article_id": found[1], "sitemap_year": year})
                kept += 1
        log.info("%s: %d previews out of %d articles", sitemap.rsplit("/", 1)[-1], kept, len(urls))
    df = pd.DataFrame(rows, columns=["url", "sport", "article_id", "sitemap_year"])
    return df.drop_duplicates("article_id").sort_values("article_id").reset_index(drop=True)


def raw_path(article_id: int) -> Path:
    return config.SPORTSMOLE_RAW / f"{article_id}.html"


def download_previews(session: PoliteSession, urls: pd.DataFrame, limit: int | None = None) -> int:
    """Save the raw HTML of each preview not downloaded yet. Returns how many were saved.

    Safe to stop and restart: pages already on disk are skipped. Every attempt
    is logged in data/raw/sportsmole/manifest.csv, including failures.
    """
    config.SPORTSMOLE_RAW.mkdir(parents=True, exist_ok=True)
    on_disk = urls["article_id"].map(lambda a: raw_path(a).exists()).astype(bool)
    todo = urls[~on_disk]
    if limit is not None:
        todo = todo.head(limit)
    log.info("%d previews to download (%d already on disk)", len(todo), int(on_disk.sum()))

    write_header = not MANIFEST_CSV.exists() or MANIFEST_CSV.stat().st_size == 0
    saved = 0
    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as fh:
        manifest = csv.writer(fh)
        if write_header:
            manifest.writerow(MANIFEST_HEADER)
        for url, article_id in zip(todo["url"].astype(str), todo["article_id"].astype(int)):
            if raw_path(article_id).exists():  # another scraper saved it in the meantime
                continue
            status = _download_one(session, url, article_id)
            manifest.writerow([article_id, url, status, _now()])
            fh.flush()  # the log survives even if the run is killed
            if status == "saved":
                saved += 1
                if saved % 50 == 0:
                    log.info("saved %d / %d", saved, len(todo))
    return saved


def _download_one(session: PoliteSession, url: str, article_id: int) -> str:
    """Download one page and save it only if it really is a preview. Returns a status word."""
    try:
        resp = session.get(url)
    except Exception as exc:  # keep going; the manifest records the failure
        log.warning("failed %s: %s", url, exc)
        return f"error: {exc}"
    if ARTICLE_MARKER not in resp.content:
        log.warning("not a preview page (blocked or moved?): %s", url)
        return "no article body"
    # Write to a temporary file, then rename: a stopped run never leaves half a page behind
    path = raw_path(article_id)
    partial = path.with_suffix(".part")
    partial.write_bytes(resp.content)  # exactly the bytes we received
    partial.replace(path)
    return "saved"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
