"""Inspect a saved preview and compare it with its recorded fixture and parse."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from ..parse import TEXT_COLUMNS, parse_preview
from ..sportsmole import classify

UTC = timezone.utc
DAY = timedelta(days=1)
UNKNOWN_OFFSET = timedelta(hours=14)
TICK = timedelta(microseconds=1)
ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::(?P<seconds>\d{2})"
    r"(?:[.,](?P<fraction>\d{1,6}))?)?(?P<zone>Z|[+-]\d{2}:?\d{2})?)?"
)
TIMESTAMP_FIELDS = ("datePublished", "dateModified", "fetched_at")
META_NAMES = {
    "article:published_time": "datePublished",
    "article:modified_time": "dateModified",
}
COMPARISON_COLUMNS = ("title", *TEXT_COLUMNS, "pick_text")
RESULT_TITLE = re.compile(r"^(?:match report|full[- ]time(?: result)?|final score)\s*:", re.I)
RESULT_LINE = re.compile(r"^(?:full[- ]time(?: score| result)?|final score)\s*:\s*(.+)$", re.I)
SCORED_TEAMS = re.compile(r"(?P<home>.+?)\s+\d{1,2}\s*[-–—:]\s*\d{1,2}\s+(?P<away>.+)")
COMPLETED_MATCH = re.compile(r"^this match (?:ended|finished)\s+\d{1,2}\s*[-–—:]\s*\d{1,2}(?:[.!]|$)", re.I)


@dataclass(frozen=True)
class Timestamp:
    source: str
    raw: str | None
    precision: str | None = None
    timezone_basis: str | None = None
    earliest: datetime | None = None
    latest: datetime | None = None

    def report(self) -> dict:
        row = asdict(self)
        for field in ("earliest", "latest"):
            row[field] = row[field].isoformat() if row[field] is not None else None
        return row


def timestamp(raw: object, source: str, *, legacy_utc: bool = False) -> Timestamp:
    """Preserve recorded resolution. Legacy CSVs use UTC and midnight for date-only."""
    value = raw.strip() if isinstance(raw, str) else None
    match = ISO_TIMESTAMP.fullmatch(value or "")
    if match is None:
        return Timestamp(source, None)
    try:
        clock = datetime.fromisoformat(value).astimezone(UTC) if match["zone"] else datetime.fromisoformat(value)
    except (ValueError, OverflowError):
        return Timestamp(source, value)

    fraction = match["fraction"]
    if len(value) == 10 or (legacy_utc and clock.time() == datetime.min.time()):
        precision, width = "date", DAY
    elif fraction:
        precision, width = "fractional_second", timedelta(microseconds=10 ** (6 - len(fraction)))
    elif match["seconds"]:
        precision, width = "second", timedelta(seconds=1)
    else:
        precision, width = "minute", timedelta(minutes=1)
    try:
        if match["zone"]:
            return Timestamp(source, value, precision, "explicit_offset", clock, clock + width - TICK)
        clock = clock.replace(tzinfo=UTC)
        if legacy_utc:
            return Timestamp(source, value, precision, "legacy_utc", clock, clock + width - TICK)
        return Timestamp(source, value, precision, "unknown", clock - UNKNOWN_OFFSET,
                         clock + UNKNOWN_OFFSET + width - TICK)
    except OverflowError:
        return Timestamp(source, value)


def relation(left: Timestamp | None, right: Timestamp | None) -> str:
    if left is None or right is None or left.earliest is None or right.earliest is None:
        return "unavailable"
    if left.earliest > right.latest:
        return "after"
    if left.latest < right.earliest:
        return "before"
    if (left.earliest, left.latest) == (right.earliest, right.latest) and (
        left.precision != "date" and left.timezone_basis != "unknown" and right.timezone_basis != "unknown"
    ):
        return "same"
    return "overlap"


@dataclass
class TimestampField:
    candidates: list[Timestamp]

    @property
    def selected(self) -> Timestamp | None:
        valid = [value for value in self.candidates if value.earliest is not None]
        if not valid or any(relation(a, b) in {"before", "after"} for a, b in combinations(valid, 2)):
            return None
        return min(valid, key=lambda value: value.latest - value.earliest)

    def report(self) -> dict:
        selected = self.selected
        valid_count = sum(value.earliest is not None for value in self.candidates)
        return {
            "status": ("resolved" if selected else "conflict" if valid_count
                       else "invalid" if self.candidates else "missing"),
            "invalid_candidates": len(self.candidates) - valid_count,
            "selected": selected.report() if selected else None,
            "candidates": [value.report() for value in self.candidates],
        }


def _json_nodes(data):
    if isinstance(data, list):
        for item in data:
            yield from _json_nodes(item)
    elif isinstance(data, dict):
        yield data
        for key in ("@graph", "mainEntity"):
            yield from _json_nodes(data.get(key))


def _article_schema(node: dict) -> bool:
    types = node.get("@type", [])
    if isinstance(types, str):
        types = [types]
    return isinstance(types, list) and (not types or any(
        isinstance(name, str)
        and (re.split(r"[/#:]", name)[-1] in {"Article", "BlogPosting", "LiveBlogPosting"}
             or name.endswith("NewsArticle"))
        for name in types
    ))


def _extract_timestamps(soup: BeautifulSoup, fetched_at: tuple[str, ...]) -> tuple[dict, int]:
    fields = {key: TimestampField([]) for key in TIMESTAMP_FIELDS}
    errors = 0
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (ValueError, RecursionError):
            errors += 1
            continue
        for node in _json_nodes(data):
            if _article_schema(node):
                for key in TIMESTAMP_FIELDS[:2]:
                    if key in node:
                        fields[key].candidates.append(timestamp(node[key], "json_ld"))
    for tag in soup.find_all(["meta", "time"]):
        for name in {tag.get("property"), tag.get("name")} & META_NAMES.keys():
            fields[META_NAMES[name]].candidates.append(timestamp(tag.get("content"), f"meta:{name}"))
        for key in tag.get("itemprop", "").split():
            if key in TIMESTAMP_FIELDS[:2]:
                raw = tag.get("datetime", tag.get("content", tag.get_text(strip=True)))
                fields[key].candidates.append(timestamp(raw, f"{tag.name}:itemprop"))
    for value in fetched_at:
        fields["fetched_at"].candidates.append(timestamp(value, "manifest:fetched_at"))
    return fields, errors


def _normalize(text: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def _compare_text(reference: dict, article: dict) -> dict:
    result = {}
    for column in COMPARISON_COLUMNS:
        old, new = _normalize(reference.get(column)), _normalize(article.get(column))
        status = "same" if old == new else "added" if not old else "removed" if not new else "changed"
        result[column] = {"status": status, "word_delta": len(new.split()) - len(old.split())}
    return result


def _pick(record: dict) -> tuple[int, int, str] | None:
    try:
        home, away = (int(record[key]) for key in ("pick_home_goals", "pick_away_goals"))
    except (KeyError, ValueError, TypeError):
        return None
    result = record.get("pick_result")
    if min(home, away) >= 0 and result == ("H" if home > away else "A" if away > home else "D"):
        return home, away, result
    return None


def _compare_pick(reference: dict, article: dict) -> dict:
    old, new = _pick(reference), _pick(article)
    if old is None or new is None:
        status = "unavailable_in_both" if old == new else "appeared" if old is None else "missing"
        return {"status": status, "result_changed": None}
    return {"status": "same" if old == new else "changed", "result_changed": old[2] != new[2]}


def _review_flags(article: dict, fixture: dict) -> dict:
    home = {_normalize(name).casefold() for name in (fixture.get("home_team"), fixture.get("side_a"))} - {""}
    away = {_normalize(name).casefold() for name in (fixture.get("away_team"), fixture.get("side_b"))} - {""}
    flags = {}
    if RESULT_TITLE.match(_normalize(article.get("title"))):
        flags["result_style_title"] = ["title"]
    for column in TEXT_COLUMNS:
        reasons = set()
        for paragraph in (article.get(column) or "").splitlines():
            normalized = _normalize(paragraph)
            line = RESULT_LINE.match(normalized)
            score = SCORED_TEAMS.fullmatch(line[1].rstrip(".!")) if line else None
            if score and score["home"].casefold() in home and score["away"].casefold() in away:
                reasons.add("labelled_fixture_result")
            if COMPLETED_MATCH.match(normalized):
                reasons.add("explicit_completed_match_score")
        for reason in sorted(reasons):
            flags.setdefault(reason, []).append(column)
    return flags


def timing_context(reference: dict) -> dict[str, Timestamp]:
    return {name: timestamp(reference.get(name), name, legacy_utc=name != "estimated_early_cutoff") for name in (
        "preview_published", "matched_published", "estimated_early_cutoff", "kickoff",
    )}


def audit_article(page: str | bytes, reference: dict, *, fetched_at: tuple[str, ...] = ()) -> dict:
    """Return text-free diagnostics for one saved HTML version without network access."""
    soup = BeautifulSoup(page, "html.parser")
    canonical = soup.find("link", rel="canonical", href=True)
    if canonical:
        canonical_url = urljoin(reference["url"], canonical["href"])
        if (urlsplit(canonical_url).hostname not in {"www.sportsmole.co.uk", "sportsmole.co.uk"}
                or classify(canonical_url) != classify(reference["url"])):
            return {"parse_status": "canonical_mismatch"}
    if soup.select_one("div.sm-article-body") is None:
        return {"parse_status": "not_preview"}
    fields, json_errors = _extract_timestamps(soup, fetched_at)
    context = timing_context(reference)
    known = {name: field.selected for name, field in fields.items()}
    pairs = [(name, other) for name in fields for other in context]
    pairs += [("dateModified", "datePublished"), ("fetched_at", "datePublished"), ("fetched_at", "dateModified")]
    timelines = context | known
    early_status = relation(context["estimated_early_cutoff"], context["kickoff"])
    report = {
        "timestamps": {name: field.report() for name, field in fields.items()},
        "timing_context": {name: value.report() for name, value in context.items()},
        "comparisons": {
            f"{a}_vs_{b}": "invalid_cutoff"
            if b == "estimated_early_cutoff" and early_status in {"after", "same", "overlap"}
            else relation(timelines[a], timelines[b]) for a, b in pairs
        },
        "invalid_json_ld_scripts": json_errors,
        "reference_review_flags": _review_flags(reference, reference),
    }
    try:
        article = parse_preview(page, reference["url"])
    except (ValueError, AttributeError):
        return report | {"parse_status": "parser_error"}
    return report | {
        "parse_status": "parsed",
        "text": _compare_text(reference, article),
        "prediction": _compare_pick(reference, article),
        "review_flags": _review_flags(article, reference),
    }
