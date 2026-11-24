"""Run a bounded retrieval pilot or audit saved football previews offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import pandas as pd
import requests

from .. import config
from ..analysis import early_odds_time
from ..http import PoliteSession
from ..sportsmole import ARTICLE_MARKER, classify
from .html_integrity import COMPARISON_COLUMNS, audit_article, relation, timing_context

SEASONS = ("2122", "2223", "2324", "2425")
PER_SEASON = 25
PILOT_DIR = config.ROOT / "data" / "pilot_html_integrity"
RESULTS_DIR = config.ROOT / "results" / "advanced" / "html_integrity"
MATCHES = config.PROCESSED / "matched_football.csv"
PREVIEWS = config.INTERIM / "previews.csv"
MATCH_COLUMNS = ("article_id", "season", "published", "kickoff", "home_team", "away_team")
PREVIEW_COLUMNS = ("article_id", "sport", "url", "published", "side_a", "side_b",
                   "pick_home_goals", "pick_away_goals", "pick_result", *COMPARISON_COLUMNS)
SAMPLE_COLUMNS = ("season", "article_id", "url")
DOWNLOAD_COLUMNS = ("article_id", "url", "status", "fetched_at", "http_status")


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(columns), dtype=str, keep_default_na=False)


def _valid_url(url: str, article_id: str) -> bool:
    parts = urlsplit(url)
    return (parts.scheme == "https" and parts.netloc == "www.sportsmole.co.uk" and not parts.query
            and not parts.fragment and re.fullmatch(r"[1-9][0-9]*", article_id) is not None
            and classify(url) == ("football", int(article_id)))


def load_development_articles(matches: Path = MATCHES, previews: Path = PREVIEWS) -> pd.DataFrame:
    """Join matched development football articles without loading outcomes or odds."""
    games = _read(matches, MATCH_COLUMNS).rename(columns={"published": "matched_published"})
    games = games.loc[games["season"].isin(SEASONS)]
    articles = _read(previews, PREVIEW_COLUMNS).rename(columns={"published": "preview_published"})
    articles = articles.loc[articles["article_id"].isin(games["article_id"]) & articles["sport"].eq("football")]
    cohort = games.merge(articles, on="article_id", how="inner", validate="one_to_one")
    if cohort["url"].duplicated().any() or not all(
        _valid_url(url, article_id) for url, article_id in zip(cohort["url"], cohort["article_id"])
    ):
        raise ValueError("Development cohort has duplicate or invalid preview URLs")
    kickoff = pd.to_datetime(cohort["kickoff"], format="ISO8601", utc=True, errors="coerce")
    usable = kickoff.notna() & kickoff.ne(kickoff.dt.normalize())
    cohort["estimated_early_cutoff"] = ""
    cohort.loc[usable, "estimated_early_cutoff"] = early_odds_time(kickoff.loc[usable]).map(datetime.isoformat)
    return cohort.sort_values(["season", "article_id"]).reset_index(drop=True)


def select_pilot(cohort: pd.DataFrame) -> pd.DataFrame:
    """Rank IDs by a fixed SHA-256 salt within each season, independent of input order."""
    ranked = cohort.assign(sample_rank=[
        hashlib.sha256(f"pvb-html-integrity-v1:{season}:{article_id}".encode()).hexdigest()
        for season, article_id in zip(cohort["season"], cohort["article_id"])
    ]).sort_values(["season", "sample_rank", "article_id"])
    sample = ranked.groupby("season", sort=False).head(PER_SEASON).drop(columns="sample_rank")
    if sample["season"].value_counts().to_dict() != dict.fromkeys(SEASONS, PER_SEASON):
        raise ValueError("Pilot requires 25 distinct articles in each development season")
    return sample.reset_index(drop=True)


class _PilotSession(PoliteSession):
    def __init__(self):
        delay = config.REQUEST_DELAY if math.isfinite(config.REQUEST_DELAY) else 2.0
        super().__init__(delay=max(2.0, delay), retries=1)
        self.session.max_redirects = 0
        self._wildcard_disallows = []

    def allowed(self, url: str) -> bool:
        return super().allowed(url) and not any(
            pattern.match(urlsplit(url).path) for pattern in self._wildcard_disallows
        )

    def _read_robots(self, site: str) -> robotparser.RobotFileParser:
        response = self._throttled_get(site + "/robots.txt")
        if response.status_code != 200 or not re.search(r"(?im)^\s*user-agent\s*:", response.text):
            raise PermissionError(f"robots.txt unavailable or unrecognized (HTTP {response.status_code})")
        rules = robotparser.RobotFileParser()
        lines = response.text.splitlines()
        rules.parse(lines)
        for agent in ("*", self.user_agent):
            crawl_delay, request_rate = rules.crawl_delay(agent), rules.request_rate(agent)
            if crawl_delay is not None:
                self.delay = max(self.delay, crawl_delay)
            if request_rate is not None and request_rate.requests > 0:
                self.delay = max(self.delay, request_rate.seconds / request_rate.requests)
        # Python 3.11's robotparser does not interpret wildcards or end anchors.
        # Treat any matching Disallow pattern as a deny, even across agent groups.
        for line in lines:
            key, _, path = line.split("#", 1)[0].partition(":")
            path = path.strip()
            if key.strip().lower() == "disallow" and path:
                pattern = re.escape(path.rstrip("$")).replace(r"\*", ".*")
                self._wildcard_disallows.append(re.compile(pattern + ("$" if path.endswith("$") else "")))
        return rules


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _download_one(session: PoliteSession, url: str, path: Path) -> tuple[str, str, int | None]:
    try:
        response = session.get(url)
    except PermissionError:
        return "robots_denied", "", None
    except requests.HTTPError as exc:
        return "http_error", "", exc.response.status_code
    except requests.RequestException:
        return "request_error", "", None
    if response.status_code != 200 or ARTICLE_MARKER not in response.content:
        return "not_preview", "", response.status_code
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    partial = path.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.replace(path)
    return "saved", fetched_at, response.status_code


def download_pilot(sample: pd.DataFrame) -> dict:
    """Fetch the locked 100-URL pilot. Requires prior written publisher consent."""
    started = time.perf_counter()
    sample = sample.loc[:, list(SAMPLE_COLUMNS)]
    if (sample["season"].value_counts().to_dict() != dict.fromkeys(SEASONS, PER_SEASON)
            or sample["article_id"].duplicated().any() or sample["url"].duplicated().any()
            or not all(_valid_url(url, article_id) for article_id, url in zip(sample["article_id"], sample["url"]))):
        raise ValueError("Invalid pilot cohort")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    lock = PILOT_DIR / "download.lock"
    lock.touch(exist_ok=False)
    try:
        return _download_sample(sample, started)
    finally:
        lock.unlink()


def _download_sample(sample: pd.DataFrame, started: float) -> dict:
    sample_file = PILOT_DIR / "sample.csv"
    serialized = sample.to_csv(index=False, lineterminator="\n")
    if sample_file.exists() and sample_file.read_text(encoding="utf-8") != serialized:
        raise ValueError("Pilot directory already belongs to a different 100-URL sample")
    if not sample_file.exists():
        sample_file.write_text(serialized, encoding="utf-8")
    manifest = PILOT_DIR / "manifest.csv"
    log = _read(manifest, DOWNLOAD_COLUMNS) if manifest.exists() and manifest.stat().st_size else None
    recorded = set(log["article_id"]) if log is not None else set()
    counts = Counter(log["status"]) if log is not None else Counter()
    attempted_this_run = 0
    stop = None
    session = _PilotSession()
    try:
        first_url = sample.iloc[0]["url"]
        try:
            session.allowed(first_url)
        except (PermissionError, requests.RequestException) as exc:
            stop = {"status": "robots_unavailable", "error_type": type(exc).__name__}
        with manifest.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if log is None:
                writer.writerow(DOWNLOAD_COLUMNS)
            for _, article_id, url in sample.itertuples(index=False, name=None):
                if stop is not None:
                    break
                path = PILOT_DIR / f"{article_id}.html"
                if article_id in recorded or path.exists():
                    continue
                if not session.allowed(url):
                    stop = {"status": "robots_denied"}
                    break
                attempted_this_run += 1
                status, fetched_at, http_status = _download_one(session, url, path)
                writer.writerow((article_id, url, status, fetched_at, http_status))
                handle.flush()
                counts[status] += 1
                if status in {"robots_denied", "not_preview", "request_error"} or (
                    status == "http_error" and http_status not in {404, 410}
                ):
                    stop = {"status": status, "http_status": http_status}
                    break
                if attempted_this_run % 10 == 0:
                    print(f"Pilot progress: {attempted_this_run} attempted this run, {counts['saved']} recorded saves", flush=True)
    finally:
        session.session.close()
    summary = {
        "selected": len(sample), "attempted_this_run": attempted_this_run, "recorded_attempts": sum(counts.values()),
        "download_statuses": dict(counts), "stopped": stop, "request_delay_seconds": session.delay,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(RESULTS_DIR / "download.json", summary)
    return summary


def _reference_counts(cohort: pd.DataFrame) -> dict:
    counts = defaultdict(Counter)
    for record in cohort.to_dict("records"):
        context = timing_context(record)
        pairs = (
            ("preview_published", "matched_published"), ("preview_published", "estimated_early_cutoff"),
            ("preview_published", "kickoff"), ("estimated_early_cutoff", "kickoff"),
        )
        for left, right in pairs:
            counts[f"{left}_vs_{right}"][relation(context[left], context[right])] += 1
    return counts


def _fetch_times(manifest: Path | None) -> dict[tuple[str, int], tuple[str, ...]]:
    if manifest is None:
        return {}
    log = _read(manifest, ("article_id", "url", "status", "fetched_at"))
    saved = log.loc[log["status"].eq("saved") & log["fetched_at"].ne("")]
    times = defaultdict(list)
    for row in saved.itertuples(index=False):
        if _valid_url(row.url, row.article_id):
            times[classify(row.url)].append(row.fetched_at)
    return {key: tuple(dict.fromkeys(values)) for key, values in times.items()}


def _update_counts(counts: dict, row: dict) -> None:
    counts["parse_status"][row["parse_status"]] += 1
    if "timestamps" in row:
        if row["invalid_json_ld_scripts"]:
            counts["metadata_issues"]["article_with_invalid_json_ld"] += 1
        for name, field in row["timestamps"].items():
            counts[f"timestamp_status/{name}"][field["status"]] += 1
            if field["invalid_candidates"]:
                counts["metadata_issues"][f"article_with_invalid_{name}"] += 1
            if field["selected"]:
                for descriptor in ("source", "precision", "timezone_basis"):
                    counts[f"timestamp_{descriptor}/{name}"][field["selected"][descriptor]] += 1
        for comparison, status in row["comparisons"].items():
            counts[f"timing/{comparison}"][status] += 1
    if row["parse_status"] != "parsed":
        return
    for section, detail in row["text"].items():
        counts[f"text/{section}"][detail["status"]] += 1
    any_change = any(row["text"][section]["status"] != "same" for section in COMPARISON_COLUMNS)
    body_change = any(row["text"][section]["status"] != "same" for section in COMPARISON_COLUMNS[1:-1])
    counts["text_any"]["changed" if any_change else "same"] += 1
    counts["text_body"]["changed" if body_change else "same"] += 1
    counts["prediction"][row["prediction"]["status"]] += 1
    if row["prediction"]["result_changed"] is not None:
        counts["prediction_result"]["changed" if row["prediction"]["result_changed"] else "same"] += 1
    for field in ("review_flags", "reference_review_flags"):
        counts[field]["flagged" if row[field] else "clear"] += 1
        counts[field].update(row[field].keys())
    new_flags = any(set(sections) - set(row["reference_review_flags"].get(reason, []))
                    for reason, sections in row["review_flags"].items())
    counts["new_review_flags"]["flagged" if new_flags else "clear"] += 1


def audit_saved(
    cohort: pd.DataFrame,
    html_dir: Path,
    *,
    version: str,
    results_dir: Path = RESULTS_DIR,
    manifest: Path | None = None,
) -> dict:
    """Audit <article_id>.html files and an optional same-version manifest offline."""
    started = time.perf_counter()
    if version not in {"original", "pilot_download"}:
        raise ValueError("Version must be 'original' or 'pilot_download'")
    if not html_dir.is_dir():
        raise FileNotFoundError(f"HTML directory unavailable: {html_dir}")
    if manifest is None and (html_dir / "manifest.csv").is_file():
        manifest = html_dir / "manifest.csv"
    fetch_times = _fetch_times(manifest)
    counts = defaultdict(Counter)
    by_season = defaultdict(Counter)
    results_dir.mkdir(parents=True, exist_ok=True)
    details = results_dir / f"{version}_articles.jsonl"
    partial = details.with_suffix(".part")
    with partial.open("w", encoding="utf-8") as handle:
        for reference in cohort.to_dict("records"):
            article_id, url, season = (reference[field] for field in ("article_id", "url", "season"))
            if not _valid_url(url, article_id):
                raise ValueError("Invalid URL or article ID in audit cohort")
            path = html_dir / f"{article_id}.html"
            try:
                page = path.read_bytes()
            except FileNotFoundError:
                observed = {"parse_status": "missing_html"}
            else:
                observed = audit_article(page, reference, fetched_at=fetch_times.get(classify(url), ()))
            row = {"article_id": article_id, "season": season, "version": version, **observed}
            _update_counts(counts, row)
            by_season[season][row["parse_status"]] += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(details)
    summary = {
        "version": version, "selected": len(cohort), "manifest_used": manifest is not None,
        "by_season": by_season, "reference_timing": _reference_counts(cohort), "observations": counts,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(results_dir / f"{version}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample", help="Summarize the deterministic development pilot cohort")
    download = commands.add_parser("download-pilot", help="Fetch at most 100 articles into the private pilot directory")
    download.add_argument("--written-consent", action="store_true",
                          help="Confirm prior written consent from the publisher or copyright owner")
    audit = commands.add_parser("audit", help="Inspect local HTML files offline")
    audit.add_argument("--version", choices=("original", "pilot_download"), required=True)
    audit.add_argument("--html-dir", type=Path)
    audit.add_argument("--manifest", type=Path)
    audit.add_argument("--sample", action="store_true", help="Restrict an original HTML audit to the pilot sample")
    args = parser.parse_args()
    if args.command == "download-pilot" and not args.written_consent:
        parser.error("Sports Mole's terms require prior written consent to download articles")
    started = time.perf_counter()
    cohort = load_development_articles()
    if args.command == "sample":
        sample = select_pilot(cohort)
        summary = {
            "selected": len(sample), "selected_by_season": sample["season"].value_counts().to_dict(),
            "pool_by_season": cohort["season"].value_counts().to_dict(), "reference_timing": _reference_counts(sample),
            "sample_sha256": hashlib.sha256(sample.loc[:, list(SAMPLE_COLUMNS)].to_csv(
                index=False, lineterminator="\n").encode()).hexdigest(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        _write_json(RESULTS_DIR / "sample.json", summary)
    elif args.command == "download-pilot":
        summary = download_pilot(select_pilot(cohort))
    else:
        if args.version == "pilot_download":
            locked = _read(PILOT_DIR / "sample.csv", SAMPLE_COLUMNS)
            cohort = locked.merge(cohort, on=list(SAMPLE_COLUMNS), how="left", validate="one_to_one", indicator=True)
            if not cohort["_merge"].eq("both").all():
                parser.error("The current inputs do not contain the recorded pilot sample")
        elif args.sample:
            cohort = select_pilot(cohort)
        default_dir = PILOT_DIR if args.version == "pilot_download" else config.SPORTSMOLE_RAW
        summary = audit_saved(cohort, args.html_dir or default_dir, version=args.version, manifest=args.manifest)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
