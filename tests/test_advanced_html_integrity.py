import json

import pandas as pd
import pytest
import requests

from pvb.advanced import html_audit
from pvb.advanced.html_integrity import audit_article, relation, timestamp


def _url(article_id):
    return f"https://www.sportsmole.co.uk/football/example/preview/alpha-vs-beta-prediction_{article_id}.html"


REFERENCE = {
    "article_id": "123", "url": _url("123"), "season": "2324",
    "title": "Preview: Alpha vs Beta - prediction",
    "home_team": "Alpha FC", "away_team": "Beta FC", "side_a": "Alpha", "side_b": "Beta",
    "preview_published": "2024-01-05T12:00:00Z", "matched_published": "2024-01-05T12:00:00+00:00",
    "estimated_early_cutoff": "2024-01-05T15:00:00Z", "kickoff": "2024-01-06T15:00:00Z",
    "text_preview": "A café\u00a0opens.\n A   former match ended 3-0.",
    "pick_text": "We say: Alpha 2-0 Beta", "pick_home_goals": "2", "pick_away_goals": "0", "pick_result": "H",
}


def _page(metadata="", extra="", pick="2-0", canonical=""):
    return f"""<html><head><meta charset="utf-8">{canonical}{metadata}</head><body>
    <h1>Preview: Alpha vs Beta - prediction</h1><div class="sm-article-body">
    <h2>Match Preview</h2><p>A cafe\u0301 opens.</p><p>A former match ended 3-0.</p>{extra}
    <h2>We say: Alpha {pick} Beta</h2></div></body></html>"""


def test_extraction_reconciles_offsets_and_retains_provenance():
    page = _page('''<script type="application/ld+json">{"@graph":[
        {"@type":"NewsArticle", "datePublished":"2024-01-05T13:00:00+01:00",
         "dateModified":"2024-01-07T00:00:00Z"},
        {"@type":"SportsEvent", "dateModified":"2024-02-01T10:00:00Z"}]}</script>
        <script type="application/ld+json">{bad JSON</script>
        <meta property="article:published_time" content="2024-01-05T12:00:00Z">
        <time itemprop="dateModified" datetime="2024-01-07"></time>''')
    report = audit_article(page, REFERENCE, fetched_at=("2024-01-08T16:38:00Z",))
    published = report["timestamps"]["datePublished"]
    modified = report["timestamps"]["dateModified"]
    assert published["status"] == modified["status"] == "resolved"
    assert {item["source"] for item in published["candidates"]} == {"json_ld", "meta:article:published_time"}
    assert [item["precision"] for item in modified["candidates"]] == ["second", "date"]
    assert report["invalid_json_ld_scripts"] == 1
    assert report["comparisons"]["datePublished_vs_preview_published"] == "same"
    assert report["comparisons"]["dateModified_vs_estimated_early_cutoff"] == "after"
    assert report["comparisons"]["dateModified_vs_kickoff"] == "after"
    assert report["comparisons"]["fetched_at_vs_dateModified"] == "after"
    assert report["text"]["text_preview"] == {"status": "same", "word_delta": 0}
    assert report["review_flags"] == {}


def test_conflicting_and_invalid_candidates_do_not_set_definite_timing():
    page = _page('''<script type="application/ld+json">{"@type":"NewsArticle",
             "datePublished":"2024-01-06T14:23Z", "dateModified":"unparseable"}</script>
          <meta property="article:published_time" content="2024-01-06T16:23Z">''')
    record = audit_article(page, REFERENCE)
    assert record["timestamps"]["datePublished"]["status"] == "conflict"
    assert record["timestamps"]["dateModified"]["status"] == "invalid"
    assert record["timestamps"]["fetched_at"]["status"] == "missing"
    assert record["comparisons"]["datePublished_vs_kickoff"] == "unavailable"
    assert "unparseable" not in json.dumps(record)


def test_coarse_timestamps_and_invalid_early_cutoffs_are_distinct():
    cut = timestamp("2024-01-06T15:00:00Z", "kickoff")
    assert relation(timestamp("2024-01-06", "html"), cut) == "overlap"
    assert relation(timestamp("2024-01-07T00:00:00", "html"), cut) == "overlap"
    assert relation(timestamp("2024-01-07T00:00:00Z", "html"), cut) == "after"
    assert timestamp("2024-01-06T00:00:00Z", "csv", legacy_utc=True).precision == "date"
    page = _page('<meta property="article:modified_time" content="2024-01-06T18:00:00Z">')
    ref = REFERENCE | {"estimated_early_cutoff": "2024-01-06T16:00:00Z"}
    assert audit_article(page, ref)["comparisons"]["dateModified_vs_estimated_early_cutoff"] == "invalid_cutoff"


def test_body_changes_prediction_changes_and_review_flags():
    extra = "<p>Full-time: Alpha 1-2 Beta.</p><p>Final score: Alpha 1-0 Gamma.</p>"
    report = audit_article(_page(extra=extra, pick="1-2"), REFERENCE)
    assert report["text"]["text_preview"]["status"] == "changed"
    assert report["text"]["pick_text"]["status"] == "changed"
    assert report["prediction"] == {"status": "changed", "result_changed": True}
    assert report["review_flags"] == {"labelled_fixture_result": ["text_preview"]}
    assert audit_article(_page(pick="X-Y"), REFERENCE)["prediction"]["status"] == "missing"
    assert "Full-time: Alpha" not in json.dumps(report)


def test_wrong_canonical_and_missing_body_excluded_from_metadata():
    canonical = f'<link rel="canonical" href="{_url("456")}">'
    assert audit_article(_page(canonical=canonical), REFERENCE) == {"parse_status": "canonical_mismatch"}
    assert audit_article("<html>error</html>", REFERENCE) == {"parse_status": "not_preview"}


def _cohort():
    return pd.DataFrame([
        {"season": season, "article_id": str(article_id), "url": _url(article_id)}
        for group, season in enumerate(html_audit.SEASONS, start=1)
        for article_id in range(group * 100, group * 100 + 30)
    ])


def test_sampling_is_balanced_deterministic_and_independent_of_input_order():
    cohort = _cohort()
    sample = html_audit.select_pilot(cohort)
    assert sample["season"].value_counts().to_dict() == dict.fromkeys(html_audit.SEASONS, 25)
    assert sample["url"].nunique() == 100
    pd.testing.assert_frame_equal(sample, html_audit.select_pilot(cohort.sample(frac=1, random_state=42)))
    with pytest.raises(ValueError, match="25 distinct articles"):
        html_audit.select_pilot(cohort.iloc[30:])


def test_cohort_excludes_holdouts_and_uses_summer_uk_snapshot(tmp_path):
    match = dict.fromkeys(html_audit.MATCH_COLUMNS, "") | {
        "article_id": "123", "season": "2425", "kickoff": "2024-08-17T16:00:00Z",
    }
    preview = dict.fromkeys(html_audit.PREVIEW_COLUMNS, "") | {
        "article_id": "123", "sport": "football", "url": _url("123"),
    }
    matches, previews = tmp_path / "matches.csv", tmp_path / "previews.csv"
    pd.DataFrame([match, match | {"season": "2526", "article_id": "456"}]).to_csv(matches, index=False)
    pd.DataFrame([preview, preview | {"url": _url("456"), "article_id": "456"}]).to_csv(previews, index=False)
    result = html_audit.load_development_articles(matches, previews)
    assert result["article_id"].tolist() == ["123"]
    assert result["estimated_early_cutoff"].tolist() == ["2024-08-16T14:00:00+00:00"]


def test_offline_batch_uses_only_same_article_saved_manifest_entries(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    (originals / "123.html").write_text(_page(), encoding="utf-8")
    (originals / "manifest.csv").write_text(
        "article_id,url,status,fetched_at\n"
        f"123,{_url('123')},error,2024-01-01T00:00:00Z\n"
        f"123,{_url('456')},saved,2024-01-01T00:00:00Z\n"
        f"123,{_url('123')},saved,2024-01-08T00:30:00Z\n", encoding="utf-8",
    )
    cohort = pd.DataFrame([REFERENCE, REFERENCE | {"article_id": "456", "url": _url("456")}])
    report = html_audit.audit_saved(cohort, originals, version="original", results_dir=tmp_path)
    assert report["observations"]["parse_status"] == {"parsed": 1, "missing_html": 1}
    assert report["observations"]["timing/fetched_at_vs_kickoff"] == {"after": 1}
    assert report["observations"]["prediction"] == {"same": 1}
    rows = [json.loads(line) for line in (tmp_path / "original_articles.jsonl").read_text().splitlines()]
    assert all(row["version"] == "original" for row in rows)
    assert len(rows[0]["timestamps"]["fetched_at"]["candidates"]) == 1


def test_download_stops_at_rate_limit_and_writes_only_pilot_files(tmp_path, monkeypatch):
    pilot = tmp_path / "private_pilot"
    monkeypatch.setattr(html_audit, "PILOT_DIR", pilot)
    monkeypatch.setattr(html_audit, "RESULTS_DIR", tmp_path / "results")
    sample = html_audit.select_pilot(_cohort())
    seen = []

    class FakeSession:
        delay = 2.0
        session = requests.Session()

        def allowed(self, url):
            return True

        def get(self, url):
            seen.append(url)
            response = requests.Response()
            response.status_code = 429 if len(seen) == 2 else 200
            response._content = b'<div class="sm-article-body"></div>'
            response.raise_for_status()
            return response

    monkeypatch.setattr(html_audit, "_PilotSession", FakeSession)
    report = html_audit.download_pilot(sample)
    assert seen == sample["url"].tolist()[:2]
    assert report["stopped"] == {"status": "http_error", "http_status": 429}
    assert report["download_statuses"] == {"saved": 1, "http_error": 1}
    assert sorted(p.name for p in pilot.glob("*.html")) == [f"{sample.iloc[0]['article_id']}.html"]
    assert not (pilot / "download.lock").exists()


def test_failed_robots_preflight_is_not_an_article_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(html_audit, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(html_audit, "RESULTS_DIR", tmp_path / "results")

    def unavailable(_session, _site):
        raise requests.ConnectionError("fixture")

    monkeypatch.setattr(html_audit._PilotSession, "_read_robots", unavailable)
    result = html_audit.download_pilot(html_audit.select_pilot(_cohort()))
    assert result["attempted_this_run"] == result["recorded_attempts"] == 0
    assert result["stopped"] == {"status": "robots_unavailable", "error_type": "ConnectionError"}
