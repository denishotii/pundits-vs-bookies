"""Tests for the small helpers: URL filtering, sitemaps, odds, text counts, the CLI."""

import argparse

import pandas as pd
import pytest

from pvb import config, sportsmole
from pvb.__main__ import season_code, year_range
from pvb.features import count_terms
from pvb.matching import normalise_name
from pvb.odds import implied_probabilities
from pvb.sportsmole import classify, download_previews, parse_sitemap

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.sportsmole.co.uk/football/arsenal/preview/arsenal-vs-chelsea-prediction-team-news-lineups_600001.html</loc></url>
  <url><loc>https://www.sportsmole.co.uk/tennis/preview/sinner-vs-alcaraz-prediction-form-head-to-head_600002.html</loc></url>
  <url><loc>https://www.sportsmole.co.uk/football/arsenal/news/arsenal-sign-a-striker_600003.html</loc></url>
</urlset>"""


def test_sitemap_and_classify():
    urls = parse_sitemap(SITEMAP)
    assert len(urls) == 3
    assert [classify(u) for u in urls] == [("football", 600001), ("tennis", 600002), None]


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


class FakeSession:
    """Answers every URL with a fixed page, so no internet is needed."""

    def __init__(self, page: bytes):
        self.page = page

    def get(self, url):
        return FakeResponse(self.page)


@pytest.fixture
def tmp_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPORTSMOLE_RAW", tmp_path)
    monkeypatch.setattr(sportsmole, "MANIFEST_CSV", tmp_path / "manifest.csv")
    return tmp_path


def test_download_saves_only_real_previews(tmp_raw):
    urls = pd.DataFrame({"url": ["https://www.sportsmole.co.uk/x_1.html"], "article_id": [1]})
    assert download_previews(FakeSession(b"<html>Just a moment...</html>"), urls) == 0
    assert not (tmp_raw / "1.html").exists()
    assert download_previews(FakeSession(b'<div class="sm-article-body"><p>Hi</p></div>'), urls) == 1
    assert (tmp_raw / "1.html").exists()
    manifest = pd.read_csv(tmp_raw / "manifest.csv")
    assert list(manifest["status"]) == ["no article body", "saved"]


def test_download_with_nothing_to_do(tmp_raw):
    empty = pd.DataFrame({"url": pd.Series(dtype=str), "article_id": pd.Series(dtype=int)})
    assert download_previews(FakeSession(b""), empty) == 0


def test_implied_probabilities_remove_margin():
    probs = implied_probabilities(2.0, 3.5, 4.0)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.4828, abs=1e-4)


def test_count_terms_whole_words_only():
    text = "Saka is injured; the injury list grows. Uninjured players train. He is ruled out."
    assert count_terms(text, ["injured", "injury", "ruled out"]) == 3
    assert count_terms("", ["injured"]) == 0


def test_normalise_name():
    assert normalise_name("Atlético Madrid") == "atletico madrid"
    assert normalise_name("Nott'm Forest") == "nott m forest"


def test_year_range():
    assert year_range("2025") == [2025]
    assert year_range("2021-2023") == [2021, 2022, 2023]
    assert year_range("2021–2022") == [2021, 2022]  # en dash, as Word writes it
    with pytest.raises(argparse.ArgumentTypeError):
        year_range("2026-2021")


def test_season_code():
    assert season_code("2526") == "2526"
    with pytest.raises(argparse.ArgumentTypeError):
        season_code("2025")
