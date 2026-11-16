"""Parser tests on small hand-written pages that copy Sports Mole's layout.

We don't keep real Sports Mole pages in the repo (their terms forbid
republishing), so these fixtures are invented text in the same structure.
"""

from pvb.parse import parse_pick, parse_preview

FOOTBALL_URL = "https://www.sportsmole.co.uk/football/brighton-and-hove-albion/preview/brighton-vs-leeds-prediction-team-news-lineups_604333.html"
FOOTBALL_PAGE = """
<html><head>
<script type="application/ld+json">{"@context": "https://schema.org", "@type": "NewsArticle",
 "datePublished": "2026-09-03T17:00:00+01:00", "author": {"@type": "Person", "name": "A. Writer"}}</script>
</head><body>
<h1>Preview: Brighton &amp; Hove Albion vs Leeds United - prediction, team news, lineups</h1>
<div class="sm-article-body">
  <p>By A. Writer | 03 Sep 2026 17:00</p>
  <p>Brighton host Leeds on Saturday.</p>
  <h2>Match preview</h2>
  <p>Brighton could extend their unbeaten run.</p>
  <div><script>ad()</script></div>
  <h2>Team News</h2>
  <p>Two defenders are injured and one midfielder is suspended.</p>
  <h2>We say: Brighton &amp; Hove Albion 2-2 Leeds United</h2>
  <p>A tight game that might end level.</p>
  <p>For data analysis of the most likely results, click here.</p>
</div></body></html>
"""

TENNIS_URL = "https://www.sportsmole.co.uk/tennis/atp-world-tour-finals/preview/daniil-medvedev-vs-flavio-cobolli-prediction-form-head-to-head_609168.html"
TENNIS_PAGE = """
<html><body>
<h1>Preview: Medvedev vs. Cobolli - prediction, form, head-to-head</h1>
<div class="sm-article-body">
  <h2>Match preview</h2><p>Medvedev returns to Turin.</p>
  <h2>Head To Head</h2><p>Medvedev leads 2-0.</p>
  <h2>We say: Medvedev to win in three sets</h2><p>Expect a long match.</p>
</div></body></html>
"""


def test_football_preview():
    r = parse_preview(FOOTBALL_PAGE, FOOTBALL_URL)
    assert r["article_id"] == 604333 and r["sport"] == "football"
    assert (r["side_a"], r["side_b"]) == ("Brighton & Hove Albion", "Leeds United")
    assert r["published"] == "2026-09-03T17:00:00+01:00" and r["author"] == "A. Writer"
    assert (r["pick_home_goals"], r["pick_away_goals"], r["pick_result"]) == (2, 2, "D")
    assert r["text_team_news"] == "Two defenders are injured and one midfielder is suspended."
    assert r["text_intro"] == "Brighton host Leeds on Saturday."  # byline dropped
    assert "click here" not in r["text_verdict"]  # promo line dropped


def test_tennis_preview():
    r = parse_preview(TENNIS_PAGE, TENNIS_URL)
    assert r["sport"] == "tennis"
    assert (r["side_a"], r["side_b"]) == ("Medvedev", "Cobolli")
    assert (r["pick_winner"], r["pick_sets"]) == ("Medvedev", "three")
    assert r["text_head_to_head"] == "Medvedev leads 2-0."
    assert r["pick_result"] is None


def test_url_is_read_from_the_page_when_not_given():
    page = FOOTBALL_PAGE.replace("<head>", f'<head><link rel="canonical" href="{FOOTBALL_URL}">')
    r = parse_preview(page.encode("utf-8"))  # saved pages are read as bytes
    assert r["url"] == FOOTBALL_URL and r["article_id"] == 604333


def test_author_html_entities_are_decoded():
    page = FOOTBALL_PAGE.replace("A. Writer", "Jon O&#039;Shea")
    assert parse_preview(page, FOOTBALL_URL)["author"] == "Jon O'Shea"


def test_tournament_prefix_is_dropped_from_names():
    page = TENNIS_PAGE.replace("Preview: Medvedev vs. Cobolli", "Preview: ATP Finals: Daniil Medvedev vs. Flavio Cobolli")
    r = parse_preview(page, TENNIS_URL)
    assert (r["side_a"], r["side_b"]) == ("Daniil Medvedev", "Flavio Cobolli")


def test_pick_variants():
    assert parse_pick("We say: Arsenal 1-0 Chelsea", "football")["pick_result"] == "H"
    assert parse_pick("We say: Arsenal 0 - 3 Chelsea", "football")["pick_result"] == "A"
    assert parse_pick("We say: Sinner to win in straight sets", "tennis")["pick_sets"] == "straight"
    assert parse_pick("We say: Sinner to win", "tennis")["pick_winner"] == "Sinner"
    assert parse_pick(None, "football")["pick_result"] is None
