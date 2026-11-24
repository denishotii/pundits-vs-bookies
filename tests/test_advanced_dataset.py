import pandas as pd
import pytest

from pvb.advanced.dataset import load_football


PUBLISHED = "2021-08-06T13:00:00Z"
KICKOFF = "2021-08-07T15:00:00Z"


def fixture(article_id="a", *, match=None, preview=None):
    game = {
        "article_id": article_id, "season": "2122", "league": "E0",
        "home_team": f"Home {article_id}", "away_team": "Away",
        "kickoff": KICKOFF, "published": PUBLISHED, "result": "A", "pick_result": "H",
        "pick_home_goals": 2, "pick_away_goals": 1,
        "home_goals": 0, "away_goals": 4, "nb_home_vs_draw": 42,
        "early_odds_home": 2.0, "early_odds_draw": 3.0, "early_odds_away": 4.0,
        "closing_odds_home": 2.1, "closing_odds_draw": 3.1, "closing_odds_away": 4.1,
        "best_closing_odds_home": 2.2,
    }
    game.update(match or {})
    article = {
        "article_id": article_id, "published": game["published"], "sport": "football",
        "title": f"Title {article_id}", "text_intro": None, "text_preview": f"Body {article_id}",
        "text_team_news": None, "text_head_to_head": None, "text_verdict": None, "text_other": None,
        "result": "D", "word_count": 100,
    }
    article.update(preview or {})
    return game, article


def write_sources(tmp_path, matches, previews):
    paths = {"matches_path": tmp_path / "matches.csv", "previews_path": tmp_path / "previews.csv"}
    pd.DataFrame(matches).to_csv(paths["matches_path"], index=False)
    pd.DataFrame(previews).to_csv(paths["previews_path"], index=False)
    return paths


def test_join_aligns_rows_and_keeps_only_allowed_features(tmp_path):
    a, article_a = fixture(preview={"published": "2021-08-06T14:00:00+01:00"})
    b, article_b = fixture("b", match={"result": "D", "kickoff": "2021-08-08T12:00:00"})
    _, extra = fixture("extra")
    paths = write_sources(tmp_path, [b, a], [article_a, extra, article_b])

    data = load_football("closing", **paths)

    assert data.metadata["article_id"].tolist() == ["a", "b"]
    assert data.target.tolist() == [2, 1]
    assert data.text["text_preview"].tolist() == ["Body a", "Body b"]
    assert data.text["text_team_news"].tolist() == ["", ""]
    assert data.metadata.loc[0, "published"] == pd.Timestamp(PUBLISHED)
    assert data.metadata.loc[1, "prediction_at"] == pd.Timestamp("2021-08-08T12:00:00Z")
    assert data.metadata.loc[1, "outcome_available_at"] == pd.Timestamp("2021-08-08T15:00:00Z")
    assert list(data.numerical) == [
        "closing_odds_home", "closing_odds_draw", "closing_odds_away",
        "pick_home", "pick_draw", "pick_goal_diff",
    ]
    assert data.numerical.iloc[0].tolist() == [2.1, 3.1, 4.1, 1, 0, 1]
    for frame in (data.metadata, data.text, data.numerical):
        assert frame.index.equals(data.target.index)
        assert {"result", "home_goals", "away_goals", "nb_home_vs_draw"}.isdisjoint(frame.columns)
    assert data.counts["excluded_rows"] == 0


@pytest.mark.parametrize("source", ["matches", "previews"])
def test_duplicate_article_ids_are_rejected(tmp_path, source):
    game, article = fixture()
    matches = [game, {**game, "article_id": " a "}] if source == "matches" else [game]
    previews = [article, article] if source == "previews" else [article]
    with pytest.raises(ValueError, match=rf"{source}.csv: duplicate article_id on 2 rows"):
        load_football("closing", **write_sources(tmp_path, matches, previews))


def test_missing_required_column_and_missing_id_are_rejected(tmp_path):
    game, article = fixture()
    with pytest.raises(ValueError, match="missing required columns.*result"):
        load_football("closing", **write_sources(tmp_path, [
            {key: value for key, value in game.items() if key != "result"},
        ], [article]))
    with pytest.raises(ValueError, match="missing article_id on 1 rows"):
        load_football("closing", **write_sources(tmp_path, [{**game, "article_id": " "}], [article]))


def test_duplicate_fixture_with_distinct_article_ids_is_rejected(tmp_path):
    game, article = fixture()
    same_game, second_article = fixture("b", match={"home_team": " Home a "})
    with pytest.raises(ValueError, match="duplicate football fixture on 2 rows"):
        load_football("closing", **write_sources(tmp_path, [game, same_game], [article, second_article]))


@pytest.mark.parametrize(("match", "preview", "reason"), [
    ({"result": "X"}, {}, "invalid_result"),
    ({"pick_result": None}, {}, "missing_pick"),
    ({"pick_result": "X"}, {}, "invalid_pick"),
    ({"pick_home_goals": 1.5}, {}, "invalid_pick_score"),
    ({"closing_odds_home": None}, {}, "missing_odds"),
    ({"closing_odds_draw": "bad"}, {}, "invalid_odds"),
    ({"closing_odds_away": float("inf")}, {}, "invalid_odds"),
    ({"closing_odds_away": 1}, {}, "invalid_odds"),
    ({"kickoff": "unknown"}, {}, "invalid_kickoff"),
    ({"published": None}, {"published": PUBLISHED}, "invalid_match_publication"),
    ({}, {"published": "unknown"}, "invalid_article_publication"),
    ({}, {"published": "2021-08-06T12:00:00Z"}, "publication_mismatch"),
    ({"published": KICKOFF}, {}, "article_not_before_cutoff"),
    ({}, {"sport": "tennis"}, "wrong_article_sport"),
    ({}, {"text_preview": "   "}, "missing_article_body"),
    ({"season": "2323"}, {}, "invalid_season"),
])
def test_invalid_rows_are_counted(tmp_path, match, preview, reason):
    good, good_article = fixture()
    bad, bad_article = fixture("b", match=match, preview=preview)
    data = load_football("closing", **write_sources(tmp_path, [good, bad], [good_article, bad_article]))
    assert data.metadata["article_id"].tolist() == ["a"]
    assert data.counts[reason] == data.counts["excluded_rows"] == 1


def test_unmatched_articles_and_overlapping_exclusions_are_counted(tmp_path):
    good, good_article = fixture()
    bad, _ = fixture("b", match={"result": None})
    data = load_football("closing", **write_sources(tmp_path, [good, bad], [good_article]))
    assert data.counts["missing_article"] == data.counts["invalid_result"] == 1
    assert data.counts["excluded_rows"] == 1


def test_markets_have_independent_odds_and_prediction_times(tmp_path):
    rows = [
        fixture("a", match={"closing_odds_home": None}),
        fixture("b", match={"early_odds_home": None}),
        fixture("c", match={"published": "2021-08-06T14:00:00Z"}),
        fixture("d", match={"published": "2021-08-06"}),
        fixture("e", match={"kickoff": "2021-08-06T13:30:00Z"}),
        fixture("f", match={"kickoff": "2021-08-11", "published": "2021-08-09"}),
        fixture("g", match={"kickoff": None}),
    ]
    matches, previews = zip(*rows)
    paths = write_sources(tmp_path, matches, previews)

    early = load_football("early", **paths)
    closing = load_football("closing", **paths)

    assert early.metadata["article_id"].tolist() == ["a", "f"]
    assert closing.metadata["article_id"].tolist() == ["e", "b", "c", "d", "f"]
    assert list(early.numerical) == [
        "early_odds_home", "early_odds_draw", "early_odds_away", "pick_home", "pick_draw", "pick_goal_diff",
    ]
    assert early.metadata["prediction_at"].tolist() == [
        pd.Timestamp("2021-08-06T14:00:00Z"), pd.Timestamp("2021-08-10T14:00:00Z"),
    ]
    assert early.metadata.loc[1, "outcome_available_at"] == pd.Timestamp("2021-08-12T03:00:00Z")
    assert early.counts["missing_odds"] == closing.counts["missing_odds"] == 1
    assert early.counts["invalid_kickoff"] == closing.counts["invalid_kickoff"] == 1
    assert early.counts["article_not_before_cutoff"] == 2
    assert early.counts["early_snapshot_not_before_kickoff"] == 1
