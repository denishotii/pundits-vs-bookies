from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pvb.advanced import team_text
from pvb.advanced.diagnostics import Fold, _reference
from pvb.advanced.historical_features import (
    HISTORY_COLUMNS, INITIAL_ELO, elo_change, historical_features, historical_fold, load_history,
)
from pvb.advanced.market import CLASS_NAMES
from pvb.advanced.models import MarketResidual
from pvb.advanced.preprocessing import PRIMARY_TEXT_COLUMNS
from pvb.advanced.team_historical import categorical_scopes, verdict_picks
from pvb.advanced.team_text import TeamAttribution, comparison_features, side_documents, side_tfidf, team_embeddings
from test_advanced_contextual import MockEncoder


def test_attribution_orients_article_sides_and_masks_names():
    rule = TeamAttribution("Man City", "Atletico Madrid", "Atlético Madrid", "Manchester City")
    assert rule.orientation == "reversed"
    assert rule.proposal("Manchester City scored twice.")[0] == "home"
    assert rule.proposal("Atlético Madrid kept a clean sheet.")[0] == "away"
    assert rule.mask("Atle\u0301tico Madrid's shape worked.") == "TEAM's shape worked."
    assert rule.proposal("The hosts have won twice.")[0] == "home"
    assert rule.proposal("They have won twice.") == ("neither", False, True, False)
    assert rule.proposal("The visitors Manchester City won.") == ("both", False, False, True)


def test_attribution_excludes_ambiguous_and_two_team_sentences():
    rule = TeamAttribution("Milan", "Inter", "AC Milan", "Inter Milan")
    assert rule.proposal("Inter Milan have held their shape.")[0] == "away"
    assert rule.mask("Inter Milan have held their shape.") == "TEAM have held their shape."
    assert rule.proposal("Milan kept the ball.")[:2] == ("neither", True)
    assert rule.proposal("The hosts still face Milan.")[:2] == ("neither", True)
    assert rule.proposal("AC Milan shared the points with Inter Milan.")[0] == "both"
    assert rule.proposal("The AC Milanese coach was watching.")[0] == "neither"
    assert rule.proposal("AC Milan have trained ahead of the game.")[0] == "home"


def _team_sections():
    index = pd.Index(["fixture-a", "fixture-b"], name="article_id")
    articles = pd.DataFrame({
        "home_team": ["Milan", "Milan"], "away_team": ["Inter", "Inter"],
        "source_side_a": ["AC Milan", "AC Milan"], "source_side_b": ["Inter Milan", "Inter Milan"],
        "matched_side_a": ["AC Milan", "AC Milan"], "matched_side_b": ["Inter Milan", "Inter Milan"],
    }, index=index)
    text = pd.DataFrame("", index=index, columns=PRIMARY_TEXT_COLUMNS)
    text["text_preview"] = ["AC Milan won. They celebrated. AC Milan met Inter Milan.", "A neutral preview."]
    return articles, text


def test_side_documents_retain_missing_rows_and_exclude_both_teams(tmp_path):
    articles, text = _team_sections()
    path = tmp_path / "private" / "attribution.json.gz"
    sides = side_documents(articles, text, path)
    assert sides.home.index.equals(articles.index) and sides.away.index.equals(articles.index)
    assert sides.home["text_preview"].tolist() == ["TEAM won.", ""]
    assert sides.away["text_preview"].tolist() == ["", ""]
    assert sides.coverage.loc["fixture-a", "sentences_both"] == 1
    assert sides.coverage.loc["fixture-a", "unresolved"] == 1
    assert not sides.cache_hit and side_documents(articles, text, path).cache_hit
    changed = text.copy()
    changed.loc["fixture-a", "text_preview"] = "Inter Milan won."
    assert side_documents(articles, changed, path).fingerprint != sides.fingerprint


def test_private_review_sampling_preserves_human_labels(tmp_path):
    articles, _ = _team_sections()
    prior = pd.DataFrame({
        "article_id": ["fixture-a"] * 4, "season": ["2122"] * 4, "section": ["text_preview"] * 4,
        "sentence": ["AC Milan won.", "Inter Milan lost.", "AC Milan drew with Inter Milan.", "They drew."],
        "human_label": [""] * 4,
    })
    source, output = tmp_path / "phase06a.csv", tmp_path / "private" / "review.csv"
    prior.to_csv(source, index=False)
    first = team_text.prepare_review(source, output, articles, per_label=1)
    assert first["sample_rows"] == 4 and first["recognized_human_labels"] == 0
    review = pd.read_csv(output, keep_default_na=False)
    review.loc[review["phase06b_proposal"].eq("home"), "human_label"] = "away"
    review.to_csv(output, index=False)
    rerun = team_text.prepare_review(source, output, articles, per_label=1)
    assert rerun["recognized_human_labels"] == 1
    assert rerun["proposal_agreement_on_labeled_subset"] == 0.0
    assert pd.read_csv(output)["review_id"].tolist() == review["review_id"].tolist()


@pytest.mark.parametrize("use_sparse", [False, True])
def test_swapping_sides_reverses_difference_and_missing_flags(use_sparse):
    home = np.array([[1., 3.], [0., 0.]])
    away = np.array([[0., 0.], [3., -2.]])
    missing = np.array([[0, 1], [1, 0]])
    if use_sparse:
        home, away = sparse.csr_matrix(home), sparse.csr_matrix(away)
    features = comparison_features(home, away, missing)
    swapped = comparison_features(away, home, missing[:, ::-1])
    if use_sparse:
        assert sparse.isspmatrix_csr(features)
        features, swapped = features.toarray(), swapped.toarray()
    np.testing.assert_array_equal(swapped[:, :2], -features[:, :2])
    np.testing.assert_array_equal(swapped[:, 2:4], features[:, 2:4])
    np.testing.assert_array_equal(swapped[:, -2:], features[:, -2:][:, ::-1])


def test_shared_tfidf_is_fit_on_training_documents_only():
    home = pd.Series(["pressure pace"] * 5 + ["reserve"] * 5 + ["validationtoken", ""])
    away = pd.Series(["shape"] * 5 + ["tackle"] * 5 + ["anotherfuture", ""])
    train, valid = np.arange(10), np.arange(10, 12)
    vectorizer, training, validation = side_tfidf((home, away), train, valid)
    assert {"pressure", "shape"} <= vectorizer.vocabulary_.keys()
    assert {"validationtoken", "anotherfuture"}.isdisjoint(vectorizer.vocabulary_)
    np.testing.assert_array_equal(validation.toarray()[:, :-2], 0)
    np.testing.assert_array_equal(validation.toarray()[:, -2:], [[0, 0], [1, 1]])
    assert training.shape == (10, 2 * len(vectorizer.vocabulary_) + 2)


def test_team_embedding_cache_tracks_content_and_attribution_configuration(tmp_path, monkeypatch):
    articles, text = _team_sections()
    sections_path = tmp_path / "attribution.json.gz"
    path = tmp_path / "team.sqlite3"
    sides = side_documents(articles, text, sections_path)
    encoder = MockEncoder()
    first, stats = team_embeddings(sides, encoder, path)
    assert stats["cache_misses"] == 4
    np.testing.assert_array_equal(first[:, -2:], [[0, 1], [1, 1]])
    np.testing.assert_array_equal(first[1, :-2], 0)
    second, stats = team_embeddings(sides, encoder, path)
    assert stats["cache_hits"] == 4
    np.testing.assert_array_equal(first, second)
    sides.home.loc["fixture-a", "text_preview"] = "TEAM won twice."
    _, stats = team_embeddings(sides, encoder, path)
    assert stats["cache_misses"] == 1
    monkeypatch.setitem(team_text.ATTRIBUTION_CONFIG, "schema", "test")
    assert not side_documents(articles, text, sections_path).cache_hit
    _, stats = team_embeddings(sides, encoder, path)
    assert stats["cache_misses"] == 4


def _results(*rows):
    history = pd.DataFrame(rows, columns=[*HISTORY_COLUMNS, "result_available_at"])
    for name in ("kickoff", "result_available_at"):
        history[name] = pd.to_datetime(history[name], utc=True)
    return history


def _prediction(cutoff="2021-08-03 16:00", home="Alpha", away="Gamma", season="2122", league="E0"):
    stamp = pd.Timestamp(cutoff, tz="UTC")
    return {"season": season, "league": league, "kickoff": stamp, "prediction_at": stamp,
            "home_team": home, "away_team": away}


def test_elo_and_rolling_stats_follow_observed_results_and_venue():
    history = _results(
        ("2122", "E0", "Alpha", "Beta", "2021-08-01 12:00", 2, 0, "2021-08-01 15:00"),
        ("2122", "E0", "Gamma", "Alpha", "2021-08-02 12:00", 2, 2, "2021-08-02 15:00"),
    )
    predictions = pd.DataFrame([_prediction()], index=["fixture"])
    row = historical_features(predictions, history).iloc[0]
    change = elo_change(1500, 1500, 2, 0)
    draw_change = elo_change(1500, 1500 + change, 2, 2)
    assert change == pytest.approx(20 * (1 - 1 / (1 + 10 ** (-60 / 400))))
    assert row["home_elo"] == pytest.approx(1500 + change - draw_change)
    assert row["away_elo"] == pytest.approx(1500 + draw_change)
    assert row["elo_difference"] == pytest.approx(change - 2 * draw_change)
    assert row["home_observed_matches"] == 2 and row["home_observed_venue_matches"] == 1
    assert row["home_points_last_5"] == 2 and row["home_goals_for_last_5"] == 2
    assert row["home_goals_against_last_10"] == 1 and row["home_goal_difference_last_10"] == 1
    assert row["home_venue_points_per_match"] == 3 and np.isnan(row["away_venue_points_per_match"])
    assert row["home_days_since_known_match"] == pytest.approx(28 / 24)


def test_rolling_windows_use_kickoff_order_and_different_lengths():
    kickoff = pd.date_range("2021-08-01 12:00", periods=11, tz="UTC")
    history = _results(*[
        ("2122", "E0", "Alpha", f"Club {i}", date, i, 0, date + pd.Timedelta(hours=(72 if i == 7 else 3)))
        for i, date in enumerate(kickoff)
    ])
    row = historical_features(pd.DataFrame([_prediction("2021-08-13")]), history).iloc[0]
    assert row["home_observed_matches"] == 11
    assert row["home_goals_for_last_5"] == 8
    assert row["home_goals_for_last_10"] == 5.5


def test_strict_availability_and_simultaneous_result_batches():
    history = _results(
        ("2122", "E0", "Alpha", "Beta", "2021-08-01 12:00", 1, 0, "2021-08-01 18:00"),
        ("2122", "E0", "Gamma", "Alpha", "2021-08-01 15:00", 0, 0, "2021-08-01 18:00"),
    )
    predictions = pd.DataFrame([
        _prediction("2021-08-01 18:01"), _prediction("2021-08-01 18:00"), _prediction("2021-08-01 16:00"),
        _prediction("2021-08-01 18:00", "Beta", "Gamma"),
    ], index=["after", "exact", "earlier", "simultaneous"])
    features = historical_features(predictions, history)
    assert features.index.equals(predictions.index)
    assert features.loc["after", "home_observed_matches"] == 2
    assert features.loc["after", "home_elo"] == pytest.approx(
        INITIAL_ELO + elo_change(1500, 1500, 1, 0) - elo_change(1500, 1500, 0, 0),
    )
    np.testing.assert_array_equal(features.loc[["exact", "earlier", "simultaneous"], "home_observed_matches"], 0)
    pd.testing.assert_frame_equal(features, historical_features(predictions, history.iloc[::-1]))


def test_season_transitions_carry_promoted_teams_and_initialize_unseen_clubs():
    history = _results(("2122", "E1", "Alpha", "Beta", "2022-05-01 12:00", 1, 0, "2022-05-01 15:00"))
    row = historical_features(pd.DataFrame([_prediction("2022-08-01", season="2223")]), history).iloc[0]
    assert row["home_observed_matches"] == 1 and row["home_elo"] > INITIAL_ELO
    assert row["away_observed_matches"] == 0 and row["away_elo"] == INITIAL_ELO
    assert np.isnan(row["away_days_since_known_match"])


def test_current_and_future_results_cannot_change_an_earlier_feature():
    history = _results(
        ("2122", "E0", "Alpha", "Beta", "2021-08-01 12:00", 1, 0, "2021-08-01 15:00"),
        ("2122", "E0", "Alpha", "Gamma", "2021-08-03 16:00", 1, 0, "2021-08-03 19:00"),
        ("2122", "E0", "Beta", "Gamma", "2021-08-07 12:00", 0, 1, "2021-08-07 15:00"),
    )
    predictions = pd.DataFrame([_prediction(), _prediction("2021-08-05", away="Delta")])
    original = historical_features(predictions, history)
    history.loc[2, "home_goals"] = 6
    pd.testing.assert_frame_equal(original, historical_features(predictions, history))
    history.loc[1, "home_goals"] = 0
    changed = historical_features(predictions, history)
    pd.testing.assert_series_equal(original.iloc[0], changed.iloc[0])
    assert original.iloc[1]["home_elo"] != changed.iloc[1]["home_elo"]
    history.loc[1, "kickoff"] -= pd.Timedelta(days=1)
    history.loc[1, "result_available_at"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="current fixture"):
        historical_features(predictions, history)


def test_history_load_filters_holdouts_and_uses_local_results_with_date_only_buffer(tmp_path):
    matches = tmp_path / "matches.csv"
    pd.DataFrame([
        ("2122", "E0", "Alpha", "Beta", "2021-08-02T00:00:00Z", 2, 0),
        ("2526", "E0", "Alpha", "Beta", "2021-08-03T00:00:00Z", 49, 0),
    ], columns=HISTORY_COLUMNS).to_csv(matches, index=False)
    pd.DataFrame({
        "Date": ["02/08/2021", "03/08/2021"], "HomeTeam": ["Alpha", "Gamma"],
        "AwayTeam": ["Beta", "Alpha"], "FTHG": [2, 1], "FTAG": [0, 1],
    }).to_csv(tmp_path / "2122_E0.csv", index=False)
    (tmp_path / "2526_E0.csv").write_text("holdout file should never be opened", encoding="utf-8")
    history = load_history(matches, tmp_path)
    assert history["season"].unique().tolist() == ["2122"]
    assert history["source"].tolist() == ["football_data", "football_data"]
    assert history["result_available_at"].iloc[0] == pd.Timestamp("2021-08-03T03:00:00Z")
    original = historical_features(pd.DataFrame([_prediction()]), history)
    holdout = history.iloc[[0]].assign(season="2526", home_team="Beta", home_goals=50)
    pd.testing.assert_frame_equal(original, historical_features(pd.DataFrame([_prediction()]),
                                                                 pd.concat([history, holdout])))
    with pytest.raises(ValueError, match="development predictions only"):
        historical_features(pd.DataFrame([_prediction(season="2526")]), history)


def test_history_checks_overlapping_kickoffs_and_keeps_agreed_precision(tmp_path):
    matches = tmp_path / "matched.csv"
    pd.DataFrame([("2122", "E0", "Alpha", "Beta", "2021-08-02T14:00:00Z", 2, 0)],
                 columns=HISTORY_COLUMNS).to_csv(matches, index=False)
    source = tmp_path / "2122_E0.csv"
    games = pd.DataFrame({"Date": ["02/08/2021"], "HomeTeam": ["Alpha"], "AwayTeam": ["Beta"],
                          "FTHG": [2], "FTAG": [0]})
    games.to_csv(source, index=False)
    history = load_history(matches, tmp_path)
    assert history["result_available_at"].iloc[0] == pd.Timestamp("2021-08-02T17:00:00Z")
    games.assign(Time="12:00").to_csv(source, index=False)
    with pytest.raises(ValueError, match="disagree on observed kickoffs"):
        load_history(matches, tmp_path)


def test_historical_fold_freezes_imputation_and_scaling():
    history = _results(("2122", "E0", "Alpha", "Beta", "2021-08-01 12:00", 1, 0, "2021-08-01 15:00"))
    predictions = pd.DataFrame([_prediction("2021-08-01"), _prediction("2021-08-02"), _prediction()])
    features = historical_features(predictions, history)
    train, valid = np.array([0, 1]), np.array([2])
    a, b = historical_fold(features, train, valid)
    features.loc[2, "home_points_last_5"] = 900
    c, d = historical_fold(features, train, valid)
    np.testing.assert_array_equal(a, c)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.array_equal(b, d)


def test_forecast_comparisons_require_identical_validation_cohorts():
    index = pd.Index(list("abcd"))
    folds = [Fold("2223", np.array([0]), np.array([1, 2, 3]), {})]
    inputs = SimpleNamespace(folds=folds)
    truth = pd.Series([0, 1, 2, 0], index=index)
    reference = _reference(folds, index, lambda _: np.full((3, 3), 1 / 3))
    categorical_scopes(inputs, truth, {"raw": reference, "model": reference.iloc[::-1]}, "raw")
    wrong_ids = reference.set_axis(pd.Index(list("abc")))
    with pytest.raises(ValueError, match="same validation match IDs"):
        categorical_scopes(inputs, truth, {"raw": reference, "model": wrong_ids}, "raw")
    assert tuple(reference.columns) == CLASS_NAMES


def test_verdict_control_is_orientation_checked_and_can_abstain():
    metadata = pd.DataFrame({"home_team": ["Milan"] * 3, "away_team": ["Inter"] * 3})
    headings = pd.Series([
        "We say: AC Milan 2-1 Inter Milan", "We say: Inter Milan 1-2 AC Milan", "We expect a close game",
    ])
    codes = verdict_picks(metadata, headings)
    assert codes.iloc[:2].tolist() == [0, 0] and np.isnan(codes.iloc[2])


def test_three_block_penalties_satisfy_independent_gradient_conditions():
    rng = np.random.default_rng(467)
    market = rng.dirichlet([3, 2, 3], size=60)
    outcomes = np.tile(np.arange(3), 20)
    features = rng.normal(size=(60, 6))
    penalties = ((2, 0.04), (3, 0.2), (1, 0.009))
    model = MarketResidual(1.0, standardize=False).fit(
        market, outcomes, sparse.csr_matrix(features), block_penalties=penalties,
    )
    assert model.convergence_.converged
    errors = model.predict_proba(market, features) - np.eye(3)[outcomes]
    l2 = np.repeat([0.04, 0.2, 0.009], [2, 3, 1])[:, None]
    np.testing.assert_allclose(features.T @ errors / 60 + l2 * model.coef_, 0, atol=1e-6)
