import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from pvb.advanced import diagnostics
from pvb.advanced.diagnostic_data import (
    DEVELOPMENT, SentenceAttribution, load_development, movement_inputs, movement_timing, review_sample, side_orientation,
)
from pvb.advanced.diagnostic_metrics import (
    categorical_metrics, date_block_bootstrap, gradient_checks, paired_summary, soft_metrics, synthetic_controls,
)
from pvb.advanced.embeddings import embed_articles
from pvb.advanced.market import CLASS_NAMES
from pvb.advanced.models import MarketResidual, residual_probabilities
from test_advanced_contextual import MockEncoder
from test_advanced_dataset import fixture, write_sources


def test_residual_gradient_and_soft_label_equivalence():
    checks = gradient_checks()
    assert len(checks) == 8 and all(check["passed"] for check in checks)
    rng = np.random.default_rng(3)
    market = rng.dirichlet([3, 2, 3], size=21)
    features = rng.normal(size=(len(market), 3))
    y = np.tile(np.arange(3), 7)
    hard = MarketResidual(0.02).fit(market, y, features)
    soft = MarketResidual(0.02).fit_soft(market, np.eye(3)[y], features)
    np.testing.assert_allclose(hard.predict_proba(market, features), soft.predict_proba(market, features), atol=1e-9)
    np.testing.assert_allclose(residual_probabilities(market, np.zeros_like(market)), market)
    np.testing.assert_allclose(
        residual_probabilities(market, [1, 0, -1] * np.ones_like(market)),
        softmax(np.log(market) + [1, 0, -1], axis=1),
    )
    with pytest.raises(ValueError, match="align with the market"):
        soft.fit_soft(market, np.eye(3)[:2])
    with pytest.raises(ValueError, match="positive total mass"):
        soft.fit_soft(market, np.tile([1.0, 0, 0], (len(market), 1)))


def test_synthetic_controls_recover_signal_and_shrink_noise():
    with threadpool_limits(limits=1, user_api="blas"):
        controls = synthetic_controls()
    assert controls["converged"].all() and controls["seed"].nunique() == 5
    moderate = controls[controls["block_l2"].eq(0.03)].set_index("case")
    assert (moderate.loc["A_market_and_noise", "expected_delta_log_loss"] > 0).all()
    assert (moderate.loc["B_omitted_signal", "expected_delta_log_loss"] < -0.06).all()
    assert (moderate.loc["C_signal_among_noise", "expected_delta_log_loss"] < -0.04).all()
    noise = controls[controls["case"].eq("A_market_and_noise")].pivot(
        index="seed", columns="block_l2", values="noise_coef_norm",
    )
    assert (noise[1.0] < noise[0.0001]).all()


def test_soft_scores_match_manual_kl_and_movement_threshold():
    early = np.array([[0.50, 0.25, 0.25], [0.25, 0.25, 0.50]])
    target = np.array([[0.51, 0.245, 0.245], [0.23, 0.25, 0.52]])
    perfect = soft_metrics(target, target, early, early=early)
    assert perfect["kl"] == perfect["mae"] == 0
    assert perfect["kl_improvement_fraction"] == 1
    assert perfect["material_changes"] == 3 and perfect["directional_accuracy"] == 1
    no_movement = soft_metrics(target, early, early, early=early)
    assert no_movement["kl"] == pytest.approx((target * np.log(target / early)).sum() / len(target))
    assert no_movement["directional_accuracy"] == 0 and np.isnan(no_movement["change_correlation_home"])
    with pytest.raises(ValueError, match="identical shapes"):
        soft_metrics(target, target[:1], early)


def test_categorical_scoring_aligns_ids_and_uses_correct_hda_confusions():
    truth = pd.Series([0, 2, 1], index=["a", "b", "c"])
    prediction = pd.DataFrame([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
                              index=truth.index, columns=CLASS_NAMES)
    reference = prediction * 0 + 1 / 3
    metrics, classwise, confusions = categorical_metrics(
        truth, {"class_frequency": reference, "diagnostic": prediction.iloc[::-1]}, "class_frequency",
    )
    row = metrics.set_index("model").loc["diagnostic"]
    assert row["log_loss"] == pytest.approx(-np.log([0.8, 0.7, 0.3]).mean())
    assert row["delta_log_loss"] == pytest.approx(row["log_loss"] - np.log(3))
    assert row["accuracy"] == pytest.approx(2 / 3)
    assert len(classwise) == 6
    cells = confusions.set_index(["model", "actual", "predicted"])["matches"]
    assert cells.loc[("diagnostic", "draw", "away")] == 1


def test_date_block_bootstrap_is_paired_stratified_and_deterministic():
    fixtures = pd.DataFrame({
        "kickoff": pd.to_datetime(["2023-01-09", "2023-01-10", "2023-01-16", "2023-06-01", "2023-06-02"], utc=True),
        "season": ["2223"] * 3 + ["2324"] * 2,
    }, index=list("abcde"))
    losses = pd.DataFrame({"variable": [1.0, 1, 3, 8, 8], "constant": [-1.0] * 5}, index=fixtures.index)
    first = date_block_bootstrap(losses, fixtures.iloc[::-1], 7)
    again = date_block_bootstrap(losses, fixtures, 7)
    pd.testing.assert_frame_equal(first, again)
    result = first.set_index("comparison")
    assert result.loc["constant", "interval_low"] == -1
    assert result.loc["constant", "conditional_detectable_delta"] == 0
    assert result.loc["variable", "interval_low"] == pytest.approx(3.8)
    assert result.loc["variable", "interval_high"] == pytest.approx(5)
    assert result.loc["variable", "occupied_blocks"] == 3
    combined = date_block_bootstrap(losses, fixtures, 14).set_index("comparison")
    assert combined.loc["variable", "occupied_blocks"] == 2
    assert paired_summary([-1, 1, -2])["fraction_improved"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="align"):
        date_block_bootstrap(losses.iloc[:-1], fixtures, 7)


def test_movement_timing_is_strict_and_treats_dates_conservatively():
    published = pd.Series(pd.to_datetime([
        "2024-03-07", "2024-03-08", "2024-03-08 11:59", "2024-03-08 12:00",
        "2024-03-08 15:00", "2024-03-08 15:30", "2024-03-09",
    ], format="mixed", utc=True))
    snapshot = pd.Series(pd.Timestamp("2024-03-08 15:00Z"), index=published.index)
    kickoff = pd.Series(pd.Timestamp("2024-03-10 15:00Z"), index=published.index)
    mask, available = movement_timing(published, kickoff, snapshot)
    assert mask.index[mask["pre_snapshot"]].tolist() == [0, 2, 3]
    assert mask.index[mask["pre_snapshot_buffer_3h"]].tolist() == [0, 2]
    assert mask.index[mask["post_snapshot_retrospective"]].tolist() == [5, 6]
    assert mask.index[mask["timing_uncertain"]].tolist() == [1, 4]
    assert available.iloc[6] == pd.Timestamp("2024-03-10 00:00Z")
    invalid, _ = movement_timing(published, snapshot, snapshot)
    assert invalid["snapshot_not_before_kickoff"].all()
    assert not invalid["pre_snapshot"].any()


def test_movement_loads_neither_result_columns_nor_holdouts(tmp_path):
    rows = [
        fixture("before", match={"published": "2021-08-06T10:00:00Z"}),
        fixture("after", match={"published": "2021-08-06T15:00:00Z"}),
        fixture("before_unbuffered"),
        fixture("missing_odds", match={"early_odds_draw": None}),
        fixture("holdout", match={"season": "2526"}),
    ]
    matches, previews = zip(*rows)
    paths = write_sources(tmp_path, [{k: v for k, v in row.items() if k != "result"} for row in matches], previews)
    data = load_development("closing", paths["matches_path"], paths["previews_path"], include_outcomes=False)
    assert data.target is None and "holdout" not in data.metadata.index
    groups, coverage = movement_inputs(data, paths["matches_path"])
    assert set(groups["pre_snapshot"][0].index) == {"before", "before_unbuffered"}
    assert list(groups["pre_snapshot_buffer_3h"][0].index) == ["before"]
    retrospective, early, close = groups["post_snapshot_retrospective"]
    assert list(retrospective.index) == ["after"]
    assert retrospective.iloc[0]["prediction_at"] == retrospective.iloc[0]["published"]
    assert early.shape == close.shape == (1, 3)
    assert coverage.set_index("season").loc["pooled", "invalid_or_missing_early_odds"] == 1


def test_training_only_scaling_and_tfidf_and_recent_training_selection():
    n = 21
    train, valid = np.arange(20), np.array([20])
    values = np.arange(n)[:, None]
    documents = pd.Series(["alpha"] * 10 + ["beta"] * 10 + ["validationonly"])
    fold = diagnostics._make_fold("2425", train, valid, {"picks": values}, {"tfidf": documents}, {})
    x, v = fold.blocks["picks"]
    assert x.mean() == pytest.approx(0)
    assert v[0, 0] == pytest.approx((20 - np.arange(20).mean()) / np.arange(20).std())
    x, v = fold.blocks["tfidf"]
    assert x.shape == (20, 2) and v.nnz == 0
    combined = tuple(fold.features(("picks", "tfidf")))
    assert combined[0].shape == (20, 3) and combined[1].shape == (1, 3)
    dates = pd.date_range("2024-01-01", periods=n)[::-1]
    metadata = pd.DataFrame({"outcome_available_at": dates}, index=[str(i) for i in range(n)])
    assert set(diagnostics.recent_training(metadata, train[::-1], 0.25)) == set(range(5))


def test_full_embedding_target_control_has_separate_cache_identity(tmp_path):
    sections = pd.DataFrame({"text_preview": ["10 20"], "text_verdict": ["100 100 100"]}, index=["article"])
    encoder = MockEncoder()
    cache = tmp_path / "articles.sqlite3"
    primary = embed_articles(sections, encoder, cache)
    full = embed_articles(sections, encoder, cache, representation="full_article")
    assert primary.statistics["total_chunks"] == 1 and full.statistics["total_chunks"] == 2
    assert primary.statistics["cache_misses"] == full.statistics["cache_misses"] == 1
    assert not np.allclose(primary.vectors, full.vectors)
    edited = sections.assign(text_verdict="2 5")
    assert embed_articles(edited, encoder, cache).statistics["cache_hits"] == 1
    assert embed_articles(edited, encoder, cache, representation="full_article").statistics["cache_misses"] == 1


def test_attribution_distinguishes_both_from_neither_and_never_propagates():
    assert side_orientation("Manchester United", "Manchester City", "Man United", "Man City") == "ordered"
    assert side_orientation("Manchester City", "Manchester United", "Man United", "Man City") == "reversed"
    assert side_orientation(pd.NA, "Manchester United", "Man United", "Man City") == "missing"
    attribution = SentenceAttribution("Man United", "Man City", "Manchester United", "Manchester City")
    assert attribution.classify("Manchester United played Manchester City.") == ("both", False, False, False)
    assert attribution.classify("Manchester United scored.") == ("home", False, False, False)
    assert attribution.classify("They were tired.") == ("neither", False, True, False)
    assert attribution.classify("Manchester scored.") == ("neither", True, False, False)
    assert attribution.classify("The visitors rallied.") == ("away", False, False, False)
    assert attribution.classify("The visitors pressured Manchester United.") == ("both", False, False, True)
    milan = SentenceAttribution("Inter", "Milan", "Inter Milan", "AC Milan")
    assert milan.classify("Inter Milan scored.")[0] == "home"


def test_review_sample_is_deterministic_and_stratified():
    seasons = np.repeat(DEVELOPMENT, 25)
    metadata = pd.DataFrame({"season": seasons}, index=[f"article-{i}" for i in range(len(seasons))])
    sample = review_sample(metadata)
    assert sample == review_sample(metadata.iloc[::-1]) and len(sample) == 50
    assert metadata.loc[sorted(sample), "season"].value_counts().sort_index().tolist() == [13, 13, 12, 12]
