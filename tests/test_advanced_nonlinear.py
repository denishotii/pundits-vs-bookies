from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pvb.advanced.author_features import AUTHOR_FEATURES, author_features, load_author_keys
from pvb.advanced.dataset import ODDS_COLUMNS
from pvb.advanced.diagnostics import Fold, _reference, _soft_table
from pvb.advanced.nonlinear_experiments import (
    ALPHAS, CATBOOST, _author_fold, _evaluate, blend_probabilities, ordered_probabilities, reduce_side_tfidf,
)


def _author_inputs():
    index = pd.Index(list("abcde"), name="article_id")
    times = pd.to_datetime(["2021-08-01T10:00", "2021-08-01T13:00", "2021-08-01T14:00",
                            "2021-08-01T18:00", "2021-08-02T11:00"], utc=True)
    fixtures = pd.DataFrame({"season": "2122", "kickoff": times, "prediction_at": times,
                             "outcome_available_at": times + pd.Timedelta(hours=3)}, index=index)
    keys = pd.Series(["alice smith", "alice smith", "bob jones", "alice smith", pd.NA], index=index)
    numerical = pd.DataFrame({"pick_home": [1, 0, 0, 0, 1], "pick_draw": [0, 1, 0, 0, 0]}, index=index)
    for market, probabilities in (("early", [0.5, 0.3, 0.2]), ("closing", [0.2, 0.3, 0.5])):
        numerical[list(ODDS_COLUMNS[market])] = np.tile(1 / np.asarray(probabilities), (len(index), 1))
    target = pd.Series([0, 2, 2, 2, 0], index=index)
    return fixtures, keys, numerical, target


def test_author_counts_rates_and_shrinkage():
    features = author_features("early", *_author_inputs())
    assert tuple(features.columns) == AUTHOR_FEATURES
    assert features.loc["a", "author_accuracy"] == pytest.approx(1 / 3)
    assert features.loc["a", "author_excess_accuracy"] == 0
    assert features.loc["b", "author_previous_predictions"] == 0  # A releases exactly at B's cutoff.
    assert features.loc["c", "author_previous_predictions"] == 0
    assert features.loc["c", "author_accuracy"] == 1
    assert features.loc["c", "author_excess_accuracy"] == pytest.approx(0.5)
    d = features.loc["d"]
    assert d["author_previous_predictions"] == 2
    assert d["author_accuracy"] == pytest.approx(21 / 22)
    assert d.filter(like="author_pick_frequency").tolist() == pytest.approx([1 / 22, 1 / 22, 20 / 22])
    assert d["author_accuracy_agree"] == pytest.approx((1 + 20 / 3) / 21)
    assert d["author_accuracy_disagree"] == pytest.approx(20 / 21)
    assert d["author_excess_accuracy"] == pytest.approx((0.5 - 0.3 + 20 * 0.8) / 22)
    assert not d.filter(like="missing").any()
    assert features.loc["e", "author_missing_byline"] == 1
    assert features.loc["e", "author_missing_history"] == 1
    assert features.loc["e", "author_previous_predictions"] == 0


def test_author_history_does_not_change_with_current_or_future_result():
    fixtures, keys, numerical, target = _author_inputs()
    original = author_features("early", fixtures, keys, numerical, target)
    target.loc["d"] = 0
    changed = author_features("early", fixtures, keys, numerical, target)
    pd.testing.assert_frame_equal(original.loc["a":"d"], changed.loc["a":"d"])
    assert original.loc["e", "author_accuracy"] != changed.loc["e", "author_accuracy"]
    shuffled = [4, 1, 3, 0, 2]
    pd.testing.assert_frame_equal(changed.iloc[shuffled], author_features(
        "early", *(frame.iloc[shuffled] for frame in (fixtures, keys, numerical, target)),
    ))


def test_author_history_is_market_specific_and_ignores_tied_favourites_in_strata():
    fixtures, keys, numerical, target = _author_inputs()
    early = author_features("early", fixtures, keys, numerical, target)
    closing = author_features("closing", fixtures, keys, numerical, target)
    assert closing.loc["d", "author_previous_predictions"] == early.loc["d", "author_previous_predictions"]
    assert closing.loc["d", "author_missing_agree_history"] == 1
    assert early.loc["d", "author_missing_agree_history"] == 0
    assert closing.loc["d", "author_excess_accuracy"] == pytest.approx((0.8 - 0.3 + 20 * 0.5) / 22)
    numerical.loc["a", list(ODDS_COLUMNS["early"])] = [2.5, 2.5, 5]
    tied = author_features("early", fixtures, keys, numerical, target)
    assert tied.loc["d", "author_missing_agree_history"] == 1
    assert tied.loc["d", "author_previous_predictions"] == 2


def test_author_loader_normalizes_safe_variants_and_withholds_ambiguous_keys(tmp_path):
    path = tmp_path / "previews.csv"
    pd.DataFrame({
        "article_id": list("abcd") + ["extra"],
        "author": ["  Áine O’Neill ", "ÁINE  O'NEILL", "Sports Mole", "Jane Doe / Joe Roe", "Another Author"],
        "text_preview": ["private"] * 5,
    }).to_csv(path, index=False, encoding="utf-8")
    keys = load_author_keys(path, pd.Index(list("badce")))
    assert keys.iloc[:2].tolist() == ["áine o'neill", "áine o'neill"]
    assert keys.iloc[2:].isna().all()


@pytest.mark.parametrize("season", ["2526", "2627"])
def test_author_holdout_isolation(season):
    fixtures, keys, numerical, target = _author_inputs()
    fixtures.loc["e", "season"] = season
    inputs = SimpleNamespace(metadata=fixtures, numerical=numerical, target=target)
    with pytest.raises(ValueError, match="development seasons only"):
        author_features("early", fixtures, keys, numerical, target)
    fold = Fold("2223", np.array([0, 1, 2]), np.array([3]), {})
    original = _author_fold("early", inputs, keys, fold)
    inputs.target.loc["e"] = 2
    fixtures.loc["e", ["prediction_at", "kickoff", "outcome_available_at"]] -= pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(original, _author_fold("early", inputs, keys, fold))


def test_author_requires_id_alignment_and_valid_availability():
    fixtures, keys, numerical, target = _author_inputs()
    with pytest.raises(ValueError, match="fixture IDs"):
        author_features("early", fixtures, keys.iloc[::-1], numerical, target)
    fixtures.loc["d", "outcome_available_at"] = fixtures.loc["d", "prediction_at"]
    with pytest.raises(ValueError, match="known timestamps"):
        author_features("early", fixtures, keys, numerical, target)


def test_svd_uses_only_training_terms_and_keeps_side_flags():
    rng = np.random.default_rng(512)
    training = csr_matrix(np.column_stack((rng.normal(size=(40, 36)), np.zeros((40, 2)))))
    valid = rng.normal(size=(5, 38))
    valid[:, -2:] = [1, 0]
    a, b = reduce_side_tfidf(training, csr_matrix(valid))
    valid[:, :-2] *= 10
    c, d = reduce_side_tfidf(training, csr_matrix(valid))
    assert a.shape == (40, 34) and b.shape == (5, 34)
    np.testing.assert_allclose(a, c)
    np.testing.assert_allclose(d[:, :-2], b[:, :-2] * 10)
    np.testing.assert_array_equal(d[:, -2:], valid[:, -2:])


def test_blending_and_class_order():
    market = np.array([[0.4, 0.3, 0.3], [0.3, 0.2, 0.5]])
    nonlinear = ordered_probabilities([2, 0, 1], [[0.7, 0.1, 0.2], [0.1, 0.2, 0.7]])
    np.testing.assert_array_equal(nonlinear, [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
    assert blend_probabilities(market, nonlinear, 0).tobytes() == market.tobytes()
    np.testing.assert_array_equal(blend_probabilities(market, nonlinear, 1), nonlinear)
    for alpha in ALPHAS:
        blend = blend_probabilities(market, nonlinear, alpha)
        np.testing.assert_allclose(blend, (1 - alpha) * market + alpha * nonlinear)
        np.testing.assert_allclose(blend.sum(axis=1), 1)
    with pytest.raises(ValueError, match="alpha"):
        blend_probabilities(market, nonlinear, -0.1)
    with pytest.raises(ValueError, match="codes"):
        ordered_probabilities([2, 0, 0], nonlinear)
    with pytest.raises(ValueError, match="three H/D/A columns"):
        ordered_probabilities([2, 0, 1], np.column_stack((nonlinear, [0, 0])))
    with pytest.raises(ValueError, match="sum to one"):
        ordered_probabilities([0, 1, 2], nonlinear / 2)


def test_catboost_produces_normalized_predictions_for_all_three_classes():
    classifier = pytest.importorskip("catboost").CatBoostClassifier
    rng = np.random.default_rng(44)
    train = rng.normal(size=(60, 3))
    model = classifier(**CATBOOST, iterations=3, depth=2).fit(train, np.tile([0, 1, 2], 20))
    forecast = ordered_probabilities(model.classes_, model.predict_proba(train[:5]))
    assert forecast.shape == (5, 3)
    np.testing.assert_allclose(forecast.sum(axis=1), 1)


def test_rejects_extra_and_wrong_validation_matches_before_scoring():
    fixtures, _, _, target = _author_inputs()
    data = SimpleNamespace(metadata=fixtures, target=target)
    folds = [Fold("2223", np.array([0]), np.array([1, 2, 3]), {})]
    raw = _reference(folds, target.index, lambda _: np.full((3, 3), 1 / 3))
    extra = pd.concat([raw, raw.iloc[[0]].set_axis(["a"])])
    with pytest.raises(ValueError, match="same validation match IDs"):
        _evaluate(data, folds, {"M0": raw, "cat_A": extra}, {})
    wrong = raw.set_axis(pd.Index(list("abc")))
    with pytest.raises(ValueError, match="same validation match IDs"):
        _soft_table(folds, target.index, np.full((5, 3), 1 / 3), {"raw": raw, "other": wrong}, "raw")
