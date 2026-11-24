import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pvb.analysis import log_loss
from pvb.advanced.market import (
    CLASS_CODES, decimal_to_implied, decimal_to_probabilities, market_probabilities, validate_probabilities,
)
from pvb.advanced.models import MarketLogistic, MarketResidual, RawMarket, residual_probabilities


def test_implied_probabilities_remove_overround_in_class_order():
    assert CLASS_CODES == ("H", "D", "A")
    odds = np.array([[2, 3, 4], [4, 3, 2]])
    np.testing.assert_allclose(decimal_to_implied(odds), 1 / odds)
    expected = np.array([[6, 4, 3], [3, 4, 6]]) / 13
    np.testing.assert_allclose(decimal_to_probabilities(odds), expected)
    np.testing.assert_allclose(RawMarket().predict_proba(expected), expected)


@pytest.mark.parametrize("odds", [
    [[1, 3, 4]], [[0, 3, 4]], [[-2, 3, 4]], [[np.inf, 3, 4]], [[2, np.nan, 4]], [2, 3, 4], [[2, 3]],
])
def test_invalid_odds_raise(odds):
    with pytest.raises(ValueError):
        decimal_to_probabilities(odds)


@pytest.mark.parametrize("probabilities", [
    [[0.5, 0.3, 0.3]], [[-0.1, 0.5, 0.6]], [[np.nan, 0.5, 0.5]], [[0.0, 1.1, -0.1]], [[1.0]],
])
def test_invalid_probabilities_raise(probabilities):
    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_odds_allowlist_uses_only_the_named_market_in_canonical_order():
    input_columns = {
        "early_odds_draw": 3, "early_odds_away": 4, "early_odds_home": 2,
        "closing_odds_home": 4, "closing_odds_away": 2, "closing_odds_draw": 3,
        "best_closing_odds_home": 1.1, "home_goals": 5, "pick_draw": np.nan,
    }
    numerical = pd.DataFrame([input_columns])
    np.testing.assert_allclose(market_probabilities(numerical, "early"), [[6 / 13, 4 / 13, 3 / 13]])
    np.testing.assert_allclose(market_probabilities(numerical, "closing"), [[3 / 13, 4 / 13, 6 / 13]])
    with pytest.raises(ValueError, match="Missing early odds columns"):
        market_probabilities(numerical.drop(columns="early_odds_home"), "early")


def test_extreme_odds_and_corrections_produce_normalized_probabilities():
    maximum = np.finfo(float).max
    market = decimal_to_probabilities([[1.01, maximum, maximum], [maximum, maximum, maximum]])
    validate_probabilities(market, positive=True)
    np.testing.assert_allclose(market[1], np.ones(3) / 3)
    np.testing.assert_allclose(residual_probabilities(market, np.zeros_like(market)), market)
    np.testing.assert_allclose(residual_probabilities(market, np.full_like(market, maximum)), market)
    prediction = residual_probabilities(market, [[-maximum, maximum, 0], [900, -900, -900]])
    np.testing.assert_allclose(prediction, [[0, 1, 0], [1, 0, 0]])
    validate_probabilities(prediction)
    with pytest.raises(ValueError, match="strictly positive"):
        residual_probabilities(prediction, np.zeros_like(prediction))


def test_residual_optimization_with_dense_and_sparse_features():
    y = np.repeat([0, 1, 2, 0, 1, 2], [68, 22, 10, 10, 22, 68])
    features = np.repeat([-1.0, 1.0], 100).reshape(-1, 1)
    market = np.tile([0.65, 0.2, 0.15], (len(y), 1))
    model = MarketResidual(0.001).fit(market, y, features)
    sparse_model = MarketResidual(0.001).fit(market, y, csr_matrix(features))
    prediction = model.predict_proba(market, features)

    assert model.convergence_.converged and sparse_model.convergence_.converged
    np.testing.assert_allclose(model.intercept_.sum(), 0, atol=1e-12)
    np.testing.assert_allclose(model.coef_.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(prediction, sparse_model.predict_proba(market, csr_matrix(features)), atol=1e-7)
    assert log_loss(prediction, y).mean() < log_loss(market, y).mean() - 0.2
    np.testing.assert_allclose(prediction[[0, -1]], [[0.68, 0.22, 0.10], [0.10, 0.22, 0.68]], atol=0.005)

    new_market = np.tile([0.25, 0.25, 0.5], (len(y), 1))
    updated = model.predict_proba(new_market, features)
    expected = prediction * new_market / market
    np.testing.assert_allclose(updated, expected / expected.sum(axis=1, keepdims=True))
    with pytest.raises(ValueError, match="same correction feature inputs"):
        model.predict_proba(market)


def test_logistic_and_residual_scalers_are_fit_only_on_training_probabilities():
    train = np.array([[0.5, 0.25, 0.25], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    valid = np.array([[0.01, 0.01, 0.98]])
    logistic = MarketLogistic(0.01).fit(train, [0, 1, 2])
    residual = MarketResidual(0.01).fit(train, [0, 1, 2])
    for model, scaler in [(logistic, logistic.pipeline_[0]), (residual, residual.scaler_)]:
        validate_probabilities(model.predict_proba(valid))
        np.testing.assert_allclose(scaler.mean_, np.log(train).mean(axis=0))


@pytest.mark.parametrize("model", [MarketLogistic, MarketResidual])
def test_learned_models_require_all_training_classes(model):
    with pytest.raises(ValueError, match="all three outcomes"):
        model(0.1).fit(np.tile([0.5, 0.25, 0.25], (3, 1)), [0, 1, 1])
