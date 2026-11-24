import numpy as np
import pandas as pd
import pytest

from pvb.advanced.evaluation import evaluate_predictions
from pvb.advanced.market import CLASS_NAMES


@pytest.fixture
def forecasts():
    index = pd.Index(["a", "b"], name="article_id")
    target = pd.Series([0, 1], index=index)
    m0 = pd.DataFrame([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], index=index, columns=CLASS_NAMES)
    m1 = pd.DataFrame([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]], index=index, columns=CLASS_NAMES)
    return target, {"M0": m0, "M1": m1.iloc[::-1]}


def test_metrics_calibration_and_paired_differences_align_by_id(forecasts):
    target, predictions = forecasts
    result = evaluate_predictions(target, predictions)
    m0 = result.metrics.set_index("model").loc["M0"]
    m1 = result.metrics.set_index("model").loc["M1"]
    assert m0.log_loss == pytest.approx(-np.log([0.5, 0.3]).mean())
    assert m0.brier == pytest.approx((0.25 + 0.0625 + 0.0625 + 0.04 + 0.49 + 0.25) / 2)
    assert m0.accuracy == 0.5
    assert m1.accuracy == 1.0
    np.testing.assert_allclose(result.paired_log_loss["M1"], np.log([0.5 / 0.8, 0.3 / 0.6]))
    assert m1.mean_log_loss_difference_vs_M0 == pytest.approx(m1.log_loss - m0.log_loss)

    classwise = result.classwise.set_index(["model", "class"])
    assert classwise.loc[("M0", "draw"), "recall"] == 0
    assert np.isnan(classwise.loc[("M0", "away"), "recall"])
    assert classwise.loc[("M1", "draw"), "log_loss_given_class"] == pytest.approx(-np.log(0.6))
    assert classwise.loc["M0", "one_vs_rest_brier"].sum() == pytest.approx(m0.brier)
    home = result.calibration.query("model == 'M1' and `class` == 'home'").set_index("mean_probability")
    assert home.loc[0.8, "observed_frequency"] == 1
    assert home.loc[0.2, "observed_frequency"] == 0
    draw = result.calibration.query("model == 'M0' and `class` == 'draw'").set_index("mean_probability")
    assert draw.loc[0.3, "bin_lower"] == 0.3
    assert (result.calibration.groupby(["model", "class"])["matches"].sum() == len(target)).all()


@pytest.mark.parametrize("ids", [["x", "a"], ["a", "a"], ["a"]])
def test_predictions_require_identical_unique_match_ids(forecasts, ids):
    target, predictions = forecasts
    predictions["M1"] = predictions["M1"].iloc[:len(ids)].set_axis(ids)
    with pytest.raises(ValueError, match="exactly the target match IDs"):
        evaluate_predictions(target, predictions)


def test_probability_column_order_is_checked(forecasts):
    target, predictions = forecasts
    predictions["M1"] = predictions["M1"][["draw", "home", "away"]]
    with pytest.raises(ValueError, match="expected probability columns"):
        evaluate_predictions(target, predictions)


def test_log_loss_floor_and_probability_tie_conventions():
    target = pd.Series([0, 2], index=["a", "b"])
    forecasts = pd.DataFrame([[0.5, 0.5, 0], [1, 0, 0]], index=target.index, columns=CLASS_NAMES)
    metrics = evaluate_predictions(target, {"M0": forecasts}).metrics.iloc[0]
    assert metrics.accuracy == 0.5
    assert metrics.log_loss == pytest.approx(-np.log([0.5, 1e-15]).mean())
