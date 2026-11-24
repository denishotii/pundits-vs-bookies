"""Match-aligned probability scores and calibration for H/D/A forecasts."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..analysis import brier, log_loss
from .market import CLASS_NAMES, N_CLASSES, validate_probabilities, validate_targets

CALIBRATION_BINS = 10


@dataclass
class Evaluation:
    metrics: pd.DataFrame
    classwise: pd.DataFrame
    calibration: pd.DataFrame
    paired_log_loss: pd.DataFrame


def evaluate_predictions(target: pd.Series, predictions: Mapping[str, pd.DataFrame]) -> Evaluation:
    """Evaluate forecasts on exactly the match IDs in target, aligning by ID.

    Target codes and probability columns are in H/D/A order. Log loss uses
    natural logs and the project's 1e-15 floor on the realized probability.
    Brier is the row-wise sum of squared class errors, averaged over matches.
    Accuracy chooses the first class in H/D/A order on a tie. Class metrics
    include recall, precision and log loss conditional on the actual class.
    Calibration uses one-vs-rest reliability in ten equal-width bins and
    returns the count, mean forecast and observed frequency for occupied bins.
    Paired log loss is model minus M0, so negative values favor the model.
    """
    if target.empty or target.index.hasnans or not target.index.is_unique:
        raise ValueError("Target must have nonempty, unique, nonmissing match IDs")
    if "M0" not in predictions:
        raise ValueError("Paired evaluation requires M0 predictions")
    y = validate_targets(target.to_numpy(), len(target))
    losses = {}
    metrics = []
    classwise = []
    calibration = []
    edges = np.arange(CALIBRATION_BINS + 1) / CALIBRATION_BINS

    for model, frame in predictions.items():
        if not frame.index.is_unique or len(frame) != len(target) or not target.index.isin(frame.index).all():
            raise ValueError(f"{model}: predictions must have exactly the target match IDs")
        if tuple(frame.columns) != CLASS_NAMES:
            raise ValueError(f"{model}: expected probability columns {CLASS_NAMES}")
        probabilities = validate_probabilities(frame.reindex(target.index).to_numpy())
        loss = log_loss(probabilities, y)
        predicted = probabilities.argmax(axis=1)
        losses[model] = loss
        metrics.append({
            "model": model, "matches": len(y), "log_loss": loss.mean(),
            "brier": brier(probabilities, y).mean(), "accuracy": (predicted == y).mean(),
        })
        for code in range(N_CLASSES):
            name = CLASS_NAMES[code]
            actual = y == code
            called = predicted == code
            correct = (actual & called).sum()
            p = probabilities[:, code]
            classwise.append({
                "model": model, "class": name, "actual_count": actual.sum(), "predicted_count": called.sum(),
                "recall": correct / actual.sum() if actual.any() else np.nan,
                "precision": correct / called.sum() if called.any() else np.nan,
                "log_loss_given_class": loss[actual].mean() if actual.any() else np.nan,
                "one_vs_rest_brier": np.square(p - actual).mean(),
            })
            bins = np.searchsorted(edges[1:-1], p, side="right")
            for number in np.unique(bins):
                here = bins == number
                calibration.append({
                    "model": model, "class": name, "bin_lower": edges[number], "bin_upper": edges[number + 1],
                    "matches": here.sum(), "mean_probability": p[here].mean(), "observed_frequency": actual[here].mean(),
                })

    paired = pd.DataFrame(losses, index=target.index).sub(losses["M0"], axis="index")
    summary = pd.DataFrame(metrics).set_index("model")
    summary["mean_log_loss_difference_vs_M0"] = paired.mean()
    return Evaluation(summary.reset_index(), pd.DataFrame(classwise), pd.DataFrame(calibration), paired)
