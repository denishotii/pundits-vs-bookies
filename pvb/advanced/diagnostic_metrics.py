"""Scoring, paired uncertainty and synthetic controls for Phase 06A."""

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import xlogy
from sklearn.metrics import confusion_matrix, f1_score

from ..analysis import log_loss
from .evaluation import evaluate_predictions
from .market import CLASS_NAMES, N_CLASSES, validate_probabilities
from .models import MarketResidual, _residual_objective, residual_probabilities

BOOTSTRAP_SEED = 60108
BOOTSTRAP_REPLICATES = 3000
MOVEMENT_THRESHOLD = 0.01
SYNTHETIC_SEEDS = (11, 23, 47, 83, 101)
SYNTHETIC_PENALTIES = (0.0001, 0.03, 1.0)


def categorical_metrics(target, forecasts, reference):
    """Reuse the ID-aligned outcome scores with an experiment-specific reference."""
    inputs = {("M0" if name == reference else name): values for name, values in forecasts.items()}
    evaluation = evaluate_predictions(target, inputs)
    scores = evaluation.metrics.replace({"model": {"M0": reference}}).rename(
        columns={"mean_log_loss_difference_vs_M0": "delta_log_loss"},
    )
    classwise = evaluation.classwise.replace({"model": {"M0": reference}})
    classes = np.arange(N_CLASSES)
    confusions = []
    for row in scores.itertuples(index=True):
        prediction = forecasts[row.model].loc[target.index].to_numpy().argmax(axis=1)
        scores.loc[row.Index, "macro_f1"] = f1_score(
            target, prediction, labels=classes, average="macro", zero_division=0,
        )
        counts = confusion_matrix(target, prediction, labels=classes)
        for i, actual in enumerate(CLASS_NAMES):
            for j, called in enumerate(CLASS_NAMES):
                confusions.append({"model": row.model, "actual": actual, "predicted": called, "matches": counts[i, j]})
    return scores, classwise, pd.DataFrame(confusions)


def soft_cross_entropy(target, prediction):
    """Per-row cross-entropy in nats, including rows with zero target mass in a class."""
    return -xlogy(target, np.clip(prediction, 1e-15, 1)).sum(axis=1)


def _column_correlations(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    norms = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    # Fold means are exactly constant in the model but centering can leave roundoff.
    return np.divide((x * y).sum(axis=0), norms, out=np.full(x.shape[1], np.nan), where=norms > 1e-14)


def soft_metrics(target, prediction, baseline, *, early=None):
    target, prediction, baseline = (validate_probabilities(values) for values in (target, prediction, baseline))
    if target.shape != prediction.shape or target.shape != baseline.shape:
        raise ValueError("Soft-target forecasts and baseline must have identical shapes")
    cross_entropy = soft_cross_entropy(target, prediction)
    kl = float((cross_entropy - soft_cross_entropy(target, target)).mean())
    baseline_kl = float((soft_cross_entropy(target, baseline) - soft_cross_entropy(target, target)).mean())
    delta = baseline_kl - kl
    error = prediction - target
    result = {
        "matches": len(target), "cross_entropy": cross_entropy.mean(), "kl": kl,
        "mae": np.abs(error).mean(), "rmse": np.sqrt(np.square(error).mean()),
        "kl_improvement": delta, "kl_improvement_fraction": delta / baseline_kl if baseline_kl > 0 else np.nan,
    }
    if early is None:
        result.update(zip((f"correlation_{name}" for name in CLASS_NAMES), _column_correlations(prediction, target)))
    else:
        early = validate_probabilities(early)
        if early.shape != target.shape:
            raise ValueError("Early offsets must align with closing targets")
        observed, forecast = target - early, prediction - early
        result.update(zip((f"change_correlation_{name}" for name in CLASS_NAMES),
                          _column_correlations(forecast, observed)))
        material = np.abs(observed) >= MOVEMENT_THRESHOLD - 1e-12
        correct = np.sign(forecast) == np.sign(observed)
        result["material_changes"] = int(material.sum())
        result["directional_accuracy"] = correct[material].mean() if material.any() else np.nan
        for i, side in enumerate(CLASS_NAMES):
            here = material[:, i]
            result[f"material_changes_{side}"] = int(here.sum())
            result[f"directional_accuracy_{side}"] = correct[here, i].mean() if here.any() else np.nan
    return result


def paired_summary(differences):
    differences = np.asarray(differences, dtype=float)
    quantiles = np.quantile(differences, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "matches": len(differences), "mean": differences.mean(), "sd": differences.std(ddof=1),
        "p05": quantiles[0], "p25": quantiles[1], "p50": quantiles[2], "p75": quantiles[3], "p95": quantiles[4],
        "fraction_improved": (differences < 0).mean(),
    }


def date_block_bootstrap(paired, metadata, block_days, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Resample occupied UTC fixture-date blocks within seasons, retaining season weights."""
    if not paired.index.is_unique or not metadata.index.is_unique or set(paired.index) != set(metadata.index):
        raise ValueError("Paired differences and fixture dates must align by unique match ID")
    if block_days < 1 or replicates < 2:
        raise ValueError("Positive block length and at least two bootstrap replicates are required")
    aligned = metadata.loc[paired.index, ["season", "kickoff"]]
    origin = pd.Timestamp("1970-01-05", tz="UTC")  # Monday
    blocks = (aligned["kickoff"].dt.floor("D") - origin).dt.days // block_days
    groups = paired.assign(season=aligned["season"], block=blocks).groupby(["season", "block"], sort=True)
    counts = groups.size()
    sums = groups[paired.columns.tolist()].sum()
    rng = np.random.default_rng(seed)
    simulated = np.zeros((replicates, paired.shape[1]))
    min_blocks = len(counts)
    for season, season_counts in counts.groupby(level="season"):
        n = len(season_counts)
        draw = rng.integers(n, size=(replicates, n))
        total = season_counts.to_numpy()[draw].sum(axis=1)
        means = sums.loc[season].to_numpy()[draw].sum(axis=1) / total[:, None]
        simulated += (season_counts.sum() / len(paired)) * means
        min_blocks = min(min_blocks, n)
    lo, hi = np.quantile(simulated, [0.025, 0.975], axis=0)
    se = simulated.std(axis=0, ddof=1)
    normal = NormalDist()
    detectable = (normal.inv_cdf(0.975) + normal.inv_cdf(0.8)) * se
    return pd.DataFrame({
        "comparison": paired.columns, "block_days": block_days, "occupied_blocks": len(counts),
        "min_blocks_per_season": min_blocks, "interval_low": lo, "interval_high": hi, "bootstrap_se": se,
        "conditional_detectable_delta": detectable,
    })


def gradient_checks():
    """Check the actual optimizer callback away from its optimum."""
    rng = np.random.default_rng(60101)
    market = rng.dirichlet([3, 2, 3], size=9)
    features = rng.normal(size=(9, 4))
    parameters = rng.normal(scale=0.3, size=(features.shape[1] + 1) * (N_CLASSES - 1))
    step = 1e-6
    offsets = np.eye(len(parameters)) * step
    targets = {"hard": np.tile(np.arange(3), 3), "soft": rng.dirichlet([3, 3, 3], size=9)}
    penalties = {"single": 0.03, "block": np.array([0.03, 0.03, 0.17, 0.17])[:, None]}
    rows = []
    for matrix_name, matrix in (("dense", features), ("csr", sparse.csr_matrix(features))):
        for target_name, target in targets.items():
            for penalty_name, penalty in penalties.items():
                def objective(values):
                    return _residual_objective(values, matrix, np.log(market), target, penalty)

                _, analytical = objective(parameters)
                numerical = np.array([
                    (objective(parameters + shift)[0] - objective(parameters - shift)[0]) / (2 * step)
                    for shift in offsets
                ])
                rows.append({
                    "matrix": matrix_name, "target": target_name, "penalty": penalty_name,
                    "step": step, "max_abs_error": np.abs(analytical - numerical).max(),
                    "passed": bool(np.allclose(analytical, numerical, atol=1e-8, rtol=1e-6)),
                })
    return rows


def synthetic_controls():
    """Simulate known probabilities, fit observed labels, score an untouched synthetic sample."""
    rows = []
    n_train, n_valid, n_noise = 2500, 5000, 96
    train, valid = slice(n_train), slice(n_train, None)
    for seed in SYNTHETIC_SEEDS:
        rng = np.random.default_rng(seed)
        market = rng.dirichlet([4, 2, 3], size=n_train + n_valid)
        signal = rng.normal(size=len(market))
        noise = rng.normal(size=(len(market), n_noise))
        oracle = residual_probabilities(market, signal[:, None] * np.array([0.7, 0, -0.7]))
        uniform = rng.random(len(market))
        for case, truth, block in (
            ("A_market_and_noise", market, noise),
            ("B_omitted_signal", oracle, signal[:, None]),
            ("C_signal_among_noise", oracle, np.column_stack((signal, noise))),
        ):
            y = (uniform[:, None] > truth[:, :-1].cumsum(axis=1)).sum(axis=1)
            matrix = np.column_stack((np.log(market), block))
            raw_loss = log_loss(market[valid], y[valid]).mean()
            raw_expected = soft_cross_entropy(truth[valid], market[valid]).mean()
            for l2 in SYNTHETIC_PENALTIES:
                model = MarketResidual(0.1, text_l2=l2).fit(market[train], y[train], matrix[train], numerical_width=3)
                prediction = model.predict_proba(market[valid], matrix[valid])
                loss = log_loss(prediction, y[valid]).mean()
                expected = soft_cross_entropy(truth[valid], prediction).mean()
                rows.append({
                    "case": case, "seed": seed, "block_l2": l2, "n_train": n_train, "n_valid": n_valid,
                    "noise_features": 0 if case.startswith("B") else n_noise,
                    "log_loss": loss, "market_log_loss": raw_loss, "delta_log_loss": loss - raw_loss,
                    "expected_delta_log_loss": expected - raw_expected,
                    "oracle_delta_log_loss": soft_cross_entropy(truth[valid], truth[valid]).mean() - raw_expected,
                    "noise_coef_norm": np.linalg.norm(model.coef_[(3 if case.startswith("A") else 4):]),
                    "signal_home_minus_away_coef": (
                        np.nan if case.startswith("A") else model.coef_[3, 0] - model.coef_[3, 2]
                    ),
                    "converged": model.convergence_.converged,
                })
    return pd.DataFrame(rows)
