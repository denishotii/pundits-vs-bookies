"""Run development-period selection for one football odds market."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn

from .. import config
from ..analysis import log_loss
from .dataset import ODDS_COLUMNS, Market, load_football
from .evaluation import CALIBRATION_BINS, evaluate_predictions
from .market import CLASS_CODES, CLASS_NAMES, market_probabilities
from .models import L2_GRID, MAX_ITER, TOL, MarketLogistic, MarketResidual, RawMarket
from .validation import chronological_folds, season_positions

RESULTS_ROOT = config.ROOT / "results" / "advanced"
MATCHES_PATH = config.PROCESSED / "matched_football.csv"
PREVIEWS_PATH = config.INTERIM / "previews.csv"
MODEL_TYPES = {"M1": MarketLogistic, "M2": MarketResidual}
EVALUATION_ROLE = "development_model_selection"


def _path_label(path: Path) -> str:
    path = path.resolve()
    return (path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path).as_posix()


def run_experiment(
    market: Market,
    results_root: Path = RESULTS_ROOT,
    *,
    matches_path: Path = MATCHES_PATH,
    previews_path: Path = PREVIEWS_PATH,
) -> pd.DataFrame:
    """Tune on expanding development folds and write fold and pooled OOF artifacts.

    The selected candidates' predictions come from the same folds used to
    select L2. The resulting scores are development/model-selection estimates.
    """
    data = load_football(market, matches_path=matches_path, previews_path=previews_path)
    development = season_positions(data.metadata)["development"]
    metadata = data.metadata.iloc[development].set_index("article_id")
    odds = data.numerical.iloc[development].loc[:, list(ODDS_COLUMNS[market])].set_axis(metadata.index)
    target = data.target.iloc[development].set_axis(metadata.index).rename("outcome_code")
    # The Phase 01 loader checks article/pick eligibility. Keep only the allowed
    # odds, target and split metadata from its output.
    del data
    folds, fold_info = chronological_folds(metadata.reset_index())
    probabilities = market_probabilities(odds, market)
    fold_labels = [f"fold_{number}" for number in fold_info["fold"]]
    forecast_folds = {}
    raw = RawMarket()

    def forecast_frame(values, valid_rows):
        return pd.DataFrame(values, index=target.index[valid_rows], columns=CLASS_NAMES)

    forecast_folds["M0"] = [forecast_frame(raw.predict_proba(probabilities[valid]), valid) for _, valid in folds]
    search = []
    selected = {}
    for name, model_type in MODEL_TYPES.items():
        best = (np.inf, 0.0)
        for l2 in L2_GRID:
            predictions = []
            total_loss = 0.0
            total_matches = 0
            converged = True
            for scope, (train, valid) in zip(fold_labels, folds, strict=True):
                model = model_type(l2).fit(probabilities[train], target.iloc[train].to_numpy())
                forecast = model.predict_proba(probabilities[valid])
                predictions.append(forecast_frame(forecast, valid))
                loss = log_loss(forecast, target.iloc[valid].to_numpy())
                total_loss += loss.sum()
                total_matches += len(loss)
                status = model.convergence_
                converged &= status.converged
                search.append({
                    "model": name, "l2": l2, "scope": scope, "matches": len(loss), "log_loss": loss.mean(),
                    "converged": status.converged, "iterations": status.iterations, "optimizer_message": status.message,
                })
            pooled_loss = total_loss / total_matches
            search.append({
                "model": name, "l2": l2, "scope": "pooled", "matches": total_matches,
                "log_loss": pooled_loss, "converged": converged,
            })
            rank = (pooled_loss, -l2)
            if converged and rank < best:
                best = rank
                selected[name] = l2
                forecast_folds[name] = predictions
        if name not in selected:
            raise RuntimeError(f"{name}: no L2 candidate converged in every fold")

    validation_rows = np.concatenate([valid for _, valid in folds])
    oof = {name: pd.concat(frames) for name, frames in forecast_folds.items()}
    pooled = evaluate_predictions(target.iloc[validation_rows], oof)
    evaluations = [
        (scope, evaluate_predictions(target.iloc[valid], {name: frames[i] for name, frames in forecast_folds.items()}))
        for i, (scope, (_, valid)) in enumerate(zip(fold_labels, folds, strict=True))
    ]
    evaluations.append(("pooled", pooled))
    metrics = pd.concat([
        evaluation.metrics.assign(scope=scope, evaluation=EVALUATION_ROLE) for scope, evaluation in evaluations
    ], ignore_index=True)
    search_results = pd.DataFrame(search)
    search_results["selected"] = search_results["l2"].eq(search_results["model"].map(selected))
    search_results["iterations"] = search_results["iterations"].astype("Int64")
    match_predictions = metadata.iloc[validation_rows][["season"]].assign(
        fold=np.repeat(fold_info["fold"].to_numpy(), fold_info["validation_rows"].to_numpy()),
        outcome_code=target.iloc[validation_rows],
    )
    match_predictions = match_predictions.join(
        [frame.add_prefix(f"{name}_p_") for name, frame in oof.items()]
        + [pooled.paired_log_loss.drop(columns="M0").add_suffix("_log_loss_difference_vs_M0")],
    )

    effective_input = metadata[["season", "kickoff", "prediction_at", "outcome_available_at"]].join([odds, target])
    fingerprint = hashlib.sha256(effective_input.to_csv(lineterminator="\n", float_format="%.17g").encode()).hexdigest()
    manifest = {
        "market": market,
        "evaluation": EVALUATION_ROLE,
        "independent_evaluation": False,
        "selection_rule": "Minimum pooled out-of-fold log loss across candidates converged in every fold; ties favor stronger L2",
        "aggregation": "Each validation match has equal weight in pooled metrics",
        "cohort": "Phase 01 eligible rows, including article body and valid pick eligibility checks",
        "cohort_limitation": (
            "Conditional on estimated pre-snapshot article availability; not representative of all matches with early odds"
            if market == "early" else "Article publication must precede the kickoff-based closing cutoff proxy"
        ),
        "timing": "Odds snapshots and training outcome availability use Phase 01 timestamp proxies",
        "inputs": {"matches": _path_label(matches_path), "previews": _path_label(previews_path)},
        "effective_development_input_sha256": fingerprint,
        "development_rows": len(target),
        "development_seasons": metadata["season"].unique().tolist(),
        "match_id": "article_id from the Phase 01 one-to-one fixture cohort",
        "odds_columns": ODDS_COLUMNS[market],
        "class_order": CLASS_CODES,
        "l2_grid": L2_GRID,
        "selected_l2": selected,
        "objective": "Mean multinomial log loss plus L2 times half the squared norm of the slopes",
        "preprocessing": "Standardize log market probabilities within each training fold for M1 and M2 correction inputs",
        "optimizer": {"max_iterations": MAX_ITER, "tolerance": TOL, "m2_relative_loss_tolerance": 1e-12},
        "metric_conventions": {
            "log_loss": "Natural logarithm with a realized probability floor of 1e-15",
            "brier": "Mean over matches of the sum of squared class errors",
            "accuracy": "Argmax, with ties resolved by class order",
            "paired_log_loss": "Model minus M0 by match",
            "calibration": f"One-vs-rest, {CALIBRATION_BINS} equal-width bins, occupied bins only",
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    output = results_root / market
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    search_results.to_csv(output / "regularization_search.csv", index=False)
    fold_info.assign(train_seasons=fold_info["train_seasons"].str.join(",")).to_csv(output / "folds.csv", index=False)
    pd.concat([result.classwise.assign(scope=scope) for scope, result in evaluations], ignore_index=True).to_csv(
        output / "class_metrics.csv", index=False,
    )
    pd.concat([result.calibration.assign(scope=scope) for scope, result in evaluations], ignore_index=True).to_csv(
        output / "calibration.csv", index=False,
    )
    match_predictions.to_csv(output / "oof_predictions.csv")
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=tuple(ODDS_COLUMNS))
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--matches-path", type=Path, default=MATCHES_PATH)
    parser.add_argument("--previews-path", type=Path, default=PREVIEWS_PATH)
    args = parser.parse_args()
    metrics = run_experiment(
        args.market, args.results_root, matches_path=args.matches_path, previews_path=args.previews_path,
    )
    print(f"{args.market.capitalize()} development model selection")
    print(metrics.loc[metrics["scope"].eq("pooled"), ["model", "matches", "log_loss", "brier", "accuracy"]].to_string(index=False))
    print(f"Results saved in {args.results_root / args.market}")


if __name__ == "__main__":
    main()
