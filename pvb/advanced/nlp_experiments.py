"""Run preliminary classical NLP experiments on football development folds."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..analysis import log_loss
from .dataset import ODDS_COLUMNS, Market, load_football
from .evaluation import evaluate_predictions
from .experiments import EVALUATION_ROLE, MATCHES_PATH, MODEL_TYPES, PREVIEWS_PATH, RESULTS_ROOT, _path_label
from .market import CLASS_CODES, CLASS_NAMES, market_probabilities
from .models import L2_GRID, MAX_ITER, TOL, MarketResidual, RawMarket
from .preprocessing import (
    PREDICTION_PATTERN, SCORELINE_PATTERN, TEXT_REPRESENTATIONS, TFIDF_CONFIG, UNICODE_FORM,
    fit_tfidf, prepare_documents, text_diagnostics,
)
from .validation import chronological_folds, season_positions

NLP_RESULTS_ROOT = RESULTS_ROOT / "nlp"
PICK_COLUMNS = ("pick_home", "pick_draw", "pick_goal_diff")
TEXT_WEIGHT = 0.5
BLAS_THREADS = 1
NLP_MODELS = ("M3", "M4", "M5")
ABLATION = "M5_full"
COMPARISONS = (("M3", "M2"), ("M4", "M2"), ("M5", "M3"))


def _combine_blocks(numerical: np.ndarray, text: sparse.csr_matrix) -> sparse.csr_matrix:
    return sparse.hstack((sparse.csr_matrix(numerical), TEXT_WEIGHT * text), format="csr")


def _text_fold(documents, train, valid, scope, representation):
    training_documents = documents.iloc[train]
    validation_documents = documents.iloc[valid]
    start = perf_counter()
    vectorizer, train_text = fit_tfidf(training_documents)
    training_seconds = perf_counter() - start
    start = perf_counter()
    valid_text = sparse.csr_matrix(vectorizer.transform(validation_documents))
    inference_seconds = perf_counter() - start
    diagnostics = []
    for partition, docs, matrix, seconds in (
        ("train", training_documents, train_text, training_seconds),
        ("validation", validation_documents, valid_text, inference_seconds),
    ):
        diagnostics.append({
            "scope": scope, "representation": representation, "partition": partition,
            "documents": len(docs), "features": matrix.shape[1], "nonzero_entries": matrix.nnz,
            "sparsity": 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
            "empty_documents": int(docs.eq("").sum()),
            "zero_vector_documents": int(np.count_nonzero(np.diff(matrix.indptr) == 0)),
            "seconds": seconds,
        })
    return (train_text, valid_text), diagnostics


def _fit_candidate(name, l2, scope, probabilities, target, train, valid, blocks=None):
    train_market, valid_market = probabilities[train], probabilities[valid]
    train_y, valid_y = target.iloc[train].to_numpy(), target.iloc[valid].to_numpy()
    train_args, valid_args = ((), ()) if blocks is None else ((blocks[0],), (blocks[1],))
    if name == "M0":
        model = RawMarket()
    elif name in MODEL_TYPES:
        model = MODEL_TYPES[name](l2)
    else:
        model = MarketResidual(l2, standardize=False)
    start = perf_counter()
    if name != "M0":
        model.fit(train_market, train_y, *train_args)
    fit_seconds = perf_counter() - start if name != "M0" else 0.0
    status = getattr(model, "convergence_", None)
    converged = status is None or status.converged
    forecast, total_loss, predict_seconds = None, np.nan, np.nan
    if converged:
        start = perf_counter()
        values = model.predict_proba(valid_market, *valid_args)
        predict_seconds = perf_counter() - start
        forecast = pd.DataFrame(values, index=target.index[valid], columns=CLASS_NAMES)
        total_loss = log_loss(values, valid_y).sum()
    return forecast, total_loss, {
        "model": name, "l2": l2, "scope": scope, "matches": len(valid),
        "log_loss": total_loss / len(valid), "converged": converged,
        "iterations": status.iterations if status else 0,
        "optimizer_message": status.message if status else "",
        "features": blocks[0].shape[1] if blocks is not None else (0 if name == "M0" else len(CLASS_CODES)),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
    }


def _candidate_summary(fits):
    forecasts, losses, rows = zip(*fits, strict=True)
    first = rows[0]
    matches = sum(row["matches"] for row in rows)
    pooled = {
        "model": first["model"], "l2": first["l2"], "scope": "pooled", "matches": matches,
        "log_loss": sum(losses) / matches, "converged": all(row["converged"] for row in rows),
        "fit_seconds": sum(row["fit_seconds"] for row in rows),
        "predict_seconds": sum(row["predict_seconds"] for row in rows),
    }
    return list(forecasts), pooled, [*rows, pooled]


def _select_candidates(candidates):
    best = {}
    forecasts = {}
    search = []
    for (name, l2), fits in candidates.items():
        frames, pooled, rows = _candidate_summary(fits)
        search.extend(rows)
        rank = (pooled["log_loss"], -(l2 or 0.0))
        if pooled["converged"] and rank < best.get(name, (np.inf, 0.0)):
            best[name] = rank
            forecasts[name] = frames
    missing = {name for name, _ in candidates} - forecasts.keys()
    if missing:
        raise RuntimeError(f"No L2 candidate converged in every fold for {', '.join(sorted(missing))}")
    return {name: -rank[1] for name, rank in best.items() if name != "M0"}, forecasts, search


def _development_fingerprint(effective_input, documents):
    digest = hashlib.sha256(effective_input.to_csv(lineterminator="\n", float_format="%.17g").encode("utf-8"))
    for representation, values in documents.items():
        digest.update(representation.encode("utf-8") + b"\0")
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def run_experiment(
    market: Market,
    results_root: Path = NLP_RESULTS_ROOT,
    *,
    matches_path: Path = MATCHES_PATH,
    previews_path: Path = PREVIEWS_PATH,
) -> pd.DataFrame:
    """Select L2 by pooled development OOF loss, then run one fixed M5 ablation."""
    output = results_root / market
    if output.resolve() == (RESULTS_ROOT / market).resolve():
        raise ValueError("Choose an NLP output directory separate from the Phase 03 results")
    start = perf_counter()
    data = load_football(market, matches_path=matches_path, previews_path=previews_path)
    if any(not frame.index.equals(data.metadata.index) for frame in (data.target, data.numerical, data.text)):
        raise ValueError("Dataset text, numerical features and target must align with the metadata rows")
    development = season_positions(data.metadata)["development"]
    metadata = data.metadata.iloc[development].set_index("article_id")
    target = data.target.iloc[development].set_axis(metadata.index).rename("outcome_code")
    columns = [*ODDS_COLUMNS[market], *PICK_COLUMNS]
    numerical = data.numerical.iloc[development].loc[:, columns].set_axis(metadata.index)
    sections = data.text.iloc[development].set_axis(metadata.index)
    del data
    folds, fold_info = chronological_folds(metadata.reset_index())
    probabilities = market_probabilities(numerical, market)
    log_market_and_picks = np.column_stack((np.log(probabilities), numerical.loc[:, list(PICK_COLUMNS)]))
    scopes = [f"fold_{number}" for number in fold_info["fold"]]
    start_documents = perf_counter()
    documents = {representation: prepare_documents(sections, representation) for representation in TEXT_REPRESENTATIONS}
    document_seconds = perf_counter() - start_documents
    candidates = {("M0", None): []}
    candidates.update({(name, l2): [] for name in (*MODEL_TYPES, *NLP_MODELS) for l2 in L2_GRID})
    feature_rows = []
    numerical_folds = []
    for i, (scope, (train, valid)) in enumerate(zip(scopes, folds, strict=True)):
        train_start = perf_counter()
        scaler = StandardScaler()
        train_numeric = scaler.fit_transform(log_market_and_picks[train])
        fold_info.loc[i, "numerical_fit_transform_seconds"] = perf_counter() - train_start
        valid_start = perf_counter()
        valid_numeric = scaler.transform(log_market_and_picks[valid])
        fold_info.loc[i, "numerical_validation_transform_seconds"] = perf_counter() - valid_start
        numerical_folds.append((train_numeric, valid_numeric))
        (train_text, valid_text), diagnostics = _text_fold(documents["primary"], train, valid, scope, "primary")
        feature_rows.extend(diagnostics)
        market_width = len(CLASS_CODES)
        matrices = {
            "M3": (train_numeric, valid_numeric),
            "M4": (_combine_blocks(train_numeric[:, :market_width], train_text),
                   _combine_blocks(valid_numeric[:, :market_width], valid_text)),
            "M5": (_combine_blocks(train_numeric, train_text), _combine_blocks(valid_numeric, valid_text)),
        }
        for (name, l2), fits in candidates.items():
            fits.append(_fit_candidate(name, l2, scope, probabilities, target, train, valid, matrices.get(name)))

    selected, forecast_folds, search = _select_candidates(candidates)
    ablation_fits = []
    for scope, (train, valid), (train_numeric, valid_numeric) in zip(scopes, folds, numerical_folds, strict=True):
        (train_text, valid_text), diagnostics = _text_fold(
            documents["full_article"], train, valid, scope, "full_article",
        )
        feature_rows.extend(diagnostics)
        blocks = (_combine_blocks(train_numeric, train_text), _combine_blocks(valid_numeric, valid_text))
        ablation_fits.append(_fit_candidate(ABLATION, selected["M5"], scope, probabilities, target, train, valid, blocks))
    ablation_forecasts, ablation_summary, ablation_search = _candidate_summary(ablation_fits)
    search.extend(ablation_search)
    if ablation_summary["converged"]:
        forecast_folds[ABLATION] = ablation_forecasts

    comparisons = [*((name, "M0") for name in forecast_folds if name != "M0"), *COMPARISONS]
    if ABLATION in forecast_folds:
        comparisons.extend(((ABLATION, "M5"), (ABLATION, "M3")))
    evaluations = [
        (scope, evaluate_predictions(target.iloc[valid], {name: frames[i] for name, frames in forecast_folds.items()}))
        for i, (scope, (_, valid)) in enumerate(zip(scopes, folds, strict=True))
    ]
    valid_rows = np.concatenate([valid for _, valid in folds])
    evaluations.append(("pooled", evaluate_predictions(
        target.iloc[valid_rows], {name: pd.concat(frames) for name, frames in forecast_folds.items()},
    )))
    metrics = pd.concat([result.metrics.assign(scope=scope) for scope, result in evaluations], ignore_index=True)
    reported_l2 = {**selected, ABLATION: selected["M5"]}
    metrics = metrics.assign(l2=metrics["model"].map(reported_l2), evaluation=EVALUATION_ROLE, preliminary=True)
    paired = []
    for scope, result in evaluations:
        for name, reference in comparisons:
            differences = result.paired_log_loss[name] - result.paired_log_loss[reference]
            paired.append({
                "scope": scope, "model": name, "reference": reference, "matches": len(differences),
                "mean_log_loss_difference": differences.mean(), "primary_comparison": (name, reference) == ("M5", "M3"),
            })
    search_results = pd.DataFrame(search)
    included = {**selected, **({ABLATION: selected["M5"]} if ablation_summary["converged"] else {})}
    search_results["included_in_reported_oof"] = (
        search_results["model"].eq("M0") | search_results["l2"].eq(search_results["model"].map(included))
    )
    search_results["iterations"] = search_results["iterations"].astype("Int64")
    search_results["features"] = search_results["features"].astype("Int64")

    effective_input = metadata[["season", "kickoff", "published", "prediction_at", "outcome_available_at"]].join(
        [numerical, target],
    )
    manifest = {
        "market": market, "evaluation": EVALUATION_ROLE, "preliminary": True, "independent_evaluation": False,
        "cohort": "Phase 01 eligibility and Phase 03 expanding folds for the named market",
        "holdouts": "2526 and 2627 are reserved historical holdouts with known original-project aggregate results",
        "limitations": [
            "Publication and odds timing proxies do not establish the scraped article version at prediction time",
            "Primary sections can contain explicit predictions and scorelines",
            "The same development folds select L2 and supply pooled OOF scores",
        ],
        "inputs": {"matches": _path_label(matches_path), "previews": _path_label(previews_path)},
        "effective_development_input_sha256": _development_fingerprint(effective_input, documents),
        "development_rows": len(target), "development_seasons": metadata["season"].unique().tolist(),
        "class_order": CLASS_CODES, "odds_columns": ODDS_COLUMNS[market], "expert_pick_columns": PICK_COLUMNS,
        "text_preprocessing": {
            "representations": TEXT_REPRESENTATIONS, "unicode_form": UNICODE_FORM,
            "whitespace": "Collapse whitespace and join sections with single spaces; missing sections are empty",
            "tfidf": TFIDF_CONFIG, "scoreline_pattern": SCORELINE_PATTERN.pattern,
            "prediction_phrase_pattern_case_insensitive": PREDICTION_PATTERN.pattern,
        },
        "models": {
            "M0": "Proportionally de-vigged market probabilities",
            "M1": "Multinomial logistic regression with standardized log market probabilities",
            "M2": "Market residual with standardized log market probabilities",
            "M3": "M2 plus standardized expert picks",
            "M4": "M2 plus weighted primary TF-IDF",
            "M5": "M3 plus weighted primary TF-IDF",
            ABLATION: "M5 with full-article TF-IDF at the selected M5 L2",
        },
        "objective": "Mean multinomial log loss + l2/2 * squared slope norm; intercepts unpenalized",
        "residual": "softmax(log(p_market) + intercept + standardized_numerical @ B + weighted_tfidf @ C)",
        "scaling": "Numerical columns centered and scaled by training-fold population SD (unit scale for constants); "
                   "no column scaling of TF-IDF; penalties use orthonormal class contrasts for residual models",
        "text_block_weight": TEXT_WEIGHT, "effective_text_l2_multiplier": 1 / TEXT_WEIGHT**2,
        "l2_grid": L2_GRID, "selected_l2": selected,
        "selection_rule": "Minimum pooled per-match OOF log loss among candidates converged in every fold; ties favor stronger L2",
        "ablation": {"model": ABLATION, "l2": selected["M5"], "converged": ablation_summary["converged"]},
        "metric_conventions": {
            "log_loss": "Natural logarithm with a realized probability floor of 1e-15",
            "brier": "Mean per-match sum of squared class errors",
            "accuracy": "Argmax with ties resolved in H/D/A order",
            "paired_log_loss": "Mean per-match model loss minus reference loss; negative favors the model",
        },
        "optimizer": {"max_iterations": MAX_ITER, "gradient_tolerance": TOL, "residual_relative_loss_tolerance": 1e-12},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    tables = {
        "metrics": metrics, "folds": fold_info.assign(train_seasons=fold_info["train_seasons"].str.join(",")),
        "paired_log_loss": pd.DataFrame(paired), "regularization_search": search_results,
        "feature_diagnostics": pd.DataFrame(feature_rows),
        "text_diagnostics": pd.DataFrame([
            {"representation": name, **text_diagnostics(values)} for name, values in documents.items()
        ]),
    }
    manifest["runtime"] = {
        "blas_threads": BLAS_THREADS, "document_preparation_seconds": document_seconds,
        "seconds_before_artifact_writes": perf_counter() - start,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=tuple(ODDS_COLUMNS))
    parser.add_argument("--results-root", type=Path, default=NLP_RESULTS_ROOT)
    parser.add_argument("--matches-path", type=Path, default=MATCHES_PATH)
    parser.add_argument("--previews-path", type=Path, default=PREVIEWS_PATH)
    args = parser.parse_args()
    metrics = run_experiment(
        args.market, args.results_root, matches_path=args.matches_path, previews_path=args.previews_path,
    )
    print(f"{args.market.capitalize()} preliminary development model selection")
    print(metrics.loc[metrics["scope"].eq("pooled"), ["model", "matches", "l2", "log_loss", "brier", "accuracy"]]
          .to_string(index=False))
    if ABLATION not in metrics["model"].values:
        print("Full-article ablation omitted because at least one fit failed to converge")
    print(f"Results saved in {args.results_root / args.market}")


if __name__ == "__main__":
    main()
