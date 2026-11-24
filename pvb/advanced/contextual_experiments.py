"""Select separate numerical and text penalties on football development folds."""

import argparse
from dataclasses import dataclass
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
from .embeddings import (
    CACHE_ROOT, MODEL_NAME, MODEL_RELEASE, MODEL_RELEASE_SOURCE, MODEL_REVISION, REPRESENTATION,
    EmbeddingUnavailable, MiniLMEncoder, embed_articles,
)
from .evaluation import evaluate_predictions
from .experiments import EVALUATION_ROLE, MATCHES_PATH, PREVIEWS_PATH, RESULTS_ROOT, _path_label
from .market import CLASS_NAMES, market_probabilities
from .models import L2_GRID, MAX_ITER, TOL, MarketResidual, RawMarket, residual_probabilities
from .nlp_experiments import (
    BLAS_THREADS, NLP_RESULTS_ROOT, PICK_COLUMNS, TEXT_WEIGHT, _development_fingerprint, _text_fold,
)
from .preprocessing import PRIMARY_TEXT_COLUMNS, TFIDF_CONFIG, prepare_documents
from .validation import chronological_folds, season_positions

CONTEXTUAL_RESULTS_ROOT = RESULTS_ROOT / "contextual"
NUMERICAL_L2 = (*L2_GRID, 1.0, 10.0)
TEXT_L2 = (0.0001, 0.001, 0.01, 0.1, 1.0)
NUMERICAL_WIDTHS = {"M2": len(CLASS_NAMES), "M3": len(CLASS_NAMES) + len(PICK_COLUMNS)}
TEXT_MODELS = {
    "M4": ("M2", "tfidf"), "M5": ("M3", "tfidf"),
    "M6": ("M2", "embedding"), "M7": ("M3", "embedding"),
}
MODEL_DESCRIPTIONS = {
    "M0": "Raw de-vigged market probabilities",
    "M2": "Market residual with log market probabilities", "M3": "M2 plus expert picks",
    "M4": "M2 plus primary TF-IDF", "M5": "M3 plus primary TF-IDF",
    "M6": "M2 plus frozen MiniLM article embeddings", "M7": "M3 plus frozen MiniLM article embeddings",
}


@dataclass
class Fold:
    scope: str
    train: np.ndarray
    valid: np.ndarray
    numerical: tuple[np.ndarray, np.ndarray]
    tfidf: tuple[sparse.csr_matrix, sparse.csr_matrix]
    embedding: tuple[np.ndarray, np.ndarray] | None


@dataclass
class Fit:
    model: MarketResidual
    forecast: pd.DataFrame | None
    total_loss: float
    report: dict


def _feature_norms(matrix, block, scope, partition):
    squared = matrix.multiply(matrix).sum(axis=1) if sparse.issparse(matrix) else np.square(matrix).sum(axis=1)
    norms = np.sqrt(np.asarray(squared).ravel())
    return {
        "scope": scope, "partition": partition, "block": block, "documents": len(norms),
        "features": matrix.shape[1], "row_l2_mean": norms.mean(), "row_l2_rms": np.sqrt(np.mean(norms**2)),
        "zero_rows": np.count_nonzero(norms == 0),
    }


def _prepare_folds(split_positions, numerical, documents, embeddings):
    folds, feature_rows, tfidf_rows = [], [], []
    for i, (train, valid) in enumerate(split_positions, start=1):
        scope = f"fold_{i}"
        scaler = StandardScaler()
        numeric = (scaler.fit_transform(numerical[train]), scaler.transform(numerical[valid]))
        tfidf, diagnostics = _text_fold(documents, train, valid, scope, "primary")
        tfidf_rows.extend(diagnostics)
        contextual = None
        if embeddings is not None:
            contextual = tuple(embeddings.loc[documents.index[rows]].to_numpy(dtype=float) for rows in (train, valid))
        folds.append(Fold(scope, train, valid, numeric, tfidf, contextual))
        for j, partition in enumerate(("train", "validation")):
            by_block = {
                "market": numeric[j][:, :NUMERICAL_WIDTHS["M2"]], "market_and_picks": numeric[j],
                "tfidf": tfidf[j], "phase04_weighted_tfidf": TEXT_WEIGHT * tfidf[j],
            }
            if contextual is not None:
                by_block["embedding"] = contextual[j]
            feature_rows.extend(_feature_norms(matrix, block, scope, partition) for block, matrix in by_block.items())
    return folds, feature_rows, tfidf_rows


def _model_matrices(folds, numerical_width, text_block=None, weight=1.0):
    matrices = []
    for fold in folds:
        numerical = tuple(values[:, :numerical_width] for values in fold.numerical)
        if text_block is None:
            matrices.append(numerical)
        elif text_block == "tfidf":
            matrices.append(tuple(
                sparse.hstack((sparse.csr_matrix(num), text if weight == 1 else weight * text), format="csr")
                for num, text in zip(numerical, fold.tfidf, strict=True)
            ))
        else:
            matrices.append(tuple(
                np.column_stack((num, text)) for num, text in zip(numerical, fold.embedding, strict=True)
            ))
    return matrices


def _fit_candidate(name, fold, market, target, matrices, width, numerical_l2, text_l2):
    train, valid = fold.train, fold.valid
    model = MarketResidual(numerical_l2, text_l2=text_l2, standardize=False)
    fit_args = {"numerical_width": width} if text_l2 is not None else {}
    start = perf_counter()
    model.fit(market[train], target.iloc[train].to_numpy(), matrices[0], **fit_args)
    fit_seconds = perf_counter() - start
    status = model.convergence_
    forecast, loss, prediction_seconds = None, np.nan, 0.0
    if status.converged:
        start = perf_counter()
        values = model.predict_proba(market[valid], matrices[1])
        prediction_seconds = perf_counter() - start
        forecast = pd.DataFrame(values, index=target.index[valid], columns=CLASS_NAMES)
        loss = log_loss(values, target.iloc[valid].to_numpy()).sum()
    return Fit(model, forecast, loss, {
        "scope": fold.scope, "model": name, "matches": len(valid), "numerical_l2": numerical_l2,
        "text_l2": text_l2, "log_loss": loss / len(valid), "converged": status.converged,
        "iterations": status.iterations, "optimizer_message": status.message,
        "numerical_features": width, "text_features": matrices[0].shape[1] - width,
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
    })


def _select_model(name, folds, market, target, matrices, width, penalties):
    best_rank, best_fits = (np.inf, 0.0), None
    search = []
    for numerical_l2, text_l2 in penalties:
        fits = [
            _fit_candidate(name, fold, market, target, pair, width, numerical_l2, text_l2)
            for fold, pair in zip(folds, matrices, strict=True)
        ]
        rows = [fit.report for fit in fits]
        n = sum(row["matches"] for row in rows)
        pooled = {
            "scope": "pooled", "model": name, "matches": n, "numerical_l2": numerical_l2, "text_l2": text_l2,
            "log_loss": sum(fit.total_loss for fit in fits) / n,
            "converged": all(row["converged"] for row in rows),
            "iterations": sum(row["iterations"] for row in rows),
            "fit_seconds": sum(row["fit_seconds"] for row in rows),
            "prediction_seconds": sum(row["prediction_seconds"] for row in rows),
        }
        search.extend([*rows, pooled])
        rank = (pooled["log_loss"], -(text_l2 if text_l2 is not None else numerical_l2))
        if pooled["converged"] and rank < best_rank:
            best_rank, best_fits = rank, fits
    if best_fits is None:
        raise RuntimeError(f"No regularization candidate for {name} converged in every development fold")
    selected = {key: best_fits[0].report[key] for key in ("numerical_l2", "text_l2")}
    for row in search:
        row["selected"] = all(row[key] == value for key, value in selected.items())
    return best_fits, selected, search


def _summarize_diagnostics(scope, name, numerical_logits, text_logits, without_text, with_text, norms, weight):
    changes = with_text - without_text
    return {
        "scope": scope, "model": name, "matches": len(changes), "text_block_weight": weight,
        "numerical_coef_frobenius": norms[0], "text_coef_frobenius": norms[1],
        "unweighted_text_coef_frobenius": weight * norms[1],
        "numerical_logit_sd": np.sqrt(np.var(numerical_logits, axis=0).mean()),
        "numerical_logit_rms": np.sqrt(np.mean(numerical_logits**2)),
        "text_logit_sd": np.sqrt(np.var(text_logits, axis=0).mean()),
        "text_logit_rms": np.sqrt(np.mean(text_logits**2)),
        "mean_absolute_probability_change": np.abs(changes).mean(),
        "rms_probability_change": np.sqrt(np.mean(changes**2)),
        "max_absolute_probability_change": np.abs(changes).max(),
        "argmax_changes": np.count_nonzero(without_text.argmax(axis=1) != with_text.argmax(axis=1)),
    }


def _block_diagnostics(name, fits, folds, matrices, market, width, weight):
    rows, components, norms = [], [], []
    for fit, fold, (_, valid) in zip(fits, folds, matrices, strict=True):
        num_coef, text_coef = fit.model.coef_[:width], fit.model.coef_[width:]
        norms.append([np.linalg.norm(num_coef), np.linalg.norm(text_coef)])
        numerical_logits = valid[:, :width] @ num_coef
        text_logits = valid[:, width:] @ text_coef
        without_text = residual_probabilities(market[fold.valid], numerical_logits + fit.model.intercept_)
        parts = (numerical_logits, text_logits, without_text, fit.forecast.to_numpy())
        components.append(parts)
        rows.append(_summarize_diagnostics(fold.scope, name, *parts, norms[-1], weight))
    pooled = [np.concatenate(values) for values in zip(*components, strict=True)]
    pooled_norms = np.sqrt(np.average(np.square(norms), axis=0, weights=[len(fold.valid) for fold in folds]))
    return [*rows, _summarize_diagnostics("pooled", name, *pooled, pooled_norms, weight)]


def _evaluate(folds, forecasts, target):
    scopes = [fold.scope for fold in folds] + ["pooled"]
    parts = [target.iloc[fold.valid] for fold in folds]
    evaluations = [
        evaluate_predictions(y, {name: frames[i] for name, frames in forecasts.items()})
        for i, y in enumerate(parts)
    ]
    evaluations.append(evaluate_predictions(
        pd.concat(parts), {name: pd.concat(frames) for name, frames in forecasts.items()},
    ))
    metrics = []
    paired = []
    comparisons = [("M2", "M0"), ("M3", "M0"), ("M3", "M2")]
    comparisons.extend((name, reference) for name, (reference, _) in TEXT_MODELS.items() if name in forecasts)
    for scope, result in zip(scopes, evaluations, strict=True):
        metrics.append(result.metrics.assign(scope=scope, evaluation=EVALUATION_ROLE))
        for name, reference in comparisons:
            differences = result.paired_log_loss[name] - result.paired_log_loss[reference]
            paired.append({
                "scope": scope, "model": name, "reference": reference, "matches": len(differences),
                "mean_log_loss_difference": differences.mean(),
                "text_conditional_on_picks": name in {"M5", "M7"},
            })
    return pd.concat(metrics, ignore_index=True), pd.DataFrame(paired)


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def run_experiment(
    market: Market,
    results_root: Path = CONTEXTUAL_RESULTS_ROOT,
    *,
    matches_path: Path = MATCHES_PATH,
    previews_path: Path = PREVIEWS_PATH,
    cache_root: Path = CACHE_ROOT,
    encoder=None,
    tfidf_only: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Tune numerical-only models first, then freeze their penalties for text comparisons."""
    start = perf_counter()
    output = results_root / "regularization_only" / market if tfidf_only else results_root / market
    if output.resolve() in {(root / market).resolve() for root in (RESULTS_ROOT, NLP_RESULTS_ROOT)}:
        raise ValueError("Choose a results directory separate from the Phase 03 and 04 results")
    data = load_football(market, matches_path=matches_path, previews_path=previews_path)
    if any(not frame.index.equals(data.metadata.index) for frame in (data.target, data.numerical, data.text)):
        raise ValueError("Dataset text, numerical features and target must align with the metadata rows")
    development = season_positions(data.metadata)["development"]
    metadata = data.metadata.iloc[development].set_index("article_id")
    target = data.target.iloc[development].set_axis(metadata.index)
    columns = [*ODDS_COLUMNS[market], *PICK_COLUMNS]
    numerical = data.numerical.iloc[development].loc[:, columns].set_axis(metadata.index)
    sections = data.text.iloc[development].set_axis(metadata.index)
    del data
    split_positions, fold_info = chronological_folds(metadata.reset_index())
    start_features = perf_counter()
    probabilities = market_probabilities(numerical, market)
    numeric_values = np.column_stack((np.log(probabilities), numerical.loc[:, list(PICK_COLUMNS)]))
    documents = prepare_documents(sections)
    embedding_info = {"status": "skipped_by_tfidf_only"}
    embeddings = None
    if not tfidf_only:
        if encoder is None:
            encoder = MiniLMEncoder(cache_root / "model", local_files_only=offline)
        used = np.unique(np.concatenate([positions for fold in split_positions for positions in fold]))
        embedding_input = sections.iloc[used].loc[:, list(PRIMARY_TEXT_COLUMNS)]
        embedded = embed_articles(embedding_input, encoder, cache_root / "articles.sqlite3")
        embeddings = embedded.vectors
        if not embeddings.index.equals(embedding_input.index):
            raise ValueError("Embedding results must align with the requested article IDs")
        embedding_info = {
            "status": "complete", "encoder": encoder.config, "runtime": encoder.runtime, **embedded.statistics,
        }
    folds, feature_rows, tfidf_rows = _prepare_folds(split_positions, numeric_values, documents, embeddings)
    feature_seconds = perf_counter() - start_features

    raw = RawMarket().predict_proba(probabilities)
    forecasts = {
        "M0": [pd.DataFrame(raw[fold.valid], index=target.index[fold.valid], columns=CLASS_NAMES) for fold in folds],
    }
    selected, search, diagnostics, boundaries = {}, [], [], []

    def run_model(name, baseline, text_block=None, *, legacy=False):
        width = NUMERICAL_WIDTHS[baseline]
        weight = TEXT_WEIGHT if legacy else 1.0
        matrices = _model_matrices(folds, width, text_block, weight)
        if legacy:
            grid, penalties = None, [(L2_GRID[-1], None)]
        elif text_block is None:
            grid, penalties = NUMERICAL_L2, [(value, None) for value in NUMERICAL_L2]
        else:
            grid = TEXT_L2
            penalties = [(selected[baseline]["numerical_l2"], value) for value in grid]
        fits, chosen, rows = _select_model(name, folds, probabilities, target, matrices, width, penalties)
        search.extend(rows)
        diagnostics.extend(_block_diagnostics(name, fits, folds, matrices, probabilities, width, weight))
        if legacy:
            return
        selected[name] = chosen
        forecasts[name] = [fit.forecast for fit in fits]
        parameter = "numerical_l2" if text_block is None else "text_l2"
        pooled = {row[parameter]: row["log_loss"] for row in rows if row["scope"] == "pooled"}
        value = chosen[parameter]
        boundaries.append({
            "model": name, "parameter": parameter, "selected": value, "grid_min": grid[0], "grid_max": grid[-1],
            "boundary": "lower" if value == grid[0] else "upper" if value == grid[-1] else "interior",
            "loss_at_min": pooled[grid[0]], "loss_at_max": pooled[grid[-1]], "selected_loss": pooled[value],
        })

    for name in NUMERICAL_WIDTHS:
        run_model(name, name)
    for name, (baseline, block) in TEXT_MODELS.items():
        if block == "embedding" and tfidf_only:
            continue
        run_model(name, baseline, block)
        if block == "tfidf":
            run_model(f"phase04_{name}", baseline, block, legacy=True)

    metrics, paired = _evaluate(folds, forecasts, target)
    for parameter in ("numerical_l2", "text_l2"):
        metrics[parameter] = metrics["model"].map({name: value[parameter] for name, value in selected.items()})
    data_fingerprint = _development_fingerprint(
        metadata[["season", "kickoff", "published", "prediction_at", "outcome_available_at"]].join(
            [numerical, target.rename("outcome_code")],
        ),
        {"primary": documents},
    )
    manifest = {
        "market": market, "evaluation": EVALUATION_ROLE, "independent_evaluation": False,
        "development_rows": len(target), "development_seasons": metadata["season"].unique().tolist(),
        "holdout_seasons_excluded": ["2526", "2627"],
        "inputs": {"matches": _path_label(matches_path), "previews": _path_label(previews_path)},
        "effective_development_input_sha256": data_fingerprint,
        "class_order": CLASS_NAMES, "odds_columns": ODDS_COLUMNS[market], "expert_pick_columns": PICK_COLUMNS,
        "models": {name: MODEL_DESCRIPTIONS[name] for name in forecasts},
        "objective": "Mean multinomial log loss + numerical_l2/2 * ||W_num||^2 + text_l2/2 * ||W_text||^2",
        "residual": "Fixed log-market offset, orthonormal sum-to-zero class contrasts, unpenalized intercept",
        "scaling": "Numerical standardization and TF-IDF vocabulary/IDF fit in the training fold. "
                   "TF-IDF and MiniLM article features retain their native L2 normalization.",
        "selection": "Pooled per-match development log loss among candidates converged in every fold. "
                     "Ties favor stronger L2. Text models inherit numerical L2 from the corresponding no-text model.",
        "numerical_grid": NUMERICAL_L2, "text_grid": TEXT_L2, "selected_penalties": selected,
        "tfidf": {"sections": PRIMARY_TEXT_COLUMNS, "settings": TFIDF_CONFIG},
        "phase04_reference": {
            "models": ["phase04_M4", "phase04_M5"], "shared_l2": L2_GRID[-1], "original_grid": L2_GRID,
            "text_weight": TEXT_WEIGHT, "effective_text_l2_in_unweighted_coordinates": L2_GRID[-1] / TEXT_WEIGHT**2,
            "diagnostic_reconstruction_only": True,
        },
        "embedding": {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "first_public_upload": MODEL_RELEASE,
            "release_source": MODEL_RELEASE_SOURCE, "representation": REPRESENTATION, **embedding_info,
        },
        "diagnostics": {
            "logit_sd": "Root mean of the three per-class population variances across validation matches",
            "logit_rms": "Root mean square across validation matches and classes",
            "pooled_coefficients": "Validation-match-weighted RMS of fold Frobenius norms",
            "probability_change": "Within fitted model, remove only the text correction and retain numerical slopes and intercept",
        },
        "limitations": [
            "Results use the same development folds for tuning and scoring.",
            "Article-version leakage remains unresolved pending the Phase 02 timing and provenance audit.",
            "Pretraining corpora may overlap evaluation-era material. The pinned revision is not established as historically available.",
            "Global article pooling may lose which team a statement refers to.",
        ],
        "optimizer": {"max_iterations": MAX_ITER, "gradient_tolerance": TOL, "relative_loss_tolerance": 1e-12},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "runtime": {
            "blas_threads": BLAS_THREADS, "feature_preparation_seconds": feature_seconds,
            "seconds_before_artifact_writes": perf_counter() - start,
        },
    }
    tables = {
        "metrics": metrics, "folds": fold_info.assign(train_seasons=fold_info["train_seasons"].str.join(",")),
        "paired_log_loss": paired, "regularization_search": pd.DataFrame(search),
        "search_boundaries": pd.DataFrame(boundaries), "block_diagnostics": pd.DataFrame(diagnostics),
        "feature_norms": pd.DataFrame(feature_rows), "tfidf_diagnostics": pd.DataFrame(tfidf_rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=(*ODDS_COLUMNS, "both"))
    parser.add_argument("--results-root", type=Path, default=CONTEXTUAL_RESULTS_ROOT)
    parser.add_argument("--matches-path", type=Path, default=MATCHES_PATH)
    parser.add_argument("--previews-path", type=Path, default=PREVIEWS_PATH)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--tfidf-only", action="store_true", help="Run the regularization experiments without MiniLM")
    parser.add_argument("--offline", action="store_true", help="Load the pinned MiniLM weights only from the local cache")
    args = parser.parse_args()
    markets = ("closing", "early") if args.market == "both" else (args.market,)
    encoder = None
    if not args.tfidf_only:
        try:
            print(f"Loading pinned MiniLM encoder {MODEL_REVISION[:8]}", flush=True)
            encoder = MiniLMEncoder(args.cache_root / "model", local_files_only=args.offline)
        except EmbeddingUnavailable as error:
            parser.exit(1, f"{error}\nUse --tfidf-only to run the regularization experiments independently.\n")
    for market in markets:
        print(f"{market.capitalize()} development model selection in progress", flush=True)
        metrics = run_experiment(
            market, args.results_root, matches_path=args.matches_path, previews_path=args.previews_path,
            cache_root=args.cache_root, encoder=encoder, tfidf_only=args.tfidf_only,
        )
        print(metrics.loc[metrics["scope"].eq("pooled"), [
            "model", "matches", "numerical_l2", "text_l2", "log_loss", "brier", "accuracy",
        ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
