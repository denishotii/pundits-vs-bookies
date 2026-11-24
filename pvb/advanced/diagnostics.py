"""Run Phase 06A signal diagnostics on football development seasons, offline."""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..analysis import log_loss
from .contextual_experiments import CONTEXTUAL_RESULTS_ROOT
from .diagnostic_data import (
    DEVELOPMENT, REVIEW_SALT, TIMING_BUFFER_HOURS, attribution_audit, load_development, movement_inputs,
)
from .diagnostic_metrics import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, MOVEMENT_THRESHOLD, SYNTHETIC_PENALTIES, SYNTHETIC_SEEDS,
    categorical_metrics, date_block_bootstrap, gradient_checks, paired_summary, soft_cross_entropy, soft_metrics,
    synthetic_controls,
)
from .embeddings import CACHE_ROOT, EmbeddingUnavailable, MiniLMEncoder, embed_articles
from .experiments import MATCHES_PATH, PREVIEWS_PATH, RESULTS_ROOT, _path_label
from .market import CLASS_NAMES, N_CLASSES, market_probabilities
from .models import MarketResidual
from .nlp_experiments import BLAS_THREADS, PICK_COLUMNS, _development_fingerprint
from .preprocessing import TEXT_REPRESENTATIONS, TFIDF_CONFIG, fit_tfidf, prepare_documents
from .validation import chronological_folds

RESULTS = RESULTS_ROOT / "diagnostics"
COMMAND = r".\.venv\Scripts\python.exe -m pvb.advanced.diagnostics"
L2_GRID = (0.0001, 0.001, 0.01, 0.1)
MOVEMENT_L2_GRID = (0.001, 0.01, 0.1, 1.0)
TRAIN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
TEXT_MODELS = {
    "expert_picks": ("picks",), "tfidf": ("tfidf",), "minilm": ("minilm",),
    "minilm_picks": ("picks", "minilm"),
}
PICK_RECOVERY_MODELS = {
    "tfidf_primary": ("tfidf",), "minilm_primary": ("minilm",),
    "tfidf_full_verdict_control": ("tfidf_full",), "minilm_full_verdict_control": ("minilm_full",),
}
MOVEMENT_MODELS = {
    "market_calibration": ("market",), "market_picks": ("market", "picks"),
    "market_minilm": ("market", "minilm"), "market_picks_minilm": ("market", "picks", "minilm"),
}
RESIDUAL_MODELS = {
    "M2": ("market",), "M3": ("market", "picks"),
    "M4": ("market", "tfidf"), "M5": ("market", "picks", "tfidf"),
    "M6": ("market", "minilm"), "M7": ("market", "picks", "minilm"),
}
PAIRED_MODELS = {"M4": "M2", "M5": "M3", "M6": "M2", "M7": "M3"}


@dataclass
class Fold:
    season: str
    train: np.ndarray
    valid: np.ndarray
    blocks: dict

    def features(self, names):
        for matrices in zip(*(self.blocks[name] for name in names), strict=True):
            if len(matrices) == 1:
                yield matrices[0]
            elif any(sparse.issparse(matrix) for matrix in matrices):
                yield sparse.hstack(matrices, format="csr")
            else:
                yield np.column_stack(matrices)


def _make_fold(season, train, valid, numerical, documents, embeddings):
    blocks = {}
    for name, values in numerical.items():
        scaler = StandardScaler()
        blocks[name] = (scaler.fit_transform(values[train]), scaler.transform(values[valid]))
    for name, values in documents.items():
        vectorizer, training = fit_tfidf(values.iloc[train])
        blocks[name] = (training, vectorizer.transform(values.iloc[valid]))
    for name, vectors in embeddings.items():
        blocks[name] = (vectors[train], vectors[valid])
    return Fold(season, train, valid, blocks)


def _folds(metadata, numerical, documents=None, embeddings=None):
    positions, stats = chronological_folds(metadata.reset_index())
    folds = [
        _make_fold(season, train, valid, numerical, documents or {}, embeddings or {})
        for season, (train, valid) in zip(stats["validation_season"], positions, strict=True)
    ]
    return folds, stats.assign(train_seasons=stats["train_seasons"].str.join(","))


def _fit(fold, blocks, offset, target, l2, text_l2=None):
    training, validation = fold.features(blocks)
    kwargs = {}
    if text_l2 is not None:
        kwargs["numerical_width"] = sum(fold.blocks[block][0].shape[1] for block in blocks[:-1])
    model = MarketResidual(l2, text_l2=text_l2, standardize=False)
    fit = model.fit if target.ndim == 1 else model.fit_soft
    fit(offset[fold.train], target[fold.train], training, **kwargs)
    status = model.convergence_
    return (model.predict_proba(offset[fold.valid], validation) if status.converged else None), status


def _forecast_frame(index, values):
    return pd.DataFrame(values, index=index, columns=CLASS_NAMES)


def _search(folds, models, index, offset, target, grid):
    """Select by pooled development cross-entropy, weighting matches equally."""
    forecasts, rows = {}, []
    score = log_loss if target.ndim == 1 else lambda pred, y: soft_cross_entropy(y, pred)
    for model, blocks in models.items():
        if not all(block in folds[0].blocks for block in blocks):
            continue
        best, split_rows = (np.inf, 0), []
        for l2 in grid:
            parts, losses = [], []
            for fold in folds:
                prediction, status = _fit(fold, blocks, offset, target, l2)
                loss = score(prediction, target[fold.valid]).mean() if prediction is not None else np.nan
                losses.append(loss)
                parts.append(_forecast_frame(index[fold.valid], prediction))
                split_rows.append({
                    "model": model, "l2": l2, "scope": fold.season, "mean_loss": loss,
                    "converged": status.converged, "iterations": status.iterations, "matches": len(fold.valid),
                })
            pooled = np.average(losses, weights=[len(fold.valid) for fold in folds])
            split_rows.append({
                "model": model, "l2": l2, "scope": "pooled", "mean_loss": pooled,
                "converged": np.isfinite(pooled), "matches": sum(len(fold.valid) for fold in folds),
            })
            if (pooled, -l2) < best:
                best = (pooled, -l2)
                forecasts[model] = pd.concat(parts, verify_integrity=True)
        if not np.isfinite(best[0]):
            raise RuntimeError(f"No converged candidate for {model}")
        rows.extend({**row, "selected": row["l2"] == -best[1]} for row in split_rows)
    return forecasts, pd.DataFrame(rows)


def _reference(folds, index, prediction):
    return pd.concat([_forecast_frame(index[fold.valid], prediction(fold)) for fold in folds], verify_integrity=True)


def _class_frequency(folds, target):
    def predict(fold):
        frequency = np.bincount(target.iloc[fold.train], minlength=N_CLASSES) / len(fold.train)
        return np.tile(frequency, (len(fold.valid), 1))

    return _reference(folds, target.index, predict)


def _scopes(folds):
    return [(fold.season, fold.valid) for fold in folds] + [("pooled", np.concatenate([fold.valid for fold in folds]))]


def _categorical_tables(folds, target, forecasts):
    tables = {"metrics": [], "classwise": [], "confusion": []}
    for scope, positions in _scopes(folds):
        truth = target.iloc[positions]
        aligned = {model: frame.loc[truth.index] for model, frame in forecasts.items()}
        for key, values in zip(tables, categorical_metrics(truth, aligned, "class_frequency"), strict=True):
            tables[key].append(values.assign(scope=scope))
    return {key: pd.concat(frames, ignore_index=True) for key, frames in tables.items()}


def _soft_table(folds, index, target, forecasts, reference, early=None):
    rows = []
    for scope, positions in _scopes(folds):
        ids = index[positions]
        baseline = forecasts[reference].loc[ids].to_numpy()
        for name, forecast in forecasts.items():
            rows.append({
                "scope": scope, "model": name, **soft_metrics(
                    target[positions], forecast.loc[ids].to_numpy(), baseline,
                    early=None if early is None else early[positions],
                ),
            })
    return pd.DataFrame(rows)


def _fixed_residuals(folds, index, market, target, settings, model_names):
    models = {}
    for name in model_names:
        blocks = RESIDUAL_MODELS[name]
        if not all(block in folds[0].blocks for block in blocks):
            continue
        parts = []
        for fold in folds:
            prediction, status = _fit(
                fold, blocks, market, target, settings[name]["numerical_l2"], settings[name]["text_l2"],
            )
            if not status.converged:
                raise RuntimeError(f"Fixed Phase 05 fit {name} did not converge in {fold.season}: {status.message}")
            parts.append(_forecast_frame(index[fold.valid], prediction))
        models[name] = pd.concat(parts, verify_integrity=True)
    return models


def recent_training(metadata, train, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("A training fraction must be in (0, 1]")
    times = metadata.iloc[train][["outcome_available_at"]].copy()
    times["position"] = train
    order = times.sort_index().sort_values("outcome_available_at", kind="stable")
    return order["position"].to_numpy()[-max(1, round(len(order) * fraction)):]


def _learning_curve(fold, data, market, settings, numerical, documents, embeddings, predictions):
    rows = []
    model_names = ("M2", "M3", "M5", "M7")
    for fraction in TRAIN_FRACTIONS:
        if fraction == 1:
            subset = fold
            forecasts = {name: predictions[name] for name in model_names if name in predictions}
        else:
            recent = recent_training(data.metadata, fold.train, fraction)
            subset = _make_fold(fold.season, recent, fold.valid, numerical, documents, embeddings)
            forecasts = _fixed_residuals(
                [subset], data.metadata.index, market, data.target.to_numpy(), settings, model_names,
            )
        target = data.target.iloc[subset.valid]
        losses = {name: log_loss(pred.loc[target.index].to_numpy(), target.to_numpy()).mean()
                  for name, pred in forecasts.items()}
        train_metadata = data.metadata.iloc[subset.train]
        for name, loss in losses.items():
            baseline = {"M2": "M2", "M3": "M2", "M5": "M3", "M7": "M3"}[name]
            rows.append({
                "training_fraction": fraction, "model": name, "baseline": baseline,
                "train_rows": len(subset.train), "valid_rows": len(subset.valid), "validation_season": subset.season,
                "first_training_kickoff": train_metadata["kickoff"].min(),
                "last_training_kickoff": train_metadata["kickoff"].max(),
                "log_loss": loss, "delta_log_loss": loss - losses[baseline], **settings[name],
            })
    return pd.DataFrame(rows)


def _uncertainty(folds, data, forecasts):
    paired_rows, intervals, private_rows, replay = [], [], [], []
    for scope, positions in _scopes(folds):
        target = data.target.iloc[positions]
        losses = pd.DataFrame({
            model: log_loss(forecast.loc[target.index].to_numpy(), target.to_numpy())
            for model, forecast in forecasts.items()
        }, index=target.index)
        differences = pd.DataFrame({
            f"{name}-{baseline}": losses[name] - losses[baseline]
            for name, baseline in PAIRED_MODELS.items() if name in losses
        }, index=target.index)
        paired_rows.extend({"scope": scope, "comparison": name, **paired_summary(differences[name])}
                           for name in differences)
        if scope == "pooled":
            replay = losses.mean().to_dict()
            metadata = data.metadata.loc[target.index]
            intervals = [date_block_bootstrap(differences, metadata, days) for days in (7, 14)]
            for name, forecast in forecasts.items():
                private_rows.append(metadata[["season", "kickoff"]].join(target).join(forecast).assign(model=name))
    return pd.DataFrame(paired_rows), pd.concat(intervals, ignore_index=True), pd.concat(private_rows), replay


def _write(table, output, name, *, index=False):
    table.to_csv(output / f"{name}.csv", index=index, lineterminator="\n")


def _indexed_embeddings(results, index):
    return {name: result.vectors.loc[index].to_numpy(dtype=float) for name, result in results.items()}


def _run_market(data, market, embedded, phase05_root, output, manifest):
    index = data.metadata.index
    book = market_probabilities(data.numerical, market)
    picks = data.numerical.loc[:, list(PICK_COLUMNS)].to_numpy(dtype=float)
    documents = {"tfidf": prepare_documents(data.text), "tfidf_full": prepare_documents(data.text, "full_article")}
    embeddings = _indexed_embeddings(embedded, index)
    numerical = {"market": np.log(book), "picks": picks}
    folds, stats = _folds(data.metadata, numerical, documents, embeddings)
    _write(stats, output, f"{market}_folds")
    uniform = np.full_like(book, 1 / N_CLASSES)
    y = data.target

    text_outcomes, outcome_search = _search(folds, TEXT_MODELS, index, uniform, y.to_numpy(), L2_GRID)
    text_outcomes["raw_market"] = _reference(folds, index, lambda fold: book[fold.valid])
    text_outcomes["class_frequency"] = _class_frequency(folds, y)
    for key, table in _categorical_tables(folds, y, text_outcomes).items():
        if key != "confusion":
            _write(table, output, f"{market}_outcome_{key}")
    pick_target = pd.Series(np.where(picks[:, 0] == 1, 0, np.where(picks[:, 1] == 1, 1, 2)), index=index)
    pick_predictions, pick_search = _search(folds, PICK_RECOVERY_MODELS, index, uniform, pick_target.to_numpy(), L2_GRID)
    pick_predictions["class_frequency"] = _class_frequency(folds, pick_target)
    for key, table in _categorical_tables(folds, pick_target, pick_predictions).items():
        _write(table, output, f"{market}_pick_{key}")

    book_predictions, book_search = _search(folds, TEXT_MODELS, index, uniform, book, L2_GRID)
    book_predictions["mean_market"] = _reference(
        folds, index, lambda fold: np.tile(book[fold.train].mean(axis=0), (len(fold.valid), 1)),
    )
    _write(_soft_table(folds, index, book, book_predictions, "mean_market"), output, f"{market}_market_recovery")
    for name, table in (("outcome", outcome_search), ("pick_recovery", pick_search), ("market_recovery", book_search)):
        table["experiment"] = name
    _write(pd.concat([outcome_search, pick_search, book_search], ignore_index=True), output, f"{market}_selection")

    phase05 = phase05_root / market
    previous = json.loads((phase05 / "metadata.json").read_text(encoding="utf-8"))
    settings = previous["selected_penalties"]
    fixed = _fixed_residuals(folds, index, book, y.to_numpy(), settings, RESIDUAL_MODELS)
    curve = _learning_curve(
        folds[-1], data, book, settings, numerical, {"tfidf": documents["tfidf"]},
        {name: value for name, value in embeddings.items() if name == "minilm"}, fixed,
    )
    _write(curve, output, f"{market}_learning_curves")
    paired, intervals, private, replay = _uncertainty(folds, data, fixed)
    _write(paired, output, f"{market}_paired")
    _write(intervals, output, f"{market}_bootstrap")
    _write(private, output / "private", f"{market}_residual_oof", index=True)
    previous_metrics = pd.read_csv(phase05 / "metrics.csv")
    previous_pooled = previous_metrics.loc[previous_metrics["scope"].eq("pooled")].set_index("model")["log_loss"]
    fingerprint_input = data.metadata[["season", "kickoff", "published", "prediction_at", "outcome_available_at"]]
    fingerprint = _development_fingerprint(
        fingerprint_input.join([data.numerical, y.rename("outcome_code")]), {"primary": documents["tfidf"]},
    )
    manifest[market] = {
        "development_rows": len(index), "validation_rows": sum(len(fold.valid) for fold in folds),
        "phase05_penalties": settings, "phase05_source": _path_label(phase05),
        "primary_development_sha256": fingerprint,
        "phase05_primary_input_matches": fingerprint == previous["effective_development_input_sha256"],
        "phase05_pooled_loss_replay_delta": {name: loss - previous_pooled[name] for name, loss in replay.items()},
        "text_sha256": _development_fingerprint(fingerprint_input, documents),
    }


def _run_movement(group, metadata, early, target, numerical, embeddings, output):
    index = metadata.index
    folds, stats = _folds(metadata, numerical, embeddings=embeddings)
    forecasts, search = _search(folds, MOVEMENT_MODELS, index, early, target, MOVEMENT_L2_GRID)
    forecasts["no_movement"] = _reference(folds, index, lambda fold: early[fold.valid])
    _write(stats, output, f"movement_{group}_folds")
    _write(search, output, f"movement_{group}_selection")
    _write(_soft_table(folds, index, target, forecasts, "no_movement", early), output, f"movement_{group}")


def _embeddings(sections, cache_root, manifest):
    results = {}
    try:
        encoder = MiniLMEncoder(cache_root / "model", local_files_only=True)
    except EmbeddingUnavailable as error:
        manifest["embedding_blocker"] = str(error)
        print("Pinned local MiniLM unavailable. Continuing with numerical and TF-IDF diagnostics.", flush=True)
        return results
    manifest["encoder"] = encoder.config
    manifest["encoder_runtime"] = encoder.runtime
    for key, representation, cache in (
        ("minilm", "primary", "articles.sqlite3"), ("minilm_full", "full_article", "diagnostics_full.sqlite3"),
    ):
        embedded = embed_articles(sections, encoder, cache_root / cache, representation=representation)
        if not embedded.vectors.index.equals(sections.index):
            raise ValueError("Embedding rows must align with input article IDs")
        results[key] = embedded
        manifest[key] = embedded.statistics
        print(f"{representation}: {embedded.statistics['cache_hits']} cache hits, "
              f"{embedded.statistics['cache_misses']} generated", flush=True)
    return results


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def run(matches_path, previews_path, phase05_root, cache_root):
    started = perf_counter()
    output = RESULTS
    private = output / "private"
    private.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": COMMAND, "development_seasons": DEVELOPMENT, "excluded_holdouts": ("2526", "2627"),
        "class_order": CLASS_NAMES, "text_representations": TEXT_REPRESENTATIONS, "tfidf": TFIDF_CONFIG,
        "grids": {"standalone": L2_GRID, "movement": MOVEMENT_L2_GRID},
        "feature_allowlists": {
            "early_odds": ("early_odds_home", "early_odds_draw", "early_odds_away"),
            "expert_picks": PICK_COLUMNS, "text_models": TEXT_MODELS, "pick_recovery": PICK_RECOVERY_MODELS,
            "odds_movement": MOVEMENT_MODELS,
        },
        "inputs": {name: _path_label(path) for name, path in (
            ("matches", matches_path), ("previews", previews_path), ("phase05", phase05_root), ("cache", cache_root),
        )},
        "validation": "Expanding prior-season windows, purged by training target availability. "
                      "Pooled development-fold loss selects regularization and supplies descriptive scores.",
        "fit": "Mean hard-label or soft-target multinomial cross-entropy and half-L2 on slopes. "
               "A uniform offset gives multinomial logistic regression. Numerical scaling and TF-IDF are training-only. "
               "TF-IDF and MiniLM keep their native unit row scale. Movement offsets are log early probabilities.",
        "movement": {
            "timing_buffer_hours": TIMING_BUFFER_HOURS, "material_change_threshold": MOVEMENT_THRESHOLD,
            "directional_accuracy": "Exact predicted sign on each class with observed absolute change >= 0.01. "
                                    "A forecast of zero change is incorrect on a material movement.",
            "target_available": "Closing market targets conservatively available at the dataset's "
                                "kickoff-plus-buffer proxy, derived solely from kickoff.",
            "timestamp": "Estimated 15:00 Europe/London on Friday or Tuesday. "
                         "Midnight UTC publication is date-only. Post-snapshot evaluation from publication "
                         "is retrospective. Actual closing capture times and article versions are unknown.",
        },
        "synthetic": {"seeds": SYNTHETIC_SEEDS, "penalties": SYNTHETIC_PENALTIES, "gradient_check_seed": 60101},
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "block_days": [7, 14],
            "definition": "Resample occupied fixed calendar blocks anchored to Monday 1970-01-05 UTC within each "
                          "validation season, using the original season match counts as pooling weights.",
            "minimum_detectable_delta": "Bootstrap standard error times (1.96 + 0.84), conditional on the fitted "
                                        "forecasts, with nominal 5% two-sided size and 80% sensitivity under a normal shift.",
        },
        "attribution": {"review_salt": REVIEW_SALT, "learned_aliases": False, "target_review_counts": [13, 13, 12, 12]},
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
        "python": platform.python_version(), "blas_threads": BLAS_THREADS,
    }
    gradients = pd.DataFrame(gradient_checks())
    _write(gradients, output, "gradient_checks")
    if not gradients["passed"].all():
        raise RuntimeError("Finite-difference residual verification failed")
    synthetic = synthetic_controls()
    _write(synthetic, output, "synthetic_by_seed")
    summary = synthetic.groupby(["case", "block_l2"], as_index=False).agg(
        seeds=("seed", "size"), converged=("converged", "sum"), log_loss=("log_loss", "mean"),
        market_log_loss=("market_log_loss", "mean"), delta_mean=("delta_log_loss", "mean"),
        delta_sd=("delta_log_loss", "std"), expected_delta_mean=("expected_delta_log_loss", "mean"),
        oracle_delta_mean=("oracle_delta_log_loss", "mean"), noise_coef_norm=("noise_coef_norm", "mean"),
        signal_home_minus_away_coef=("signal_home_minus_away_coef", "mean"),
    )
    _write(summary, output, "synthetic_summary")
    markets = {name: load_development(name, matches_path, previews_path) for name in ("early", "closing")}
    closing = load_development("closing", matches_path, previews_path, include_outcomes=False)
    movement, coverage = movement_inputs(closing, matches_path)
    _write(coverage, output, "movement_coverage")
    attribution, review = attribution_audit(markets["closing"], matches_path, previews_path)
    _write(attribution, output, "attribution_coverage")
    _write(review, private, "attribution_review")
    sections = pd.concat([market.text for market in markets.values()])
    sections = sections.loc[~sections.index.duplicated(keep="first")].sort_index()
    extra_ids = closing.text.index.difference(sections.index)
    # Only the primary representation is needed for any outcome-free extra rows.
    embedded = _embeddings(sections, cache_root, manifest)
    motion_embedded = {key: value for key, value in embedded.items() if key == "minilm"}
    if len(extra_ids) and motion_embedded:
        encoder = MiniLMEncoder(cache_root / "model", local_files_only=True)
        primary = embedded["minilm"].vectors
        extra = embed_articles(closing.text.loc[extra_ids], encoder, cache_root / "articles.sqlite3")
        extra.vectors = pd.concat([primary, extra.vectors], verify_integrity=True)
        motion_embedded = {"minilm": extra}
        manifest["minilm_outcome_free_extra"] = extra.statistics
    for name, data in markets.items():
        print(f"Running {name} diagnostics", flush=True)
        _run_market(data, name, embedded, phase05_root, output, manifest)
    for group, (fixtures, early, closing_probs) in movement.items():
        print(f"Running movement diagnostics for {group}", flush=True)
        numerical = {"market": np.log(early), "picks": closing.numerical.loc[fixtures.index, list(PICK_COLUMNS)].to_numpy()}
        _run_movement(group, fixtures, early, closing_probs, numerical,
                      _indexed_embeddings(motion_embedded, fixtures.index), output)
    manifest["runtime_seconds"] = perf_counter() - started
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-path", type=Path, default=MATCHES_PATH)
    parser.add_argument("--previews-path", type=Path, default=PREVIEWS_PATH)
    parser.add_argument("--phase05-root", type=Path, default=CONTEXTUAL_RESULTS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    run(args.matches_path, args.previews_path, args.phase05_root, args.cache_root)


if __name__ == "__main__":
    main()
