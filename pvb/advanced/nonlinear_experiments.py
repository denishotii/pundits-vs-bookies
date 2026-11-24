"""Run Phase 07 author reliability, CatBoost and anchoring on development folds."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..analysis import log_loss
from .author_features import AUTHOR_FEATURES, MISSING_FLAGS, PRIOR_STRENGTH, author_features, load_author_keys
from .diagnostic_data import DEVELOPMENT, load_development
from .diagnostic_metrics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, date_block_bootstrap, paired_summary
from .diagnostics import _check_validation_cohort, _folds, _reference, _scopes, _write
from .embeddings import CACHE_ROOT, _fingerprint
from .evaluation import evaluate_predictions
from .experiments import MATCHES_PATH, PREVIEWS_PATH, RESULTS_ROOT
from .historical_features import (
    HISTORY_FEATURES, RAW_ODDS_DIR, WITH_MISSINGNESS, historical_features, historical_fold, load_history, source_coverage,
)
from .market import CLASS_NAMES, N_CLASSES, market_probabilities, validate_probabilities
from .nlp_experiments import BLAS_THREADS, PICK_COLUMNS, _development_fingerprint
from .preprocessing import TFIDF_CONFIG, prepare_documents
from .team_historical import OUTPUT as PHASE06B, _candidate
from .team_text import ATTRIBUTION_CONFIG, aligned_articles, side_documents, side_tfidf

OUTPUT = RESULTS_ROOT / "nonlinear"
SEED = 70107
SVD = {"n_components": 32, "algorithm": "randomized", "n_iter": 7, "n_oversamples": 10, "random_state": SEED}
AUTHOR_GRID = (0.1, 1.0, 10.0)
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
CATBOOST = {
    "loss_function": "MultiClass", "classes_count": N_CLASSES, "task_type": "CPU", "boosting_type": "Plain",
    "bootstrap_type": "No", "random_strength": 1.0, "random_seed": SEED, "thread_count": 4,
    "allow_writing_files": False, "verbose": False, "use_best_model": False,
}
CAT_GRID = (
    {"depth": 2, "iterations": 400, "learning_rate": 0.035, "l2_leaf_reg": 30.0},
    {"depth": 4, "iterations": 500, "learning_rate": 0.025, "l2_leaf_reg": 45.0},
)
CAT_BLOCKS = {
    "A": ("book",),
    "B": ("book", "picks"),
    "C": ("book", "picks", "history"),
    "D": ("book", "picks", "history", "author"),
    "E": ("book", "picks", "history", "author", "team_text_svd"),
}
LINEAR_NESTED = (
    ("linear_picks", "linear_market"),
    ("linear_picks_author", "linear_picks"),
    ("linear_history_picks", "linear_picks"),
    ("linear_history_picks_author", "linear_history_picks"),
    ("linear_history_picks_author", "linear_picks_author"),
)
CAT_NESTED = tuple(zip("BCDE", "ABCD", strict=True))


def reduce_side_tfidf(training, validation, *, settings=SVD):
    """Project the comparative term columns using only training data and retain side flags."""
    reducer = TruncatedSVD(**settings)
    train_terms = reducer.fit_transform(training[:, :-2])
    valid_terms = reducer.transform(validation[:, :-2])
    return (np.column_stack((terms, original[:, -2:].toarray()))
            for terms, original in ((train_terms, training), (valid_terms, validation)))


def blend_probabilities(market, nonlinear, alpha):
    market = validate_probabilities(market)
    nonlinear = validate_probabilities(nonlinear)
    if market.shape != nonlinear.shape or not 0 <= alpha <= 1:
        raise ValueError("Blending needs matched H/D/A arrays and an alpha in [0, 1]")
    if alpha == 0:
        return market.copy()
    return validate_probabilities((1 - alpha) * market + alpha * nonlinear)


def ordered_probabilities(classes, probabilities):
    """Reorder model probability columns to outcome codes 0, 1 and 2."""
    classes = np.asarray(classes)
    if classes.shape != (N_CLASSES,) or not np.array_equal(np.sort(classes), np.arange(N_CLASSES)):
        raise ValueError("Expected model probability columns for all H/D/A outcome codes")
    return validate_probabilities(probabilities)[:, np.argsort(classes)]


def _author_fold(market, data, keys, fold):
    positions = np.r_[fold.train, fold.valid]
    train_metadata, valid_metadata = (data.metadata.iloc[p] for p in (fold.train, fold.valid))
    if train_metadata["outcome_available_at"].max() >= valid_metadata["prediction_at"].min():
        raise ValueError("Author histories require fully released training targets before validation")
    return author_features(market, data.metadata.iloc[positions], keys.iloc[positions],
                           data.numerical.iloc[positions], data.target.iloc[positions])


def _coverage(scope, split, authors, historical, sides):
    flags = authors.loc[:, list(MISSING_FLAGS)].sum().astype(int).to_dict()
    counts = authors["author_previous_predictions"]
    return {
        "scope": scope, "split": split, "matches": len(authors), **flags,
        "author_history_1_to_4": int(counts.between(1, 4).sum()),
        "author_history_median": counts.median(), "author_history_p90": counts.quantile(0.9),
        "history_missing_side": int(historical[["home_observed_matches", "away_observed_matches"]].eq(0).any(axis=1).sum()),
        "history_short_last_5": int(historical[["home_observed_matches", "away_observed_matches"]].lt(5).any(axis=1).sum()),
        "history_missing_venue": int(historical[["home_observed_venue_matches", "away_observed_venue_matches"]]
                                     .eq(0).any(axis=1).sum()),
        "text_missing_side": int(sides[["sentences_home", "sentences_away"]].eq(0).any(axis=1).sum())
        if sides is not None else None,
    }


def _prepare(market, data, keys, history, attributed):
    book = market_probabilities(data.numerical, market)
    picks = data.numerical.loc[:, list(PICK_COLUMNS)].to_numpy(dtype=float)
    log_book = np.log(book)
    book_features = np.column_stack((book, log_book))
    folds, stats = _folds(data.metadata, {"market": log_book, "market_and_picks": np.column_stack((log_book, picks))})
    historical = historical_features(data.metadata, history)
    documents = tuple(map(prepare_documents, (attributed.home, attributed.away))) if attributed is not None else None
    coverage, dimensions, validation_authors = [], [], []
    rate_columns = list(AUTHOR_FEATURES[:-len(MISSING_FLAGS)])
    for fold in folds:
        train, valid = fold.train, fold.valid
        if documents is not None:
            _, train_tfidf, valid_tfidf = side_tfidf(documents, train, valid)
            fold.blocks["team_text_svd"] = tuple(reduce_side_tfidf(train_tfidf, valid_tfidf))
            dimensions.append({"scope": fold.season, "block": "side_tfidf_before_svd", "dimensions": train_tfidf.shape[1]})
        fold.blocks["history"] = historical_fold(historical, train, valid)
        authors = _author_fold(market, data, keys, fold)
        train_authors, valid_authors = authors.iloc[:len(train)], authors.iloc[len(train):]
        validation_authors.append(valid_authors)
        scaler = StandardScaler().fit(train_authors[rate_columns])
        fold.blocks["author"] = tuple(
            np.column_stack((scaler.transform(part[rate_columns]), part.loc[:, list(MISSING_FLAGS)]))
            for part in (train_authors, valid_authors)
        )
        fold.blocks["book"] = (book_features[train], book_features[valid])
        fold.blocks["picks"] = (picks[train], picks[valid])
        dimensions.extend({"scope": fold.season, "block": key, "dimensions": pair[0].shape[1]}
                          for key, pair in fold.blocks.items())
        for split, rows, part in (("training", train, train_authors), ("validation", valid, valid_authors)):
            sides = attributed.coverage.iloc[rows] if attributed is not None else None
            coverage.append(_coverage(fold.season, split, part, historical.iloc[rows], sides))
    pooled = pd.concat(validation_authors, verify_integrity=True)
    sides = attributed.coverage.loc[pooled.index] if attributed is not None else None
    coverage.append(_coverage("pooled", "validation", pooled, historical.loc[pooled.index], sides))
    return book, folds, stats, pd.DataFrame(coverage), pd.DataFrame(dimensions)


def _run_linear(data, folds, book, previous):
    candidates = {
        "linear_market": [previous["market_only"]],
        "linear_picks": [previous["market_picks"]],
        "linear_picks_author": [{**previous["market_picks"], "author": value} for value in AUTHOR_GRID],
        "linear_history_picks": [previous["market_history_picks"]],
        "linear_history_picks_author": [{**previous["market_history_picks"], "author": value} for value in AUTHOR_GRID],
    }
    forecasts, settings, rows = {}, {}, []
    inputs = SimpleNamespace(folds=folds, data=data, name="Phase 07 linear")
    for name, grid in candidates.items():
        forecasts[name], settings[name], _, search = _candidate(
            inputs, tuple(grid[0]), name, book, data.target.to_numpy(), grid,
        )
        rows.extend(search)
    return forecasts, settings, pd.DataFrame(rows)


def _run_catboost(classifier, data, folds, raw):
    forecasts, fixed, selections, search, blends = {}, {}, {}, [], []
    target = data.target.loc[raw.index].to_numpy()
    for block, names in CAT_BLOCKS.items():
        matrices = {fold.season: tuple(fold.features(names)) for fold in folds}
        options, scores, rows = [], [], []
        for candidate, params in enumerate(CAT_GRID, start=1):
            parts = {}
            for fold in folds:
                start = perf_counter()
                train, valid = matrices[fold.season]
                model = classifier(**CATBOOST, **params)
                model.fit(train, data.target.iloc[fold.train].to_numpy())
                values = ordered_probabilities(model.classes_, model.predict_proba(valid))
                parts[fold.season] = values
                rows.append({"block": block, "candidate": candidate, "scope": fold.season, "matches": len(valid),
                             "log_loss": log_loss(values, data.target.iloc[fold.valid].to_numpy()).mean(),
                             "seconds": perf_counter() - start})
            frame = _reference(folds, data.metadata.index, lambda fold: parts[fold.season])
            options.append(frame)
            score = log_loss(frame.loc[raw.index].to_numpy(), target).mean()
            scores.append(score)
            rows.append({"block": block, "candidate": candidate, "scope": "pooled", "matches": len(frame),
                         "log_loss": score, "seconds": sum(row["seconds"] for row in rows[-len(folds):])})
        chosen = int(np.argmin(scores))
        search.extend({**row, "selected": row["candidate"] == chosen + 1} for row in rows)
        best = options[chosen].loc[raw.index]
        fixed[f"fixed_{block}"] = options[0]
        anchored = [blend_probabilities(raw.to_numpy(), best.to_numpy(), alpha) for alpha in ALPHAS]
        losses = [log_loss(values, target).mean() for values in anchored]
        alpha_index = int(np.argmin(losses))
        for alpha, loss in zip(ALPHAS, losses, strict=True):
            blends.append({"block": block, "alpha": alpha, "log_loss": loss, "selected": alpha == ALPHAS[alpha_index]})
        forecasts[f"cat_{block}"] = best
        forecasts[f"blend_{block}"] = pd.DataFrame(anchored[alpha_index], index=raw.index, columns=CLASS_NAMES)
        selections[block] = {"candidate": chosen + 1, "parameters": CAT_GRID[chosen], "alpha": ALPHAS[alpha_index]}
    return forecasts, fixed, selections, pd.DataFrame(search), pd.DataFrame(blends)


def _evaluate(data, folds, forecasts, fixed):
    _check_validation_cohort(folds, data.target.index, forecasts | fixed)
    names = [name for name in forecasts if name != "M0"]
    pairs = [(name, "M0") for name in names] + list(LINEAR_NESTED)
    if fixed:
        pairs.extend((f"{prefix}_{model}", f"{prefix}_{baseline}")
                     for prefix in ("cat", "blend", "fixed") for model, baseline in CAT_NESTED)
    metrics, calibration, paired = [], [], []
    for scope, positions in _scopes(folds):
        truth = data.target.iloc[positions]
        aligned = {name: frame.loc[truth.index] for name, frame in forecasts.items()}
        result = evaluate_predictions(truth, aligned)
        metrics.append(result.metrics.assign(scope=scope))
        calibration.append(result.calibration.assign(scope=scope))
        losses = result.paired_log_loss
        raw_loss = log_loss(aligned["M0"].to_numpy(), truth.to_numpy())
        for name, frame in fixed.items():
            losses[name] = log_loss(frame.loc[truth.index].to_numpy(), truth.to_numpy()) - raw_loss
        differences = pd.DataFrame({f"{model} - {baseline}": losses[model] - losses[baseline]
                                    for model, baseline in pairs}, index=truth.index)
        paired.extend({"scope": scope, "comparison": name, **paired_summary(differences[name])} for name in differences)
        if scope == "pooled":
            dates = data.metadata.loc[truth.index]
            intervals = pd.concat([date_block_bootstrap(differences, dates, days) for days in (7, 14)], ignore_index=True)
    return {"metrics": pd.concat(metrics, ignore_index=True), "calibration": pd.concat(calibration, ignore_index=True),
            "paired": pd.DataFrame(paired), "bootstrap": intervals}


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def run(matches_path=MATCHES_PATH, previews_path=PREVIEWS_PATH, raw_dir=RAW_ODDS_DIR,
        cache_root=CACHE_ROOT, output=None, *, linear_only=False):
    start = perf_counter()
    if not linear_only:
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise SystemExit("Install requirements-nonlinear.txt in the existing environment, or run --linear-only.") from error
    previous = json.loads((PHASE06B / "metadata.json").read_text(encoding="utf-8"))
    if previous["selected_team_representation"] != "side_tfidf":
        raise ValueError("Phase 07 expects the side TF-IDF representation selected in Phase 06B")
    data = {market: load_development(market, matches_path, previews_path) for market in ("early", "closing")}
    closing = data["closing"]
    keys = load_author_keys(previews_path, closing.metadata.index)
    if not data["early"].text.equals(closing.text.reindex(data["early"].text.index)):
        raise ValueError("Early documents must be an identical subset of closing documents")
    attributed = None
    if not linear_only:
        articles = aligned_articles(closing.metadata, matches_path, previews_path)
        attributed = side_documents(articles, closing.text, cache_root / "phase06b_attribution.json.gz")
    history = load_history(matches_path, raw_dir)
    if output is None:
        output = OUTPUT / "linear_only" if linear_only else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    _write(source_coverage(history), output, "historical_sources")
    manifest = {
        "development_seasons": DEVELOPMENT, "holdout_seasons_excluded": ["2526", "2627"], "class_order": CLASS_NAMES,
        "author_grid": AUTHOR_GRID, "author_prior_strength": PRIOR_STRENGTH, "author_features": AUTHOR_FEATURES,
        "history_features": HISTORY_FEATURES, "history_missing_flags": WITH_MISSINGNESS, "expert_picks": PICK_COLUMNS,
        "catboost_fixed": CATBOOST, "catboost_grid": CAT_GRID, "catboost_features": CAT_BLOCKS, "alpha_grid": ALPHAS,
        "selection": "Pooled match-weighted development log loss. Select unblended fits first, then alpha. "
                     "Ties use stronger linear penalties, the first tree candidate and the smaller alpha.",
        "attribution": ATTRIBUTION_CONFIG, "tfidf": TFIDF_CONFIG, "svd": SVD,
        "attribution_cache_hit": attributed.cache_hit if attributed is not None else None,
        "attribution_input_sha256": attributed.fingerprint if attributed is not None else None,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "days": [7, 14]},
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
        "catboost_version": None if linear_only else version("catboost"),
        "python": python_version(), "blas_threads": BLAS_THREADS,
        "historical_source_counts": history["source"].value_counts().to_dict(),
        "historical_input_sha256": _fingerprint(pd.util.hash_pandas_object(history, index=False).tolist()),
        "preparation_seconds": perf_counter() - start, "markets": {},
    }
    for market, item in data.items():
        begin = perf_counter()
        index = item.metadata.index
        sides = attributed.reindex(index) if attributed is not None else None
        book, folds, stats, coverage, dimensions = _prepare(market, item, keys.loc[index], history, sides)
        fingerprint = _fingerprint(index[np.concatenate([fold.valid for fold in folds])].tolist())
        prior = previous["market"][market]
        timings = item.metadata[["season", "kickoff", "published", "prediction_at", "outcome_available_at"]]
        input_sha = _development_fingerprint(
            timings.join([item.numerical, item.target.rename("outcome_code")]), {"primary": prepare_documents(item.text)},
        )
        if fingerprint != prior["validation_ids_sha256"] or input_sha != prior["development_inputs_sha256"]:
            raise ValueError(f"{market}: development cohort and model inputs must match Phase 06B")
        prep_seconds = perf_counter() - begin
        begin = perf_counter()
        raw = _reference(folds, index, lambda fold: book[fold.valid])
        linear, settings, linear_search = _run_linear(item, folds, book, prior["selected_penalties"])
        linear_seconds = perf_counter() - begin
        print(f"{market}: prepared in {prep_seconds:.1f}s, linear in {linear_seconds:.1f}s", flush=True)
        forecasts, fixed, cat_settings = {"M0": raw, **linear}, {}, {}
        tables = {"folds": stats, "coverage": coverage, "dimensions": dimensions, "linear_search": linear_search}
        begin = perf_counter()
        if not linear_only:
            trees, fixed, cat_settings, cat_search, blend_search = _run_catboost(CatBoostClassifier, item, folds, raw)
            forecasts.update(trees)
            tables.update(cat_search=cat_search, blend_search=blend_search)
        cat_seconds = perf_counter() - begin
        tables.update(_evaluate(item, folds, forecasts, fixed))
        for name, table in tables.items():
            _write(table, output, f"{market}_{name}")
        manifest["markets"][market] = {
            "validation_ids_sha256": fingerprint, "development_inputs_sha256": input_sha,
            "matches_phase06b_fingerprints": True, "development_matches": len(index), "validation_matches": len(raw),
            "eligible_distinct_bylines": int(keys.loc[index].nunique()),
            "author_input_sha256": _fingerprint(keys.loc[index].fillna("").tolist()),
            "preprocessing_seconds": prep_seconds, "linear_seconds": linear_seconds, "catboost_seconds": cat_seconds,
            "linear": settings, "catboost_and_blend": cat_settings,
        }
        print(f"{market}: CatBoost and blending in {cat_seconds:.1f}s", flush=True)
    manifest["total_seconds"] = perf_counter() - start
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Phase 07 experiments: {manifest['total_seconds']:.1f}s", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("matches-path", MATCHES_PATH), ("previews-path", PREVIEWS_PATH), ("raw-dir", RAW_ODDS_DIR),
                          ("cache-root", CACHE_ROOT), ("output", None)):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--linear-only", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
