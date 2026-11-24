"""Run the Phase 06B development diagnostics and controlled residual comparison."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..analysis import log_loss
from ..parse import FOOTBALL_PICK, parse_pick
from .dataset import FootballDataset, OUTCOME_CODES, _read_source
from .diagnostic_data import load_development, side_orientation
from .diagnostic_metrics import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, categorical_metrics, date_block_bootstrap, paired_summary,
    soft_cross_entropy,
)
from .diagnostics import (
    L2_GRID as DIAGNOSTIC_GRID, RESULTS as PHASE06A, _check_validation_cohort, _class_frequency, _folds, _reference, _scopes,
    _soft_table, _write,
)
from .embeddings import CACHE_ROOT, MiniLMEncoder, _fingerprint, embed_articles
from .experiments import MATCHES_PATH, PREVIEWS_PATH, RESULTS_ROOT
from .historical_features import (
    ELO_K, ELO_SCALE, HISTORY_FEATURES, HOME_ADVANTAGE, INITIAL_ELO, RAW_ODDS_DIR, WITH_MISSINGNESS, historical_features,
    historical_fold, load_history, source_coverage,
)
from .market import N_CLASSES, market_probabilities
from .models import MarketResidual
from .nlp_experiments import BLAS_THREADS, PICK_COLUMNS, _development_fingerprint
from .preprocessing import TFIDF_CONFIG, prepare_documents
from .team_text import (
    ATTRIBUTION_CONFIG, aligned_articles, prepare_review, side_documents, side_tfidf, team_embeddings,
)

OUTPUT = RESULTS_ROOT / "team_historical"
MARKETS = ("early", "closing")
NUMERICAL_GRID = (0.01, 0.1, 1.0, 10.0)
HISTORY_GRID = (0.01, 0.1, 1.0, 10.0)
TEXT_GRID = (0.001, 0.01, 0.1, 1.0)
REPRESENTATIONS = ("global_tfidf", "global_minilm", "side_tfidf", "team_minilm")
TEAM_REPRESENTATIONS = REPRESENTATIONS[2:]
CONTROL_CONFIDENCE = 0.98


@dataclass
class MarketInputs:
    name: str
    data: FootballDataset
    folds: list
    book: np.ndarray
    picks: pd.Series
    verdict: pd.Series


def verdict_picks(metadata, verdict):
    """Parse the explicit scoreline heading and verify its orientation against the fixture."""
    if not verdict.index.equals(metadata.index):
        raise ValueError("Verdict headings must align with fixtures by ID")
    codes = []
    teams = metadata[["home_team", "away_team"]].itertuples(index=False, name=None)
    for heading, (home, away) in zip(verdict.fillna(""), teams, strict=True):
        line = heading.splitlines()[0].strip() if heading else ""
        match = FOOTBALL_PICK.match(line)
        code = np.nan
        if match:
            orientation = side_orientation(match["home"].strip(), match["away"].strip(), home, away)
            if orientation in ("ordered", "reversed"):
                code = OUTCOME_CODES[parse_pick(line, "football")["pick_result"]]
                if orientation == "reversed":
                    code = 2 - code
        codes.append(code)
    return pd.Series(codes, index=metadata.index)


def _candidate(inputs, block_names, name, offset, target, candidates, *, corrections=False):
    """Select pooled OOF loss over a predefined grid, sharing preprocessed matrices."""
    folds, index = inputs.folds, inputs.data.metadata.index
    matrices = [tuple(fold.features(block_names)) for fold in folds]
    weights = [len(fold.valid) for fold in folds]
    best, rows, selected = None, [], None
    for penalties in candidates:
        penalties = dict(penalties)
        parts, fits, losses = [], [], []
        for fold, (training, validation) in zip(folds, matrices, strict=True):
            widths = [fold.blocks[block][0].shape[1] for block in block_names]
            block_penalties = [(width, penalties[block]) for block, width in zip(block_names, widths, strict=True)]
            start = perf_counter()
            model = MarketResidual(1.0, standardize=False)
            fit = model.fit if target.ndim == 1 else model.fit_soft
            fit(offset[fold.train], target[fold.train], training, block_penalties=block_penalties)
            status = model.convergence_
            loss = np.nan
            if status.converged:
                forecast = model.predict_proba(offset[fold.valid], validation)
                truth = target[fold.valid]
                loss = (log_loss(forecast, truth) if target.ndim == 1 else soft_cross_entropy(truth, forecast)).mean()
                parts.append((fold, forecast))
                if corrections:
                    fits.append((fold, model, widths))
            losses.append(loss)
            rows.append({
                "model": name, "scope": fold.season, "matches": len(fold.valid), "loss": loss,
                "converged": status.converged, "iterations": status.iterations, "message": status.message,
                "seconds": perf_counter() - start, "penalties": json.dumps(penalties, sort_keys=True),
            })
        pooled = np.average(losses, weights=weights)
        rows.append({
            "model": name, "scope": "pooled", "matches": sum(weights), "loss": pooled,
            "converged": bool(np.isfinite(pooled)), "iterations": sum(row["iterations"] for row in rows[-len(folds):]),
            "seconds": sum(row["seconds"] for row in rows[-len(folds):]),
            "penalties": json.dumps(penalties, sort_keys=True),
        })
        key = (pooled, tuple(-penalties[block] for block in block_names))
        if np.isfinite(pooled) and (best is None or key < best):
            best = key
            values_by_season = {fold.season: forecast for fold, forecast in parts}
            selected = (_reference(folds, index, lambda fold: values_by_season[fold.season]), penalties, fits)
    if selected is None:
        raise RuntimeError(f"No converged candidate for {inputs.name} {name}")
    for row in rows:
        row["selected"] = row["penalties"] == json.dumps(selected[1], sort_keys=True)
    return (*selected, rows)


def categorical_scopes(inputs, truth, forecasts, reference):
    _check_validation_cohort(inputs.folds, truth.index, forecasts)
    metrics, classwise, confusion = [], [], []
    for scope, rows in _scopes(inputs.folds):
        target = truth.iloc[rows]
        aligned = {name: forecast.loc[target.index] for name, forecast in forecasts.items()}
        for collector, table in zip((metrics, classwise, confusion),
                                    categorical_metrics(target, aligned, reference), strict=True):
            collector.append(table.assign(scope=scope))
    return tuple(pd.concat(tables, ignore_index=True) for tables in (metrics, classwise, confusion))


def _run_diagnostics(inputs, output):
    data, folds, book = inputs.data, inputs.folds, inputs.book
    uniform = np.full_like(book, 1 / N_CLASSES)
    picks = inputs.picks
    searches, pick_metrics = [], None
    for task, target in (("outcome", data.target), ("pick_recovery", picks), ("market_recovery", book)):
        start = perf_counter()
        forecasts = {}
        blocks = {name: name for name in REPRESENTATIONS}
        if task != "pick_recovery":
            blocks["expert_picks"] = "picks"
        for name, block in blocks.items():
            forecasts[name], _, _, rows = _candidate(
                inputs, (block,), name, uniform, target.to_numpy() if target.ndim == 1 else target,
                ({block: l2} for l2 in DIAGNOSTIC_GRID),
            )
            searches.extend({**row, "task": task} for row in rows)
        if task == "market_recovery":
            forecasts["mean_market"] = _reference(
                folds, data.metadata.index, lambda fold: np.tile(book[fold.train].mean(axis=0), (len(fold.valid), 1)),
            )
            result = _soft_table(folds, data.metadata.index, target, forecasts, "mean_market")
        else:
            reference = _class_frequency(folds, target)
            forecasts["class_frequency"] = reference
            if task == "outcome":
                forecasts["raw_market"] = _reference(folds, data.metadata.index, lambda fold: book[fold.valid])
            else:
                valid = inputs.verdict.dropna()
                parsed = valid.index.intersection(reference.index)
                control = reference.copy()
                one_hot = np.eye(N_CLASSES)[valid.loc[parsed].to_numpy(dtype=int)]
                remainder = (1 - CONTROL_CONFIDENCE) / (N_CLASSES - 1)
                control.loc[parsed] = np.where(one_hot == 1, CONTROL_CONFIDENCE, remainder)
                forecasts["verdict_parser_control"] = control
                parser_rows = []
                for scope, rows in _scopes(folds):
                    ids = picks.index[rows]
                    parsed = ids.intersection(valid.index)
                    parser_rows.append({
                        "scope": scope, "matches": len(ids), "parsed": len(parsed),
                        "matches_label": int(valid.loc[parsed].eq(picks.loc[parsed]).sum()),
                        "coverage": len(parsed) / len(ids),
                    })
                _write(pd.DataFrame(parser_rows), output, f"{inputs.name}_verdict_control")
            result, classwise, confusion = categorical_scopes(inputs, target, forecasts, "class_frequency")
            _write(classwise, output, f"{inputs.name}_{task}_classwise")
            if task == "pick_recovery":
                pick_metrics = result
                _write(confusion, output, f"{inputs.name}_{task}_confusion")
        _write(result, output, f"{inputs.name}_{task}_metrics")
        print(f"{inputs.name} {task}: {perf_counter() - start:.1f}s", flush=True)
    _write(pd.DataFrame(searches), output, f"{inputs.name}_diagnostic_search")
    return pick_metrics


def _block_corrections(name, fits, penalties):
    rows = []
    for fold, model, widths in fits:
        block_start = 0
        for block, width in zip(penalties, widths, strict=True):
            correction = fold.blocks[block][1] @ model.coef_[block_start:block_start + width]
            rows.append({
                "model": name, "block": block, "scope": fold.season, "matches": len(fold.valid),
                "rms_logit": np.sqrt(np.square(correction).mean()), "mean_abs_logit": np.abs(correction).mean(),
            })
            block_start += width
        rows.append({
            "model": name, "block": "intercept", "scope": fold.season, "matches": len(fold.valid),
            "rms_logit": np.sqrt(np.square(model.intercept_).mean()), "mean_abs_logit": np.abs(model.intercept_).mean(),
        })
    frame = pd.DataFrame(rows)
    for block, part in frame.groupby("block", sort=False):
        rows.append({
            "model": name, "block": block, "scope": "pooled", "matches": part["matches"].sum(),
            "rms_logit": np.sqrt(np.average(part["rms_logit"] ** 2, weights=part["matches"])),
            "mean_abs_logit": np.average(part["mean_abs_logit"], weights=part["matches"]),
        })
    return rows


def _run_residuals(inputs, selected_team_text, output):
    start = perf_counter()
    data, folds, book, market = inputs.data, inputs.folds, inputs.book, inputs.name
    forecasts = {"raw_market": _reference(folds, data.metadata.index, lambda fold: book[fold.valid])}
    settings, searches, corrections = {}, [], []

    def select(name, fixed, tuned, grid):
        candidates = [{**fixed, tuned: value} for value in grid]
        forecasts[name], settings[name], fits, rows = _candidate(
            inputs, tuple(candidates[0]), name, book, data.target.to_numpy(), candidates, corrections=True,
        )
        searches.extend(rows)
        corrections.extend(_block_corrections(name, fits, settings[name]))

    select("market_only", {}, "market", NUMERICAL_GRID)
    select("market_picks", {}, "market_and_picks", NUMERICAL_GRID)
    for feature in ("global_minilm", *TEAM_REPRESENTATIONS):
        select(f"market_picks_{feature}", settings["market_picks"], feature, TEXT_GRID)
    select("market_history", settings["market_only"], "history", HISTORY_GRID)
    select("market_history_picks", settings["market_picks"], "history", HISTORY_GRID)
    select("market_history_picks_team_text", settings["market_history_picks"], selected_team_text, TEXT_GRID)

    metrics, _, _ = categorical_scopes(inputs, data.target, forecasts, "raw_market")
    _write(metrics, output, f"{market}_residual_metrics")
    _write(pd.DataFrame(searches), output, f"{market}_residual_search")
    _write(pd.DataFrame(corrections), output, f"{market}_block_corrections")
    nested = [(model, "raw_market") for model in settings] + [
        ("market_picks", "market_only"),
        *((f"market_picks_{text}", "market_picks") for text in ("global_minilm", *TEAM_REPRESENTATIONS)),
        *((f"market_picks_{text}", "market_picks_global_minilm") for text in TEAM_REPRESENTATIONS),
        ("market_history", "market_only"),
        ("market_history_picks", "market_history"),
        ("market_history_picks", "market_picks"),
        ("market_history_picks_team_text", "market_history_picks"),
        ("market_history_picks_team_text", f"market_picks_{selected_team_text}"),
    ]
    paired, intervals = [], []
    for scope, positions in _scopes(folds):
        truth = data.target.iloc[positions]
        losses = {name: log_loss(forecast.loc[truth.index].to_numpy(), truth.to_numpy())
                  for name, forecast in forecasts.items()}
        differences = pd.DataFrame({f"{model} - {baseline}": losses[model] - losses[baseline]
                                    for model, baseline in nested}, index=truth.index)
        paired.extend({"scope": scope, "comparison": name, **paired_summary(differences[name])} for name in differences)
        if scope == "pooled":
            intervals.extend(date_block_bootstrap(differences, data.metadata.loc[truth.index], days) for days in (7, 14))
    _write(pd.DataFrame(paired), output, f"{market}_paired")
    _write(pd.concat(intervals, ignore_index=True), output, f"{market}_bootstrap")
    elapsed = perf_counter() - start
    print(f"{market} residuals: {elapsed:.1f}s", flush=True)
    return settings, elapsed


def _write_coverage(market, folds, stats, attributed, history, output):
    attribution_rows, history_rows, feature_rows = [], [], []
    for scope, positions in [("development", np.arange(len(history))), *_scopes(folds)]:
        stats_at = attributed.coverage.iloc[positions]
        counts = stats_at.select_dtypes(include=["number", "bool"]).sum().to_dict()
        counts.update({f"{col}_{key}": count for col in ("source_orientation", "matched_orientation")
                       for key, count in stats_at[col].value_counts().items()})
        home_missing = stats_at["sentences_home"].eq(0)
        away_missing = stats_at["sentences_away"].eq(0)
        attribution_rows.append({
            "scope": scope, "matches": len(positions), **counts,
            "missing_home": int(home_missing.sum()), "missing_away": int(away_missing.sum()),
            "missing_either": int((home_missing | away_missing).sum()), "missing_both": int((home_missing & away_missing).sum()),
        })
        historical = history.iloc[positions]
        for feature, values in historical.items():
            history_rows.append({
                "scope": scope, "feature": feature, "matches": len(positions), "missing": int(values.isna().sum()),
                "zero": int(values.eq(0).sum()), "median": values.median(), "p05": values.quantile(0.05),
                "p95": values.quantile(0.95),
            })
    for fold in folds:
        for block, (matrix, _) in fold.blocks.items():
            feature_rows.append({"scope": fold.season, "block": block, "dimensions": matrix.shape[1]})
    _write(stats, output, f"{market}_folds")
    _write(pd.DataFrame(attribution_rows).fillna(0), output, f"{market}_attribution_coverage")
    _write(pd.DataFrame(history_rows), output, f"{market}_history_coverage")
    _write(pd.DataFrame(feature_rows), output, f"{market}_feature_dimensions")


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def run(matches_path=MATCHES_PATH, previews_path=PREVIEWS_PATH, raw_dir=RAW_ODDS_DIR,
        cache_root=CACHE_ROOT, review_path=PHASE06A / "private" / "attribution_review.csv", output=OUTPUT):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    data = {market: load_development(market, matches_path, previews_path) for market in MARKETS}
    closing = data["closing"]
    early_ids = data["early"].metadata.index
    if not closing.text.index.is_unique or not early_ids.isin(closing.text.index).all() or not data["early"].text.equals(
        closing.text.loc[early_ids]
    ):
        raise ValueError("Expected Phase 06A early documents to be an identical subset of closing documents")
    articles = aligned_articles(closing.metadata, matches_path, previews_path)
    headings = _read_source(previews_path, ("article_id", "pick_text")).set_index("article_id")["pick_text"]
    verdict = verdict_picks(closing.metadata, headings.loc[closing.metadata.index])
    review = prepare_review(review_path, output / "private" / "attribution_review.csv", articles)
    attributed = side_documents(articles, closing.text, cache_root / "phase06b_attribution.json.gz")
    print(f"Attribution ready at {perf_counter() - start:.1f}s (cache hit: {attributed.cache_hit})", flush=True)

    encoder = MiniLMEncoder(cache_root / "model", local_files_only=True)
    global_embedding = embed_articles(closing.text, encoder, cache_root / "articles.sqlite3")
    team_vectors, team_stats = team_embeddings(attributed, encoder, cache_root / "phase06b_team.sqlite3")
    team_vectors = pd.DataFrame(team_vectors, index=closing.metadata.index)
    print(f"Global and team embeddings ready at {perf_counter() - start:.1f}s", flush=True)

    history = load_history(matches_path, raw_dir)
    _write(source_coverage(history), output, "historical_sources")
    historical = historical_features(pd.concat({market: item.metadata for market, item in data.items()}), history)
    manifest = {
        "grids": {"diagnostics": DIAGNOSTIC_GRID, "numerical": NUMERICAL_GRID, "history": HISTORY_GRID, "text": TEXT_GRID},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "days": [7, 14]},
        "development_only": True, "review": review, "attribution": ATTRIBUTION_CONFIG,
        "attribution_cache_hit": attributed.cache_hit, "attribution_input_sha256": attributed.fingerprint,
        "encoder": encoder.config, "encoder_runtime": encoder.runtime, "tfidf": TFIDF_CONFIG,
        "verdict_control": {"source": "previews.csv pick_text", "parsed_class_probability": CONTROL_CONFIDENCE},
        "embeddings": {"global": global_embedding.statistics, "team": team_stats},
        "historical_source_counts": history["source"].value_counts().to_dict(),
        "historical_input_sha256": _fingerprint(pd.util.hash_pandas_object(history, index=False).tolist()),
        "historical_features": HISTORY_FEATURES, "historical_missing_flags": WITH_MISSINGNESS,
        "elo": {"initial": INITIAL_ELO, "home_advantage": HOME_ADVANTAGE, "k": ELO_K, "scale": ELO_SCALE},
        "market": {}, "preparation_seconds": perf_counter() - start,
    }
    prepared, diagnostics = {}, {}
    previous_file = PHASE06A / "metadata.json"
    previous = json.loads(previous_file.read_text(encoding="utf-8")) if previous_file.exists() else {}
    for market, item in data.items():
        begin = perf_counter()
        index = item.metadata.index
        book = market_probabilities(item.numerical, market)
        pick = item.numerical.loc[:, list(PICK_COLUMNS)].to_numpy(dtype=float)
        documents = prepare_documents(item.text)
        sides = attributed.reindex(index)
        side_docs = tuple(prepare_documents(frame) for frame in (sides.home, sides.away))
        features = historical.loc[market]
        if not features.index.equals(index):
            raise ValueError("Historical features must align with the existing market cohort")
        folds, stats = _folds(item.metadata, {
            "market": np.log(book), "picks": pick, "market_and_picks": np.column_stack((np.log(book), pick)),
        }, {"global_tfidf": documents}, {
            "global_minilm": global_embedding.vectors.loc[index].to_numpy(dtype=float),
            "team_minilm": team_vectors.loc[index].to_numpy(dtype=float),
        })
        for fold in folds:
            _, train_side, valid_side = side_tfidf(side_docs, fold.train, fold.valid)
            fold.blocks["side_tfidf"] = (train_side, valid_side)
            fold.blocks["history"] = historical_fold(features, fold.train, fold.valid)
        _write_coverage(market, folds, stats, sides, features, output)
        picks = pd.Series(np.where(pick[:, 0] == 1, 0, np.where(pick[:, 1] == 1, 1, 2)), index=index)
        inputs = MarketInputs(market, item, folds, book, picks, verdict.loc[index])
        prepared[market] = inputs
        eligible = np.concatenate([fold.valid for fold in folds])
        timings = item.metadata[["season", "kickoff", "published", "prediction_at", "outcome_available_at"]]
        fingerprint = _development_fingerprint(timings.join([item.numerical, item.target.rename("outcome_code")]),
                                               {"primary": documents})
        previous_fingerprint = previous.get(market, {}).get("primary_development_sha256")
        manifest["market"][market] = {
            "eligibility_counts": item.counts, "matches_removed_for_features": 0, "validation_matches": len(eligible),
            "validation_ids_sha256": _fingerprint(index[eligible].tolist()), "development_inputs_sha256": fingerprint,
            "matches_phase06a_development_fingerprint": (
                fingerprint == previous_fingerprint if previous_fingerprint is not None else None
            ), "fold_preprocessing_seconds": perf_counter() - begin,
        }
        begin = perf_counter()
        diagnostics[market] = _run_diagnostics(inputs, output)
        manifest["market"][market]["diagnostic_seconds"] = perf_counter() - begin
    # Predefine the larger cohort's pick-recovery log loss as the representation selection criterion.
    directional = diagnostics["closing"].query("scope == 'pooled' and model in @TEAM_REPRESENTATIONS")
    selected = directional.sort_values(["log_loss", "model"]).iloc[0]["model"]
    manifest["selected_team_representation"] = selected
    manifest["team_selection"] = "Lowest pooled closing-cohort expert-pick recovery log loss"
    for market, inputs in prepared.items():
        penalties, elapsed = _run_residuals(inputs, selected, output)
        manifest["market"][market].update(selected_penalties=penalties, residual_seconds=elapsed)
    manifest["total_seconds"] = perf_counter() - start
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("matches-path", MATCHES_PATH), ("previews-path", PREVIEWS_PATH), ("raw-dir", RAW_ODDS_DIR),
                          ("cache-root", CACHE_ROOT), ("review-path", PHASE06A / "private" / "attribution_review.csv"),
                          ("output", OUTPUT)):
        parser.add_argument(f"--{name}", type=Path, default=default)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
