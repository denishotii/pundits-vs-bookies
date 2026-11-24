"""Plan the frozen evaluation, or score historical holdouts with explicit approval."""

import argparse
import csv
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .dataset import MATCH_DURATION_BUFFER, ODDS_COLUMNS, OUTCOME_CODES, load_football
from .diagnostic_metrics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, date_block_bootstrap
from .embeddings import CACHE_ROOT, _fingerprint
from .evaluation import CALIBRATION_BINS, evaluate_predictions
from .experiments import MATCHES_PATH, PREVIEWS_PATH, RESULTS_ROOT
from .market import CLASS_NAMES, market_probabilities
from .models import MAX_ITER, TOL, MarketResidual
from .nlp_experiments import BLAS_THREADS, PICK_COLUMNS, _development_fingerprint
from .preprocessing import PRIMARY_TEXT_COLUMNS, TFIDF_CONFIG, UNICODE_FORM, prepare_documents
from .team_text import ATTRIBUTION_CONFIG, aligned_articles, side_documents, side_tfidf
from .validation import FOOTBALL_SPLITS

OUTPUT = RESULTS_ROOT / "final"
FREEZE = OUTPUT / "frozen_models.json"
FREEZE_SHA256 = "db9346b6728d14dc33413a080ed9d1d20fd3ea1c96594092866901fa54081705"
MARKETS = tuple(ODDS_COLUMNS)
DEVELOPMENT = FOOTBALL_SPLITS["train"]
HOLDOUTS = (*FOOTBALL_SPLITS["test"], *FOOTBALL_SPLITS["check"])
MODELS = ("F0", "F1", "F2", "F3")
COMPARISONS = (("F3", "F2"), ("F2", "F1"), ("F1", "F0"), ("F3", "F0"))
BLOCK_DAYS = (7, 14)
TIMING_COLUMNS = ("season", "kickoff", "published", "prediction_at", "outcome_available_at")
AUDIT_FLAGS = (
    "timestamp_conflict",
    "potential_post_prediction_content_modification",
    "confirmed_outcome_information_contamination",
)
CONTAMINATION = AUDIT_FLAGS[-1]
EXCLUSION_RULES = ("metadata_missing", "metadata_unverifiable", *AUDIT_FLAGS)
AUDIT_COLUMNS = ("article_id", "market", "metadata_status", *AUDIT_FLAGS, "evidence_reference")
VERSION_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "tzdata")


def _json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def runtime_versions():
    return {"python": python_version(), **{package: version(package) for package in VERSION_PACKAGES}}


def _runtime_settings():
    return {
        "class_codes": OUTCOME_CODES, "probability_columns": CLASS_NAMES, "odds_columns": ODDS_COLUMNS,
        "expert_pick_columns": PICK_COLUMNS, "development_seasons": DEVELOPMENT, "historical_seasons": HOLDOUTS,
        "attribution": ATTRIBUTION_CONFIG, "primary_sections": PRIMARY_TEXT_COLUMNS, "tfidf": TFIDF_CONFIG,
        "unicode_normalization": UNICODE_FORM, "blas_threads": BLAS_THREADS,
        "optimizer_maxiter": MAX_ITER, "optimizer_gtol": TOL,
        "outcome_buffer_hours": MATCH_DURATION_BUFFER / pd.Timedelta(hours=1),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "block_days": BLOCK_DAYS},
        "calibration_bins": CALIBRATION_BINS, "comparisons": COMPARISONS,
        "audit_flags": AUDIT_FLAGS, "exclusion_rules": EXCLUSION_RULES,
    }


def load_freeze(path=FREEZE):
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if _json_hash(frozen) != FREEZE_SHA256:
        raise ValueError("Configuration does not match the Phase 08 freeze")
    if _json_hash(frozen["implementation_settings"]) != _json_hash(_runtime_settings()):
        raise ValueError("Runtime settings have drifted from the Phase 08 freeze")
    if frozen["software_versions"] != runtime_versions():
        raise ValueError("Runtime versions do not match the frozen software versions")
    return frozen


def _read_outcomes(path, seasons):
    # The development pass accesses result fields only for development seasons.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"article_id", "season", "result"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected article_id, season and result in the matched export")
        results = [(row["article_id"].strip(), row["result"]) for row in reader if row["season"] in seasons]
    frame = pd.DataFrame(results, columns=["article_id", "result"]).set_index("article_id")
    if not frame.index.is_unique:
        raise ValueError("Expected unique article IDs in outcome rows")
    return frame["result"].map(OUTCOME_CODES).rename("outcome_code")


def _load_inputs(market, seasons, matches_path, previews_path):
    data = load_football(market, seasons=seasons, include_outcomes=False,
                         matches_path=matches_path, previews_path=previews_path)
    if any(not frame.index.equals(data.metadata.index) for frame in (data.numerical, data.text)):
        raise ValueError("Fixture, odds and text rows must align")
    data.metadata = data.metadata.set_index("article_id")
    data.numerical, data.text = (frame.set_axis(data.metadata.index) for frame in (data.numerical, data.text))
    return data


def _development_inputs(frozen, matches_path, previews_path):
    outcomes = _read_outcomes(matches_path, DEVELOPMENT)
    development = {}
    for market in MARKETS:
        data = _load_inputs(market, DEVELOPMENT, matches_path, previews_path)
        labels = outcomes.reindex(data.metadata.index).dropna().astype(int)
        metadata, numerical, sections = (frame.loc[labels.index] for frame in (data.metadata, data.numerical, data.text))
        fingerprints = frozen["data_fingerprints"]["markets"][market]
        effective_input = metadata[list(TIMING_COLUMNS)].join([numerical, labels])
        if (_development_fingerprint(effective_input, {"primary": prepare_documents(sections)})
                != fingerprints["development_inputs_sha256"]
                or _fingerprint(metadata.index[metadata["season"].isin(DEVELOPMENT[1:])].tolist())
                != fingerprints["validation_ids_sha256"]
                or len(metadata) != fingerprints["development_matches"]):
            raise ValueError(f"{market}: development inputs differ from Phase 07")
        development[market] = (metadata, numerical, sections, labels)
    closing, _, sections, _ = development["closing"]
    articles = aligned_articles(closing, matches_path, previews_path)
    sides = side_documents(articles, sections, CACHE_ROOT / "phase06b_attribution.json.gz")
    if sides.fingerprint != frozen["data_fingerprints"]["development_attribution_sha256"]:
        raise ValueError("Development attribution inputs differ from Phase 07")
    return development, sides


def _training_rows(development, test, check):
    for metadata, seasons in ((development, DEVELOPMENT), (test, HOLDOUTS[:1]), (check, HOLDOUTS[1:])):
        if set(metadata["season"]) != set(seasons) or metadata.index.hasnans or not metadata.index.is_unique:
            raise ValueError("Each partition must contain its specified seasons and unique match IDs")
        for column in ("prediction_at", "kickoff", "outcome_available_at"):
            times = metadata[column]
            if not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any():
                raise ValueError("Expected nonmissing timezone-aware fixture timings")
        if (metadata["prediction_at"].gt(metadata["kickoff"])
                | metadata["outcome_available_at"].le(metadata["kickoff"])).any():
            raise ValueError("Expected prediction_at <= kickoff < outcome_available_at")
    if not pd.concat([development, test, check]).index.is_unique:
        raise ValueError("Development, test and check IDs must be disjoint")
    cutoff = test["prediction_at"].min()
    if test["prediction_at"].max() >= check["prediction_at"].min():
        raise ValueError("The check prediction period must follow the test period")
    eligible = development["outcome_available_at"].lt(cutoff)
    training = development.loc[eligible]
    if training.empty:
        raise ValueError("No development outcomes were available strictly before the first holdout prediction")
    return training.index, {
        "first_test_prediction_at": cutoff.isoformat(),
        "first_check_prediction_at": check["prediction_at"].min().isoformat(),
        "last_training_outcome_available_at": training["outcome_available_at"].max().isoformat(),
        "development_candidates": len(development), "training_matches": len(training),
        "purged_at_or_after_cutoff": int((~eligible).sum()),
        "training_ids_sha256": _fingerprint(training.index.tolist()),
    }


@threadpool_limits.wrap(limits=BLAS_THREADS, user_api="blas")
def _fit_forecasts(market, model_settings, numerical, labels, documents):
    """Fit once on the label IDs and apply the fixed transforms to both holdouts."""
    if (not numerical.index.is_unique or not labels.index.is_unique or not labels.index.isin(numerical.index).all()
            or any(not side.index.equals(numerical.index) for side in documents)):
        raise ValueError("Training labels and side documents must align with the fixture IDs")
    train = numerical.index.get_indexer(labels.index)
    valid = np.flatnonzero(~numerical.index.isin(labels.index))
    book = market_probabilities(numerical, market)
    log_book = np.log(book)
    picks = numerical.loc[:, list(PICK_COLUMNS)].to_numpy(dtype=float)
    numeric = {"F1": log_book, "F2": np.column_stack((log_book, picks))}
    for model, values in numeric.items():
        scaler = StandardScaler().fit(values[train])
        numeric[model] = (scaler.transform(values[train]), scaler.transform(values[valid]))
    _, train_text, holdout_text = side_tfidf(documents, train, valid)
    numeric["F3"] = tuple(sparse.hstack((values, text), format="csr")
                          for values, text in zip(numeric["F2"], (train_text, holdout_text), strict=True))
    forecast = {"F0": book[valid]}
    convergence = {}
    for model, (training, holdout) in numeric.items():
        widths = (numeric["F2"][0].shape[1], train_text.shape[1]) if model == "F3" else (training.shape[1],)
        penalties = tuple(zip(widths, model_settings[model]["block_penalties"], strict=True))
        residual = MarketResidual(l2=1.0, standardize=False).fit(
            book[train], labels.to_numpy(), training, block_penalties=penalties,
        )
        if not residual.convergence_.converged:
            raise RuntimeError(f"{market} {model}: optimizer failed: {residual.convergence_.message}")
        convergence[model] = residual.convergence_.iterations
        forecast[model] = residual.predict_proba(book[valid], holdout)
    return {model: pd.DataFrame(p, index=numerical.index[valid], columns=CLASS_NAMES)
            for model, p in forecast.items()}, convergence


def read_integrity_audit(path):
    """Read the separately adjudicated market-specific audit table."""
    audit = pd.read_csv(path, dtype="string", keep_default_na=False)
    if not set(AUDIT_COLUMNS).issubset(audit.columns):
        raise ValueError(f"Integrity audit requires {', '.join(AUDIT_COLUMNS)}")
    audit = audit.loc[:, list(AUDIT_COLUMNS)].apply(lambda column: column.str.strip())
    if (audit["article_id"].eq("").any() or not audit["market"].isin(MARKETS).all()
            or not audit["metadata_status"].isin(("verified", "unverifiable", "missing")).all()
            or not audit[list(AUDIT_FLAGS)].isin(("yes", "no", "unknown")).all().all()):
        raise ValueError("Invalid integrity audit IDs, markets or statuses")
    if (audit[CONTAMINATION].eq("yes") & audit["evidence_reference"].eq("")).any():
        raise ValueError("Confirmed contamination requires an evidence reference")
    audit = audit.set_index(["market", "article_id"])
    if not audit.index.is_unique:
        raise ValueError("Expected unique market and article pairs in the integrity audit")
    return audit


def _integrity_cohort(audit, market, ids, rules):
    rows = audit.reindex(pd.MultiIndex.from_product([[market], ids], names=audit.index.names))
    rows.index = ids
    flags = rows[list(AUDIT_FLAGS)].fillna("unknown")
    metadata = rows["metadata_status"].fillna("missing")
    reasons = flags.eq("yes").assign(metadata_missing=metadata.eq("missing"),
                                     metadata_unverifiable=metadata.eq("unverifiable"))
    excluded = reasons[list(rules)].any(axis=1)
    return ids[~excluded], {
        "original_matches": len(ids), "screened_matches": int((~excluded).sum()),
        "excluded_matches": int(excluded.sum()), "excluded_if_any": list(rules),
        "missing_audit_rows": int(rows["metadata_status"].isna().sum()),
        "metadata_status_counts": {key: int(metadata.eq(key).sum()) for key in ("verified", "unverifiable", "missing")},
        "flag_counts": {flag: {key: int(flags[flag].eq(key).sum()) for key in ("yes", "no", "unknown")}
                        for flag in AUDIT_FLAGS},
        "exclusions_by_selected_rule_overlapping": reasons[list(rules)].sum().astype(int).to_dict(),
    }


def _approval(path, audit_sha, inputs):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or policy.get("status") != "approved"
            or not isinstance(policy.get("approved_by"), str) or not policy["approved_by"].strip()
            or policy.get("freeze_sha256") != FREEZE_SHA256 or policy.get("audit_sha256") != audit_sha
            or policy.get("input_sha256") != inputs):
        raise ValueError("Integrity approval must bind an approver, the freeze, audit and exact input files")
    approved_at = pd.Timestamp(policy.get("approved_at"))
    if pd.isna(approved_at) or approved_at.tzinfo is None or approved_at > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Integrity approval requires a timezone-aware approval timestamp")
    rules = policy.get("exclude_if_any")
    if not isinstance(rules, dict) or set(rules) != set(MARKETS):
        raise ValueError("Integrity approval must specify exclusion rules for both markets")
    for values in rules.values():
        if (not isinstance(values, list) or not all(isinstance(value, str) for value in values)
                or len(set(values)) != len(values) or not set(values).issubset(EXCLUSION_RULES)
                or CONTAMINATION not in values):
            raise ValueError("Invalid integrity exclusion rules; confirmed contamination must be screened")
    return policy


def _score(target, forecasts, metadata):
    if set(forecasts) != set(MODELS):
        raise ValueError("Scoring requires precisely the four frozen models")
    # The shared evaluator calls its raw market reference M0.
    scores = evaluate_predictions(target, {("M0" if key == "F0" else key): frame for key, frame in forecasts.items()})
    differences = scores.paired_log_loss.rename(columns={"M0": "F0"})
    paired = pd.DataFrame({f"{model}-{reference}": differences[model] - differences[reference]
                           for model, reference in COMPARISONS})
    intervals = pd.concat([date_block_bootstrap(paired, metadata, days) for days in BLOCK_DAYS], ignore_index=True)
    paired_summary = paired.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    paired_summary["fraction_improved"] = paired.lt(0).mean()
    for frame in (scores.metrics, scores.classwise, scores.calibration):
        frame["model"] = frame["model"].replace({"M0": "F0"})
    return {
        "metrics": scores.metrics.rename(columns={"mean_log_loss_difference_vs_M0": "mean_log_loss_difference_vs_F0"}),
        "classwise": scores.classwise, "calibration": scores.calibration,
        "paired_summary": paired_summary.rename_axis("comparison").reset_index(), "bootstrap": intervals,
        "paired_per_match": paired,
    }


def _write_scores(directory, target, forecasts, metadata):
    if target.empty:
        return
    directory.mkdir(parents=True)
    selected = {model: frame.loc[target.index] for model, frame in forecasts.items()}
    for table, frame in _score(target, selected, metadata.loc[target.index]).items():
        path = directory / ("private/paired_per_match.csv" if table == "paired_per_match" else f"{table}.csv")
        path.parent.mkdir(exist_ok=True)
        frame.to_csv(path, index=table == "paired_per_match")


def _forecast_market(market, frozen, development, sides, matches_path, previews_path, cache):
    loaded = {season: _load_inputs(market, (season,), matches_path, previews_path) for season in HOLDOUTS}
    metadata, numerical, _, labels = development
    train, fit = _training_rows(metadata, *(item.metadata for item in loaded.values()))
    holdouts = pd.concat([item.metadata for item in loaded.values()], verify_integrity=True)
    holdout_text = pd.concat([item.text for item in loaded.values()])
    articles = aligned_articles(holdouts, matches_path, previews_path)
    attributed = side_documents(articles, holdout_text, cache / f"{market}_attribution.json.gz")
    fit["holdout_attribution_input_sha256"] = attributed.fingerprint
    documents = tuple(prepare_documents(pd.concat([dev.loc[train], held], verify_integrity=True))
                      for dev, held in ((sides.home, attributed.home), (sides.away, attributed.away)))
    combined = pd.concat([numerical.loc[train], *(item.numerical for item in loaded.values())], verify_integrity=True)
    forecasts, fit["residual_optimizer_iterations"] = _fit_forecasts(
        market, frozen["markets"][market], combined, labels.loc[train], documents,
    )
    if any(not frame.index.equals(holdouts.index) for frame in forecasts.values()):
        raise ValueError("Every model must predict the exact holdout fixture IDs")
    return loaded, forecasts, fit


def _input_hashes(matches_path, previews_path):
    return {"matches": _file_hash(matches_path), "previews": _file_hash(previews_path)}


def evaluation_plan(frozen, matches_path, previews_path, audit_path=None, approval_path=None):
    plan = {
        "holdout_outcomes_evaluated": False, "freeze_sha256": FREEZE_SHA256,
        "markets": list(MARKETS), "models": list(MODELS), "development_seasons": list(DEVELOPMENT),
        "historical_test": HOLDOUTS[0], "historical_check": HOLDOUTS[1],
        "fits_per_market": {"F0": 0, "F1": 1, "F2": 1, "F3": 1},
        "protocol": "results/advanced/final/evaluation_protocol.md",
        "data_fingerprints": frozen["data_fingerprints"],
        "integrity_policy_approved": False,
    }
    if approval_path and not audit_path:
        raise ValueError("An approval requires its integrity audit table")
    if audit_path:
        audit_sha = _file_hash(audit_path)
        read_integrity_audit(audit_path)
        if audit_sha != _file_hash(audit_path):
            raise ValueError("Integrity audit changed during verification")
        plan.update(audit_sha256=audit_sha, input_sha256=_input_hashes(matches_path, previews_path))
        if approval_path:
            plan["integrity_approval"] = _approval(approval_path, audit_sha, plan["input_sha256"])
            plan["integrity_policy_approved"] = True
    return plan


def run_evaluation(*, execute_holdouts=False, matches_path=MATCHES_PATH, previews_path=PREVIEWS_PATH,
                   audit_path=None, approval_path=None, output=OUTPUT / "evaluation"):
    if execute_holdouts is not True:
        raise ValueError("Historical outcome evaluation requires --execute-holdouts")
    if not audit_path or not approval_path:
        raise ValueError("Historical scoring requires an integrity audit and frozen approval for both markets")
    if output.exists():
        raise FileExistsError("Choose a new evaluation output directory")
    frozen = load_freeze()
    manifest = evaluation_plan(frozen, matches_path, previews_path, audit_path, approval_path)
    rules = manifest["integrity_approval"]["exclude_if_any"]
    audit = read_integrity_audit(audit_path)
    if _file_hash(audit_path) != manifest["audit_sha256"]:
        raise ValueError("Integrity audit changed after approval verification")
    development, sides = _development_inputs(frozen, matches_path, previews_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as cache, TemporaryDirectory(dir=output.parent) as staging:
        scored = Path(staging) / "scored"
        prepared = {market: _forecast_market(market, frozen, development[market], sides,
                                             matches_path, previews_path, Path(cache)) for market in MARKETS}
        if _input_hashes(matches_path, previews_path) != manifest["input_sha256"]:
            raise ValueError("Input files changed during forecast preparation")
        outcomes = _read_outcomes(matches_path, HOLDOUTS)
        manifest["evaluation"] = {}
        for market, (loaded, forecasts, fit) in prepared.items():
            manifest["evaluation"][market] = {"fit": fit, "seasons": {}}
            for season, item in loaded.items():
                metadata = item.metadata
                target = outcomes.reindex(metadata.index).dropna().astype(int)
                if target.empty:
                    raise ValueError(f"{market} {season}: no eligible historical outcomes")
                screened, counts = _integrity_cohort(audit, market, target.index, rules[market])
                manifest["evaluation"][market]["seasons"][season] = {
                    "input_eligibility": item.counts, "missing_or_invalid_outcomes": len(metadata) - len(target),
                    "first_scored_kickoff": metadata.loc[target.index, "kickoff"].min().isoformat(),
                    "last_scored_kickoff": metadata.loc[target.index, "kickoff"].max().isoformat(),
                    "integrity": counts, "original_ids_sha256": _fingerprint(target.index.tolist()),
                    "screened_ids_sha256": _fingerprint(screened.tolist()),
                }
                directory = scored / market / season
                _write_scores(directory / "original", target, forecasts, metadata)
                _write_scores(directory / "integrity_screened", target.loc[screened], forecasts, metadata)
        if _input_hashes(matches_path, previews_path) != manifest["input_sha256"]:
            raise ValueError("Input files changed during historical scoring")
        manifest["holdout_outcomes_evaluated"] = True
        manifest["evaluation_generated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        (scored / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        scored.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-holdouts", action="store_true")
    mode.add_argument("--verify-development", action="store_true", help="Check development fingerprints without fitting")
    parser.add_argument("--integrity-audit", type=Path)
    parser.add_argument("--integrity-approval", type=Path)
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    parser.add_argument("--previews", type=Path, default=PREVIEWS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT / "evaluation")
    args = parser.parse_args(argv)
    try:
        if args.execute_holdouts:
            run_evaluation(execute_holdouts=True, matches_path=args.matches, previews_path=args.previews,
                           audit_path=args.integrity_audit, approval_path=args.integrity_approval, output=args.output)
            print(f"Historical evaluation saved to {args.output}")
        else:
            frozen = load_freeze()
            plan = evaluation_plan(frozen, args.matches, args.previews, args.integrity_audit, args.integrity_approval)
            if args.verify_development:
                _development_inputs(frozen, args.matches, args.previews)
                plan["development_fingerprints_verified"] = True
            print(json.dumps(plan, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
