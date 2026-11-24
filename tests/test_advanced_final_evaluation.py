import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from pvb.advanced import final_evaluation as final
from pvb.advanced.dataset import TEXT_COLUMNS
from pvb.advanced.diagnostic_data import load_development


def test_frozen_configuration_matches_development_selection(tmp_path, monkeypatch):
    frozen = final.load_freeze()
    assert {
        market: [models[model]["block_penalties"] for model in final.MODELS]
        for market, models in frozen["markets"].items()
    } == {"early": [[], [1.0], [10.0], [10.0, 1.0]], "closing": [[], [0.1], [1.0], [1.0, 1.0]]}
    assert frozen["implementation_settings"]["class_codes"] == {"H": 0, "D": 1, "A": 2}
    assert frozen["training"]["hyperparameter_search"] is False
    changed = tmp_path / "changed.json"
    frozen["markets"]["early"]["F3"]["block_penalties"][-1] = 0.1
    changed.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="Phase 08 freeze"):
        final.load_freeze(changed)
    monkeypatch.setattr(final, "CALIBRATION_BINS", 9)
    with pytest.raises(ValueError, match="Runtime settings"):
        final.load_freeze()


def test_explicit_execution_guard_and_default_plan(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Plan mode accessed fixture data")

    monkeypatch.setattr(final, "_load_inputs", unexpected)
    monkeypatch.setattr(final, "_read_outcomes", unexpected)
    with pytest.raises(ValueError, match="--execute-holdouts"):
        final.run_evaluation()
    with pytest.raises(ValueError, match="integrity audit and frozen approval"):
        final.run_evaluation(execute_holdouts=True)
    final.main([])
    plan = json.loads(capsys.readouterr().out)
    assert plan["holdout_outcomes_evaluated"] is False
    assert plan["historical_test"] == "2526"
    assert plan["historical_check"] == "2627"


def test_training_uses_strict_release_cutoff_and_disjoint_partitions():
    def timing(seasons, predictions, releases, prefix):
        result = pd.DataFrame({
            "season": seasons,
            "prediction_at": pd.to_datetime(predictions, utc=True),
            "outcome_available_at": pd.to_datetime(releases, utc=True),
        }, index=[f"{prefix}{i}" for i in range(len(seasons))])
        return result.assign(kickoff=result["prediction_at"])

    development = timing(list(final.DEVELOPMENT) + ["2425"],
                         ["2022-05-01", "2023-05-01", "2024-05-01", "2025-05-01", "2025-05-01"],
                         ["2022-05-02", "2023-05-02", "2024-05-02", "2025-05-02", "2025-08-01"], "dev")
    test = timing(["2526"], ["2025-08-01"], ["2025-08-02"], "test")
    check = timing(["2627"], ["2026-08-01"], ["2026-08-02"], "check")
    ids, record = final._training_rows(development, test, check)
    assert ids.tolist() == ["dev0", "dev1", "dev2", "dev3"]
    assert record["purged_at_or_after_cutoff"] == 1
    with pytest.raises(ValueError, match="disjoint"):
        final._training_rows(development, test, check.set_axis(test.index))
    with pytest.raises(ValueError, match="must follow"):
        final._training_rows(development, test, test.assign(season="2627").set_axis(check.index))


def _audit_entry(article, market="early", status="verified", flags=("no", "no", "no"), evidence=""):
    return dict(zip(final.AUDIT_COLUMNS, (article, market, status, *flags, evidence), strict=True))


def test_integrity_rules_cover_missing_metadata_without_automatic_timestamp_exclusion(tmp_path):
    path = tmp_path / "audit.csv"
    rows = [
        {**_audit_entry("modified_after_kickoff"), "dateModified": "2026-12-31T00:00:00Z"},
        _audit_entry("potential", "early", "unverifiable", ("yes", "yes", "unknown")),
        _audit_entry("potential", "closing"),
        _audit_entry("confirmed", flags=("no", "yes", "yes"), evidence="archive:123"),
        _audit_entry("unverifiable", status="unverifiable", flags=("unknown", "unknown", "unknown")),
        _audit_entry("explicit_missing", status="missing", flags=("unknown", "unknown", "unknown")),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    audit = final.read_integrity_audit(path)
    ids = pd.Index(["modified_after_kickoff", "potential", "confirmed", "unverifiable", "explicit_missing", "absent"])
    screened, counts = final._integrity_cohort(audit, "early", ids,
                                               [final.CONTAMINATION, "timestamp_conflict", final.AUDIT_FLAGS[1]])
    assert screened.tolist() == ["modified_after_kickoff", "unverifiable", "explicit_missing", "absent"]
    assert counts["excluded_matches"] == 2
    assert counts["missing_audit_rows"] == 1
    assert counts["metadata_status_counts"] == {"verified": 2, "unverifiable": 2, "missing": 2}
    assert counts["exclusions_by_selected_rule_overlapping"][final.AUDIT_FLAGS[1]] == 2
    assert counts["flag_counts"][final.CONTAMINATION] == {"yes": 1, "no": 1, "unknown": 4}
    screened, _ = final._integrity_cohort(audit, "early", ids,
                                         [final.CONTAMINATION, "metadata_missing", "metadata_unverifiable"])
    assert screened.tolist() == ["modified_after_kickoff"]
    screened, _ = final._integrity_cohort(audit, "closing", ids, [final.CONTAMINATION, "timestamp_conflict"])
    assert screened.equals(ids)
    rows[3]["evidence_reference"] = ""
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="evidence reference"):
        final.read_integrity_audit(path)


def test_metrics_and_bootstrap_on_identical_samples():
    ids = pd.Index(["one", "two"], name="article_id")
    target = pd.Series([0, 2], index=ids)
    dates = pd.DataFrame({"season": ["2526", "2526"],
                          "kickoff": pd.to_datetime(["2025-08-02", "2025-08-23"], utc=True)}, index=ids)
    f0 = pd.DataFrame([[0.6, 0.2, 0.2], [0.5, 0.25, 0.25]], index=ids, columns=final.CLASS_NAMES)
    f3 = pd.DataFrame([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]], index=ids, columns=final.CLASS_NAMES)
    forecasts = {"F0": f0, "F1": f0, "F2": f0, "F3": f3.iloc[::-1]}
    scores = final._score(target, forecasts, dates)
    f0_metrics = scores["metrics"].set_index("model").loc["F0"]
    assert f0_metrics["log_loss"] == pytest.approx((-np.log(0.6) - np.log(0.25)) / 2)
    assert f0_metrics["brier"] == pytest.approx((0.24 + 0.875) / 2)
    assert f0_metrics["accuracy"] == 0.5
    assert scores["paired_per_match"]["F3-F2"].to_numpy() == pytest.approx(np.log([0.6 / 0.8, 0.25 / 0.6]))
    assert set(scores["bootstrap"]["block_days"]) == {7, 14}
    assert set(scores["bootstrap"]["comparison"]) == {"F3-F2", "F2-F1", "F1-F0", "F3-F0"}
    classwise = scores["classwise"]
    away = classwise.loc[classwise["model"].eq("F3") & classwise["class"].eq("away")].iloc[0]
    assert away["recall"] == 1
    assert scores["calibration"].groupby(["model", "class"])["matches"].sum().eq(2).all()
    with pytest.raises(ValueError, match="exactly the target match IDs"):
        final._score(target, {**forecasts, "F3": f3.set_axis(["one", "extra"])}, dates)


def _synthetic_exports(tmp_path):
    matches, previews = [], []
    parts = ["2022-04-09", "2023-04-08", "2024-04-06", "2025-04-05", "2025-08-16", "2026-08-15"]
    for season, date in zip((*final.DEVELOPMENT, *final.HOLDOUTS), parts, strict=True):
        for i in range(9):
            common = {"article_id": f"{season}-{i}", "published": date + "T10:00:00Z",
                      "side_a": "Alpha", "side_b": "Beta"}
            outcome = ("H", "D", "A")[i % 3]
            row = {**common, "season": season, "league": "E0", "home_team": "Alpha", "away_team": "Beta",
                   "kickoff": date + f"T15:{i:02d}:00Z", "result": outcome, "pick_result": outcome,
                   "pick_home_goals": (2, 1, 0)[i % 3], "pick_away_goals": 1}
            row["published"] = (pd.Timestamp(row["published"]) - pd.Timedelta(days=2)).isoformat()
            for market, columns in final.ODDS_COLUMNS.items():
                values = (1.7 + 0.1 * i, 3.1, 4.0) if market == "early" else (2.0 + 0.1 * i, 3.0, 3.8)
                row.update(zip(columns, values, strict=True))
            matches.append(row)
            text = dict.fromkeys(TEXT_COLUMNS, "")
            vocabulary = "developmentword" if season in final.DEVELOPMENT else "holdoutword"
            text["text_preview"] = f"Alpha attack {vocabulary}. Beta defend firmly."
            previews.append({**common, **text, "sport": "football", "published": row["published"]})
    match_path, preview_path = tmp_path / "matches.csv", tmp_path / "previews.csv"
    pd.DataFrame(matches).to_csv(match_path, index=False)
    pd.DataFrame(previews).to_csv(preview_path, index=False)
    return match_path, preview_path


def _synthetic_fingerprints(frozen, match_path, preview_path, cache):
    for market in final.MARKETS:
        data = load_development(market, match_path, preview_path)
        metadata = data.metadata
        inputs = metadata[list(final.TIMING_COLUMNS)].join([data.numerical, data.target.rename("outcome_code")])
        frozen["data_fingerprints"]["markets"][market] = {
            "development_matches": len(metadata),
            "development_inputs_sha256": final._development_fingerprint(
                inputs, {"primary": final.prepare_documents(data.text)}),
            "validation_ids_sha256": final._fingerprint(metadata.index[metadata["season"].ne("2122")].tolist()),
        }
    articles = final.aligned_articles(metadata, match_path, preview_path)
    sides = final.side_documents(articles, data.text, cache / "phase06b_attribution.json.gz")
    frozen["data_fingerprints"]["development_attribution_sha256"] = sides.fingerprint


def test_synthetic_runner_fits_once_per_market_and_isolates_holdouts(tmp_path, monkeypatch):
    match_path, preview_path = _synthetic_exports(tmp_path)
    cache = tmp_path / "cache"
    frozen = final.load_freeze()
    _synthetic_fingerprints(frozen, match_path, preview_path, cache)
    monkeypatch.setattr(final, "CACHE_ROOT", cache)
    monkeypatch.setattr(final, "load_freeze", lambda: frozen)
    audit_path, approval_path = tmp_path / "audit.csv", tmp_path / "approval.json"
    pd.DataFrame([_audit_entry("2526-0", flags=("no", "no", "yes"), evidence="synthetic reference")]).to_csv(
        audit_path, index=False,
    )
    policy = {"status": "approved", "approved_by": "synthetic reviewer", "approved_at": "2025-01-01T10:00:00Z",
              "freeze_sha256": final.FREEZE_SHA256, "audit_sha256": final._file_hash(audit_path),
              "exclude_if_any": {market: [final.CONTAMINATION] for market in final.MARKETS}}
    events, expected_training = [], 36
    original_scaler = final.StandardScaler.fit
    original_vectorizer = TfidfVectorizer.fit_transform
    original_model = final.MarketResidual.fit
    original_outcomes = final._read_outcomes

    def track_outcomes(path, seasons):
        events.append(seasons)
        return original_outcomes(path, seasons)

    def check_scaler(self, values, *args, **kwargs):
        assert len(values) == expected_training
        return original_scaler(self, values, *args, **kwargs)

    def check_vocabulary(self, documents, *args, **kwargs):
        assert len(documents) == 2 * expected_training
        assert all(not article.startswith(final.HOLDOUTS) for article in documents.index)
        matrix = original_vectorizer(self, documents, *args, **kwargs)
        assert "developmentword" in self.vocabulary_
        assert "holdoutword" not in self.vocabulary_
        return matrix

    def check_fit(self, market, y, features, **kwargs):
        events.append(tuple(penalty for _, penalty in kwargs["block_penalties"]))
        assert len(y) == expected_training
        return original_model(self, market, y, features, **kwargs)

    monkeypatch.setattr(final, "_read_outcomes", track_outcomes)
    monkeypatch.setattr(final.StandardScaler, "fit", check_scaler)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", check_vocabulary)
    monkeypatch.setattr(final.MarketResidual, "fit", check_fit)

    def run(name):
        policy["input_sha256"] = final._input_hashes(match_path, preview_path)
        approval_path.write_text(json.dumps(policy), encoding="utf-8")
        result = final.run_evaluation(
            execute_holdouts=True, matches_path=match_path, previews_path=preview_path,
            audit_path=audit_path, approval_path=approval_path, output=tmp_path / name,
        )
        assert result["holdout_outcomes_evaluated"] is True
        assert events == [final.DEVELOPMENT, (1.0,), (10.0,), (10.0, 1.0), (0.1,), (1.0,), (1.0, 1.0), final.HOLDOUTS]
        events.clear()
        return result["evaluation"]

    first = run("first")
    assert first["early"]["seasons"]["2526"]["integrity"]["screened_matches"] == 8
    assert first["closing"]["seasons"]["2526"]["integrity"]["screened_matches"] == 9
    old = pd.read_csv(match_path, dtype=str)
    old.loc[old["season"].eq("2627"), "result"] = "A"
    old.loc[old["article_id"].eq("2627-0"), "result"] = ""
    old.loc[old["season"].eq("2627"), "early_odds_home"] = "9.0"
    old.to_csv(match_path, index=False)
    with pytest.raises(ValueError, match="exact input files"):
        final.evaluation_plan(frozen, match_path, preview_path, audit_path, approval_path)
    second = run("second")
    assert second["early"]["seasons"]["2627"]["missing_or_invalid_outcomes"] == 1
    for market in final.MARKETS:
        assert second[market]["fit"]["training_ids_sha256"] == first[market]["fit"]["training_ids_sha256"]
        base = f"{market}/2526/original/metrics.csv"
        assert (tmp_path / "first" / base).read_bytes() == (tmp_path / "second" / base).read_bytes()
        holdout = f"{market}/2627/original/metrics.csv"
        assert (tmp_path / "first" / holdout).read_bytes() != (tmp_path / "second" / holdout).read_bytes()
