import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from pvb.advanced import experiments, nlp_experiments as nlp
from pvb.advanced.dataset import FootballDataset, ODDS_COLUMNS
from pvb.advanced.market import market_probabilities
from pvb.advanced.models import L2_GRID, MarketResidual
from pvb.advanced.preprocessing import prepare_documents
from pvb.advanced.validation import chronological_folds


def test_preprocessed_sparse_residual_matches_default_dense_scaling_and_zero_correction():
    market = np.array([[0.65, 0.20, 0.15], [0.2, 0.4, 0.4], [0.15, 0.2, 0.65]] * 3)
    features = np.repeat([[13.0, 1.0], [16.0, 0.0], [25.0, 0.0]], 3, axis=0)
    outcomes = np.array([0, 1, 2, 1, 1, 0, 2, 0, 2])
    default = MarketResidual(0.01).fit(market, outcomes, features)
    scaled = sparse.csr_matrix(default.scaler_.transform(features))
    preprocessed = MarketResidual(0.01, standardize=False).fit(market, outcomes, scaled)
    assert preprocessed.scaler_ is None
    assert default.convergence_.converged and preprocessed.convergence_.converged
    np.testing.assert_allclose(preprocessed.predict_proba(market, scaled), default.predict_proba(market, features))
    with pytest.raises(ValueError, match="same column count"):
        preprocessed.predict_proba(market, scaled[:, :1])

    uniform = np.full_like(market, 1 / 3)
    balanced = MarketResidual(0.01, standardize=False).fit(uniform, np.tile([0, 1, 2], 3), scaled)
    assert balanced.convergence_.converged
    np.testing.assert_allclose(balanced.coef_, 0)
    np.testing.assert_allclose(balanced.predict_proba(market, scaled), market)


@pytest.fixture
def nlp_dataset():
    seasons = ["2122", "2223", "2324", "2425", "2526", "2627"]
    metadata_rows, text_rows, numeric_rows, labels = [], [], [], []
    for i, (season, size) in enumerate(zip(seasons, [21, 18, 24, 27, 2, 2], strict=True)):
        for j in range(size):
            kickoff = pd.Timestamp(f"{2021 + i}-08-{j + 1:02}T12:00:00Z")
            if i == 0 and j == size - 1:
                kickoff = pd.Timestamp("2022-08-01T08:00:00Z")  # result at the next season's first cutoff
            metadata_rows.append({
                "article_id": f"{season}-{j}", "season": season, "kickoff": kickoff,
                "published": kickoff - pd.Timedelta(days=1), "prediction_at": kickoff - pd.Timedelta(hours=1),
                "outcome_available_at": kickoff + pd.Timedelta(hours=3),
            })
            text_rows.append({
                "text_preview": ["press high", "counter attack", "set piece"][j % 3] if i < 4 else "holdouttoken",
                "text_team_news": None, "text_head_to_head": "", "text_verdict": "we predict 2-1" if j % 2 == 0 else "",
                "article_id": "idtoken", "url": "urltoken", "result": "resulttoken",
                "home_goals": "scoretoken", "nb_home_vs_draw": "nbtoken",
            })
            numeric_rows.append({
                **dict(zip(ODDS_COLUMNS["early"], (2 + j / 24, 3.0, 3.1 + i / 4), strict=True)),
                "pick_home": int(j % 3 == 0), "pick_draw": int(j % 3 == 1), "pick_goal_diff": 1 - j % 3,
                "closing_odds_home": 0, "home_goals": 99, "nb_home_vs_draw": -99,
            } if i < 4 else {})
            labels.append((j + j // 4) % 3 if i < 4 else 99)
    order = np.random.default_rng(1).permutation(len(labels))
    return FootballDataset(
        pd.Series(labels).iloc[order], pd.DataFrame(metadata_rows).iloc[order],
        pd.DataFrame(text_rows).iloc[order], pd.DataFrame(numeric_rows).iloc[order], {},
    )


def test_runner_keeps_blocks_and_oof_ids_aligned_and_reuses_phase03(tmp_path, monkeypatch, nlp_dataset):
    data = nlp_dataset
    monkeypatch.setattr(nlp, "load_football", lambda *args, **kwargs: data)
    monkeypatch.setattr(experiments, "load_football", lambda *args, **kwargs: data)
    development = data.metadata["season"].isin(["2122", "2223", "2324", "2425"]).to_numpy()
    metadata = data.metadata.iloc[development].set_index("article_id")
    folds, _ = chronological_folds(metadata.reset_index())
    numerical = data.numerical.iloc[development].set_axis(metadata.index)
    numerical_values = np.column_stack((
        np.log(market_probabilities(numerical, "early")), numerical.loc[:, ["pick_home", "pick_draw", "pick_goal_diff"]],
    ))
    sections = data.text.iloc[development].set_axis(metadata.index)
    fit_tfidf = nlp.fit_tfidf
    tfidf_fits = {}

    def observed_tfidf(training_documents):
        scope = f"fold_{len(tfidf_fits) % len(folds) + 1}"
        train, valid = folds[len(tfidf_fits) % len(folds)]
        expected_representation = "primary" if len(tfidf_fits) < len(folds) else "full_article"
        assert training_documents.name == expected_representation
        documents = prepare_documents(sections, expected_representation)
        pd.testing.assert_series_equal(training_documents, documents.iloc[train])
        vectorizer, train_matrix = fit_tfidf(training_documents)
        assert {"holdouttoken", "idtoken", "urltoken", "resulttoken", "scoretoken", "nbtoken"}.isdisjoint(
            vectorizer.vocabulary_,
        )
        assert ("predict" in vectorizer.vocabulary_) == (expected_representation == "full_article")
        tfidf_fits[scope, expected_representation] = (train_matrix, vectorizer.transform(documents.iloc[valid]))
        return vectorizer, train_matrix

    fit_candidate = nlp._fit_candidate
    used_blocks = {}

    def observed_fit(name, l2, scope, probabilities, target, train, valid, blocks=None):
        assert target.index.equals(metadata.index)
        assert np.isin(target.to_numpy(), [0, 1, 2]).all()
        expected_train, expected_valid = folds[int(scope[-1]) - 1]
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(valid, expected_valid)
        if blocks is not None and (name, scope) not in used_blocks:
            scaler = StandardScaler().fit(numerical_values[train])
            width = 3 if name == "M4" else 6
            for partition, rows in enumerate((train, valid)):
                matrix = blocks[partition]
                expected_numeric = scaler.transform(numerical_values[rows])[:, :width]
                if name == "M3":
                    np.testing.assert_allclose(matrix, expected_numeric)
                    continue
                assert isinstance(matrix, sparse.csr_matrix)
                representation = "full_article" if name == nlp.ABLATION else "primary"
                expected_text = tfidf_fits[scope, representation][partition] * nlp.TEXT_WEIGHT
                difference = matrix - sparse.hstack((sparse.csr_matrix(expected_numeric), expected_text), format="csr")
                np.testing.assert_allclose(difference.data, 0, atol=1e-12)
            used_blocks[name, scope] = blocks
        elif blocks is not None:
            assert blocks is used_blocks[name, scope]
        return fit_candidate(name, l2, scope, probabilities, target, train, valid, blocks)

    evaluate = nlp.evaluate_predictions
    seen_ids = []

    def observed_evaluation(target, predictions):
        assert all(frame.index.equals(target.index) for frame in predictions.values())
        seen_ids.append(target.index.tolist())
        return evaluate(target, predictions)

    monkeypatch.setattr(nlp, "fit_tfidf", observed_tfidf)
    monkeypatch.setattr(nlp, "_fit_candidate", observed_fit)
    monkeypatch.setattr(nlp, "evaluate_predictions", observed_evaluation)
    scores = nlp.run_experiment("early", tmp_path / "nlp")
    phase03 = experiments.run_experiment("early", tmp_path / "phase03")

    assert len(tfidf_fits) == 2 * len(folds)
    assert seen_ids == [metadata.index[valid].tolist() for _, valid in folds] + [
        metadata.index[np.concatenate([valid for _, valid in folds])].tolist(),
    ]
    pd.testing.assert_frame_equal(
        scores.loc[scores.model.isin(["M0", "M1", "M2"]), phase03.columns].reset_index(drop=True),
        phase03,
    )
    output = tmp_path / "nlp" / "early"
    manifest = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert set(manifest["development_seasons"]) == {"2122", "2223", "2324", "2425"}
    assert manifest["preliminary"] and not manifest["independent_evaluation"]
    assert len(manifest["effective_development_input_sha256"]) == 64
    fold_info = pd.read_csv(output / "folds.csv")
    assert fold_info["purged_train_rows"].tolist() == [1, 0, 0]
    paired = pd.read_csv(output / "paired_log_loss.csv")
    assert set(zip(paired.model, paired.reference)) == {
        (model, "M0") for model in ("M1", "M2", "M3", "M4", "M5", nlp.ABLATION)
    } | {("M3", "M2"), ("M4", "M2"), ("M5", "M3"), (nlp.ABLATION, "M5"), (nlp.ABLATION, "M3")}
    for row in paired.itertuples(index=False):
        losses = scores.loc[scores.scope.eq(row.scope)].set_index("model")["log_loss"]
        assert row.mean_log_loss_difference == pytest.approx(losses[row.model] - losses[row.reference])
        assert row.primary_comparison == ((row.model, row.reference) == ("M5", "M3"))
    search = pd.read_csv(output / "regularization_search.csv")
    assert len(search.query("model == 'M5' and scope != 'pooled'")) == len(L2_GRID) * len(folds)
    ablation = search.loc[search["model"].eq(nlp.ABLATION)]
    assert ablation["l2"].unique().tolist() == [manifest["selected_l2"]["M5"]]
    assert ablation["included_in_reported_oof"].all()
    assert not (output / "oof_predictions.csv").exists()


def test_runner_refuses_to_overwrite_phase03_and_checks_input_alignment(tmp_path, monkeypatch, nlp_dataset):
    nlp_dataset.text = nlp_dataset.text.iloc[::-1]
    monkeypatch.setattr(nlp, "load_football", lambda *args, **kwargs: nlp_dataset)
    with pytest.raises(ValueError, match="separate from the Phase 03 results"):
        nlp.run_experiment("early", nlp.RESULTS_ROOT)
    with pytest.raises(ValueError, match="must align with the metadata rows"):
        nlp.run_experiment("early", tmp_path)


def test_selection_requires_every_fold_to_converge_and_favors_stronger_l2_on_a_tie():
    def fits(l2, losses, statuses):
        return [(None, loss, {
            "model": "M5", "l2": l2, "matches": 1, "converged": status,
            "fit_seconds": 0.0, "predict_seconds": 0.0,
        }) for loss, status in zip(losses, statuses, strict=True)]

    selected, _, diagnostics = nlp._select_candidates({
        ("M5", 0.0001): fits(0.0001, [0.8, 0.8], [True, False]),
        ("M5", 0.001): fits(0.001, [1.0, 1.0], [True, True]),
        ("M5", 0.01): fits(0.01, [1.0, 1.0], [True, True]),
    })
    assert selected == {"M5": 0.01}
    assert not diagnostics[2]["converged"]
