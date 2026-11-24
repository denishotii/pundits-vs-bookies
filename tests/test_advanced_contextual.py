import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from pvb.advanced import contextual_experiments as contextual
from pvb.advanced import embeddings, nlp_experiments
from pvb.advanced.market import market_probabilities
from pvb.advanced.models import MarketResidual
from pvb.advanced.preprocessing import prepare_documents
from pvb.advanced.validation import chronological_folds

from test_advanced_nlp_experiments import nlp_dataset


class MockTokenizer:
    def encode(self, section, *, add_special_tokens, truncation, verbose):
        assert not add_special_tokens and not truncation
        return [int(word) if word.isdigit() else sum(map(ord, word)) for word in section.split()]

    def num_special_tokens_to_add(self, *, pair):
        assert not pair
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [-1, *ids, -2]


class MockEncoder:
    tokenizer = MockTokenizer()
    max_input_tokens = 5
    dimension = 2
    device = "cpu"
    runtime = {"device": device}

    def __init__(self, revision="mock-revision"):
        self.config = {"model": "mock", "revision": revision, "max_tokens": self.max_input_tokens}
        self.batches = []

    def encode_chunks(self, input_ids):
        self.batches.extend(input_ids)
        return np.array([[sum(ids[1:-1]), 10 * len(ids[1:-1])] for ids in input_ids], dtype=float)


def test_block_penalties_satisfy_objective_and_match_single_penalty():
    rng = np.random.default_rng(73)
    market = rng.dirichlet([3, 2, 3], size=90)
    y = np.tile(np.arange(3), len(market) // 3)
    features = rng.normal(size=(len(y), 5))
    features[np.arange(len(y)), y + 2] += 1.8
    matrix = sparse.csr_matrix(features)
    numerical_l2, text_l2 = 1.0, 0.001
    model = MarketResidual(numerical_l2, text_l2=text_l2, standardize=False).fit(
        market, y, matrix, numerical_width=2,
    )
    assert model.convergence_.converged
    np.testing.assert_allclose(model.coef_.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(model.intercept_.sum(), 0, atol=1e-12)
    errors = model.predict_proba(market, matrix) - np.eye(3)[y]
    penalties = np.array([numerical_l2] * 2 + [text_l2] * 3)[:, None]
    np.testing.assert_allclose(features.T @ errors / len(y) + penalties * model.coef_, 0, atol=1e-6)
    np.testing.assert_allclose(errors.mean(axis=0), 0, atol=1e-6)
    strong_text = MarketResidual(numerical_l2, text_l2=1.0, standardize=False).fit(
        market, y, matrix, numerical_width=2,
    )
    assert np.linalg.norm(strong_text.coef_[2:]) < np.linalg.norm(model.coef_[2:])
    old = MarketResidual(1.0, standardize=False).fit(market, y, matrix)
    np.testing.assert_allclose(old.predict_proba(market, matrix), strong_text.predict_proba(market, matrix))
    for width in (None, 0, True, 5, 2.5):
        with pytest.raises(ValueError, match="two nonempty feature blocks"):
            model.fit(market, y, matrix, numerical_width=width)
    with pytest.raises(ValueError, match="requires text_l2"):
        old.fit(market, y, matrix, numerical_width=2)
    with pytest.raises(ValueError, match="strictly positive"):
        MarketResidual(1.0, text_l2=0)


def test_token_chunks_preserve_all_ids_and_section_boundaries():
    sections = ["1 2 3 4 5 6 7", "", "8 9"]
    assert embeddings.tokenize_chunks(sections, MockTokenizer(), 5) == [
        ([-1, 1, 2, 3, -2], 3),
        ([-1, 4, 5, 6, -2], 3),
        ([-1, 7, -2], 1),
        ([-1, 8, 9, -2], 2),
    ]
    with pytest.raises(ValueError, match="at least one content token"):
        embeddings.tokenize_chunks(sections, MockTokenizer(), 2)


def test_embedding_pooling_and_cache_invalidation(tmp_path, monkeypatch):
    chunks = np.array([[8.0, 0.0], [0.0, 3.0]])
    pooled = embeddings.pool_chunks(chunks, [3, 1])
    np.testing.assert_allclose(pooled, [3 / np.sqrt(10), 1 / np.sqrt(10)])
    np.testing.assert_array_equal(embeddings.pool_chunks(chunks[::-1], [1, 3]), pooled)
    np.testing.assert_array_equal(embeddings.pool_chunks(np.empty((0, 2)), []), [0, 0])

    sections = pd.DataFrame({
        "text_preview": ["1 2 3 4", None], "text_team_news": ["  5\n6 ", " \t "],
        "text_verdict": ["other", "unused"],
    }, index=["article-a", "article-b"])
    path = tmp_path / "private" / "cache.sqlite3"
    encoder = MockEncoder()
    first = embeddings.embed_articles(sections, encoder, path)
    assert first.statistics["cache_misses"] == 2
    assert first.statistics["total_chunks"] == 3
    assert first.statistics["coverage"] == 0.5
    assert first.statistics["empty_articles"] == 1
    assert len(encoder.batches) == 3
    np.testing.assert_array_equal(first.vectors.loc["article-b"], [0, 0])
    np.testing.assert_allclose(first.vectors.loc["article-a"], embeddings.pool_chunks(
        np.array([[6, 30], [4, 10], [11, 20]]), [3, 1, 2],
    ))

    reordered = sections.iloc[::-1].assign(text_verdict="edited")
    reordered.loc["article-a", "text_team_news"] = "5 6"
    repeat = embeddings.embed_articles(reordered, encoder, path)
    assert repeat.statistics["cache_hits"] == 2 and len(encoder.batches) == 3
    pd.testing.assert_frame_equal(repeat.vectors, first.vectors.iloc[::-1])

    edited = sections.assign(text_preview=["1 2 3 99", None])
    assert embeddings.embed_articles(edited, encoder, path).statistics["cache_misses"] == 1
    renamed = sections.rename(index={"article-a": "article-c"})
    assert embeddings.embed_articles(renamed, encoder, path).statistics["cache_misses"] == 1
    assert embeddings.embed_articles(sections, MockEncoder("changed-revision"), path).statistics["cache_misses"] == 2
    monkeypatch.setattr(embeddings, "REPRESENTATION", {**embeddings.REPRESENTATION, "schema": 2})
    assert embeddings.embed_articles(sections, encoder, path).statistics["cache_misses"] == 2


def test_contextual_runner_aligns_all_blocks_and_isolates_holdouts(tmp_path, monkeypatch, nlp_dataset):
    data = nlp_dataset
    monkeypatch.setattr(contextual, "load_football", lambda *args, **kwargs: data)
    unavailable = data.metadata["article_id"].eq("2122-0")
    data.metadata.loc[unavailable, "outcome_available_at"] += pd.Timedelta(days=1500)
    development = data.metadata["season"].isin(["2122", "2223", "2324", "2425"]).to_numpy()
    metadata = data.metadata.iloc[development].set_index("article_id")
    valid_only_word = data.metadata["season"].eq("2223")
    data.text.loc[valid_only_word, "text_preview"] += " newword"
    sections = data.text.iloc[development].set_axis(metadata.index)
    documents = prepare_documents(sections)
    splits, _ = chronological_folds(metadata.reset_index())
    numerical = data.numerical.iloc[development].set_axis(metadata.index)
    expected_numeric = np.column_stack((
        np.log(market_probabilities(numerical, "early")), numerical[list(contextual.PICK_COLUMNS)],
    ))
    encoder = MockEncoder()
    original_embed, original_tfidf = contextual.embed_articles, nlp_experiments.fit_tfidf
    original_fit, original_evaluate = contextual._fit_candidate, contextual.evaluate_predictions
    transformed, text_fits, used_matrices, evaluation_ids = {}, [], {}, []

    def checked_embed(articles, passed_encoder, cache_path):
        used = np.unique(np.concatenate([rows for fold in splits for rows in fold]))
        pd.testing.assert_frame_equal(articles, sections.iloc[used][list(embeddings.PRIMARY_TEXT_COLUMNS)])
        result = original_embed(articles, passed_encoder, cache_path)
        transformed["embeddings"] = result.vectors
        return result

    def checked_tfidf(training_documents):
        train, valid = splits[len(text_fits)]
        pd.testing.assert_series_equal(training_documents, documents.iloc[train])
        vectorizer, matrix = original_tfidf(training_documents)
        assert {"holdouttoken", "idtoken", "scoretoken", "nbtoken", "predict"}.isdisjoint(vectorizer.vocabulary_)
        assert ("newword" in vectorizer.vocabulary_) == bool(text_fits)
        text_fits.append((matrix, vectorizer.transform(documents.iloc[valid])))
        return vectorizer, matrix

    def checked_fit(name, fold, probabilities, target, matrices, width, numerical_l2, text_l2):
        assert target.index.equals(metadata.index)
        assert target.isin([0, 1, 2]).all()
        i = int(fold.scope[-1]) - 1
        train, valid = splits[i]
        np.testing.assert_array_equal(fold.train, train)
        np.testing.assert_array_equal(fold.valid, valid)
        key = name, fold.scope
        if key in used_matrices:
            assert used_matrices[key] is matrices
        else:
            used_matrices[key] = matrices
            scaler = StandardScaler().fit(expected_numeric[train])
            for j, (rows, matrix) in enumerate(zip((train, valid), matrices, strict=True)):
                expected = scaler.transform(expected_numeric[rows])[:, :width]
                if sparse.issparse(matrix):
                    assert isinstance(matrix, sparse.csr_matrix)
                    np.testing.assert_allclose(matrix[:, :width].toarray(), expected)
                    multiplier = contextual.TEXT_WEIGHT if name.startswith("phase04_") else 1.0
                    np.testing.assert_allclose((matrix[:, width:] - multiplier * text_fits[i][j]).data, 0)
                else:
                    np.testing.assert_allclose(matrix[:, :width], expected)
                    if name in {"M6", "M7"}:
                        np.testing.assert_allclose(matrix[:, width:], transformed["embeddings"].loc[metadata.index[rows]])
        return original_fit(name, fold, probabilities, target, matrices, width, numerical_l2, text_l2)

    def checked_evaluate(target, forecasts):
        assert set(forecasts) == {"M0", "M2", "M3", "M4", "M5", "M6", "M7"}
        assert all(frame.index.equals(target.index) for frame in forecasts.values())
        evaluation_ids.append(target.index)
        return original_evaluate(target, forecasts)

    monkeypatch.setattr(contextual, "embed_articles", checked_embed)
    monkeypatch.setattr(nlp_experiments, "fit_tfidf", checked_tfidf)
    monkeypatch.setattr(contextual, "_fit_candidate", checked_fit)
    monkeypatch.setattr(contextual, "evaluate_predictions", checked_evaluate)
    scores = contextual.run_experiment("early", tmp_path, encoder=encoder, cache_root=tmp_path / "cache")
    assert len(text_fits) == len(splits)
    expected_scopes = [metadata.index[valid] for _, valid in splits]
    expected_scopes.append(metadata.index[np.concatenate([valid for _, valid in splits])])
    for seen, expected in zip(evaluation_ids, expected_scopes, strict=True):
        pd.testing.assert_index_equal(seen, expected)

    output = tmp_path / "early"
    manifest = json.loads((output / "metadata.json").read_text())
    assert set(manifest["development_seasons"]) == {"2122", "2223", "2324", "2425"}
    assert manifest["embedding"]["articles"] == len(metadata) - 1
    assert not manifest["independent_evaluation"]
    search = pd.read_csv(output / "regularization_search.csv")
    paired = pd.read_csv(output / "paired_log_loss.csv")
    for name, (baseline, _) in contextual.TEXT_MODELS.items():
        rows = search.loc[search.model.eq(name)]
        assert rows["numerical_l2"].unique().tolist() == [manifest["selected_penalties"][baseline]["numerical_l2"]]
        assert rows["text_l2"].unique().tolist() == list(contextual.TEXT_L2)
    for row in paired.itertuples(index=False):
        losses = scores.loc[scores.scope.eq(row.scope)].set_index("model")["log_loss"]
        assert row.mean_log_loss_difference == pytest.approx(losses[row.model] - losses[row.reference])
    assert set(paired.loc[paired.text_conditional_on_picks, "model"]) == {"M5", "M7"}
    assert not (output / "oof_predictions.csv").exists()

    data.text = data.text.iloc[::-1]
    with pytest.raises(ValueError, match="must align with the metadata rows"):
        contextual.run_experiment("early", tmp_path, encoder=encoder)
