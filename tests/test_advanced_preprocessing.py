import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from pvb.advanced.preprocessing import fit_tfidf, prepare_documents, text_diagnostics


def test_document_allowlists_normalize_unicode_and_handle_missing_sections():
    sections = pd.DataFrame({
        "text_preview": ["\tCafe\u0301 will not win\u00a02–1\n", None],
        "text_team_news": ["He isn’t fit.", pd.NA],
        "title": ["Title", ""], "text_intro": ["Intro", np.nan],
        "text_verdict": ["We predict 2-1", "  "], "text_other": ["Other", None],
        "article_id": ["id1", "id2"], "url": ["url1", "url2"],
        "result": ["H", "D"], "home_goals": [3, 0], "nb_home_vs_draw": [12, 30],
    }, index=[9, 5])
    primary = prepare_documents(sections)
    full = prepare_documents(sections, "full_article")

    assert primary.index.tolist() == [9, 5]
    assert primary.tolist() == ["Café will not win 2–1 He isn’t fit.", ""]
    assert full.tolist() == ["Title Intro Café will not win 2–1 He isn’t fit. We predict 2-1 Other", ""]
    assert text_diagnostics(primary) == {
        "documents": 2, "empty_documents": 1, "scoreline_mentions": 1, "documents_with_scorelines": 1,
        "prediction_phrase_mentions": 0, "documents_with_prediction_phrases": 0,
    }
    assert text_diagnostics(full)["scoreline_mentions"] == 2
    assert text_diagnostics(full)["documents_with_prediction_phrases"] == 1


def test_word_tfidf_fit_uses_training_vocabulary_and_idf_and_leaves_empty_rows_sparse():
    train = pd.Series(["Press press not 2 isn’t"] * 5 + ["counter attack"] * 5)
    valid = pd.Series(["validationonly press not 2", "validationonly", "", "validationonly press"])
    vectorizer, train_matrix = fit_tfidf(train)
    vocabulary = vectorizer.vocabulary_.copy()
    train_idf = vectorizer.idf_.copy()
    valid_matrix = vectorizer.transform(valid)

    assert isinstance(train_matrix, csr_matrix) and issparse(valid_matrix) and valid_matrix.format == "csr"
    assert {"press", "press not", "not", "2", "isn’t"} <= vocabulary.keys()
    assert "validationonly" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_allclose(train_idf, np.log(11 / 6) + 1)
    np.testing.assert_array_equal(vectorizer.idf_, train_idf)
    np.testing.assert_allclose(np.asarray(train_matrix.power(2).sum(axis=1)).ravel(), 1)
    np.testing.assert_allclose(np.asarray(valid_matrix.power(2).sum(axis=1)).ravel(), [1, 0, 0, 1])
