"""Fixed article representations and training-fold word TF-IDF."""

import re
import unicodedata

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .dataset import TEXT_COLUMNS

PRIMARY_TEXT_COLUMNS = ("text_preview", "text_team_news", "text_head_to_head")
TEXT_REPRESENTATIONS = {"primary": PRIMARY_TEXT_COLUMNS, "full_article": TEXT_COLUMNS}
UNICODE_FORM = "NFC"
TFIDF_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w+(?:['’]\w+)*\b",
    "lowercase": True,
    "sublinear_tf": True,
    "norm": "l2",
    "min_df": 5,
    "max_df": 0.95,
    "max_features": 10_000,
    "use_idf": True,
    "smooth_idf": True,
}
SCORELINE_PATTERN = re.compile(r"(?<!\w)\d{1,2}\s*[-‐‑‒–—−:]\s*\d{1,2}(?!\w)")
PREDICTION_PATTERN = re.compile(
    r"\b(?:we (?:say|predict|expect|tip)|our (?:prediction|pick)|"
    r"predicted (?:score(?:line)?|result)|(?:match|score(?:line)?) prediction)\b",
    re.IGNORECASE,
)


def prepare_documents(sections: pd.DataFrame, representation: str = "primary") -> pd.Series:
    """Join normalized, allowlisted sections in fixed order and preserve row IDs."""
    if representation not in TEXT_REPRESENTATIONS:
        raise ValueError(f"Unknown text representation {representation!r}")
    documents = []
    for values in sections.reindex(columns=TEXT_REPRESENTATIONS[representation]).itertuples(index=False, name=None):
        joined = " ".join(value for value in values if isinstance(value, str))
        documents.append(" ".join(unicodedata.normalize(UNICODE_FORM, joined).split()))
    return pd.Series(documents, index=sections.index, dtype="string", name=representation)


def text_diagnostics(documents: pd.Series) -> dict[str, int]:
    """Count heuristic mentions and affected documents without retaining excerpts."""
    scorelines = documents.str.count(SCORELINE_PATTERN)
    predictions = documents.str.count(PREDICTION_PATTERN)
    return {
        "documents": len(documents),
        "empty_documents": int(documents.eq("").sum()),
        "scoreline_mentions": int(scorelines.sum()),
        "documents_with_scorelines": int(scorelines.gt(0).sum()),
        "prediction_phrase_mentions": int(predictions.sum()),
        "documents_with_prediction_phrases": int(predictions.gt(0).sum()),
    }


def fit_tfidf(training_documents: pd.Series) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit only on the supplied training documents and return their CSR matrix."""
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    return vectorizer, csr_matrix(vectorizer.fit_transform(training_documents))
