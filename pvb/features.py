"""Text features: turn each preview's text into a few numbers.

Status: the pick is already parsed in parse.py, and word counting works.
The word lists below are starter lists to show the approach. The text
analysis role replaces them with proper dictionaries (Harvard IV / Loughran-
McDonald from the course) and adds the features marked TODO.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config
from .parse import PREVIEWS_CSV

FEATURES_CSV = config.PROCESSED / "features.csv"

# Starter lists: extend or replace with the course dictionaries.
ABSENCE_TERMS = ["injured", "injury", "ruled out", "suspended", "suspension", "doubt", "doubtful", "sidelined", "unavailable", "fitness"]
UNCERTAINTY_TERMS = ["could", "might", "perhaps", "tight", "unpredictable", "hard to call", "uncertain"]


def count_terms(text: str, terms: list[str]) -> int:
    """How often any of `terms` appears as whole words (case-insensitive)."""
    if not text:
        return 0
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
    return len(re.findall(pattern, text, flags=re.I))


def per_1000_words(count: int, words: int) -> float:
    return 1000 * count / words if words else 0.0


def basic_features(previews: pd.DataFrame) -> pd.DataFrame:
    """Features that need no training: absence words and uncertainty words."""
    out = previews[["article_id", "sport", "pick_result", "pick_home_goals", "pick_away_goals", "pick_winner", "pick_sets"]].copy()
    news = previews["text_team_news"].fillna("")
    full = previews[[c for c in previews.columns if c.startswith("text_")]].fillna("").agg("\n".join, axis=1)
    out["absence_terms"] = news.map(lambda t: count_terms(t, ABSENCE_TERMS))
    out["uncertainty_per_1000"] = [
        per_1000_words(count_terms(t, UNCERTAINTY_TERMS), n) for t, n in zip(full, previews["word_count"])
    ]
    return out


def tone_by_side(previews: pd.DataFrame) -> pd.DataFrame:
    """TODO (text analysis): net tone of the sentences that name each team or player.

    Plan: split into sentences, keep those naming side_a or side_b, score them
    with a sentiment dictionary, return tone_a, tone_b and tone_diff.
    """
    raise NotImplementedError("tone_by_side: see docs/plan.md, text analysis role")


def naive_bayes_probabilities(previews: pd.DataFrame) -> pd.DataFrame:
    """TODO (text analysis): P(home win / draw / away win) learned from the words.

    Plan: bag of words (sklearn CountVectorizer) + MultinomialNB, trained on
    2021/22-2024/25 and applied to 2025/26 and 2026/27 only, so the test
    seasons are never seen during training.
    """
    raise NotImplementedError("naive_bayes_probabilities: see docs/plan.md, text analysis role")


def build() -> pd.DataFrame:
    previews = pd.read_csv(PREVIEWS_CSV)
    features = basic_features(previews)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)
    return features
