"""As-of journalist histories for a single market and chronological fold."""

from collections import defaultdict
import unicodedata

import numpy as np
import pandas as pd

from .dataset import _read_source
from .diagnostic_data import DEVELOPMENT
from .market import CLASS_NAMES, market_probabilities, validate_targets

PRIOR_STRENGTH = 20.0
RATE_FEATURES = (
    "author_accuracy", *(f"author_pick_frequency_{name}" for name in CLASS_NAMES),
    "author_accuracy_agree", "author_accuracy_disagree", "author_excess_accuracy",
)
MISSING_FLAGS = (
    "author_missing_byline", "author_missing_history", "author_missing_agree_history", "author_missing_disagree_history",
)
AUTHOR_FEATURES = ("author_previous_predictions", *RATE_FEATURES, *MISSING_FLAGS)
NEUTRAL_RATES = np.array([1 / 3] * (len(RATE_FEATURES) - 1) + [0.0])
GENERIC_BYLINES = frozenset({
    "sports mole", "sports mole staff", "sports mole team", "editorial team", "staff writer", "unknown author",
})
PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-"})


def normalize_author(value):
    """Match only simple byline variants, leaving ambiguous or generic bylines unidentified."""
    if not isinstance(value, str):
        return None
    name = " ".join(unicodedata.normalize("NFKC", value).casefold().translate(PUNCTUATION).split())
    if (len(name.split()) < 2 or name in GENERIC_BYLINES or " and " in name
            or any(not (char.isalpha() or char in " '.-") for char in name)):
        return None
    return name


def load_author_keys(previews_path, index):
    """Read only byline metadata and align it to eligible article IDs."""
    source = _read_source(previews_path, ("article_id", "author")).set_index("article_id")
    return source["author"].reindex(index).map(normalize_author)


def expert_pick_codes(numerical):
    flags = numerical[["pick_home", "pick_draw"]].to_numpy()
    if not np.isin(flags, [0, 1]).all() or (flags.sum(axis=1) > 1).any():
        raise ValueError("Expected valid home and draw expert-pick flags")
    return np.where(flags[:, 0] == 1, 0, np.where(flags[:, 1] == 1, 1, 2))


def author_features(market, fixtures, author_keys, numerical, target):
    """Replay eligible outcomes within a fold and return numeric features in fixture order.

    Each rate shrinks toward 20 equivalent prior observations from the already
    released market cohort, excluding the current author's own records. The
    population can include missing bylines. The neutral fallback is 1/3 for
    accuracy and pick frequencies and zero for excess accuracy. Tied market
    favourites enter overall rates only. No fitted parameters cross folds.
    """
    index = fixtures.index
    if not index.is_unique or index.hasnans or any(
        not frame.index.equals(index) for frame in (author_keys, numerical, target)
    ):
        raise ValueError("Author history requires identical, unique fixture IDs")
    if not fixtures["season"].isin(DEVELOPMENT).all():
        raise ValueError("Author history accepts development seasons only")
    prediction, kickoff, release = (fixtures[name] for name in ("prediction_at", "kickoff", "outcome_available_at"))
    if (any(not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any()
            for times in (prediction, kickoff, release))
            or prediction.gt(kickoff).any() or release.le(kickoff).any()):
        raise ValueError("Author history requires prediction <= kickoff < release with known timestamps")

    book = market_probabilities(numerical, market)
    picked = expert_pick_codes(numerical)
    correct = validate_targets(target.to_numpy(), len(index)) == picked
    selected_probability = book[np.arange(len(index)), picked]
    sole_favourite = (book == book.max(axis=1, keepdims=True)).sum(axis=1) == 1
    agrees = sole_favourite & (picked == book.argmax(axis=1))
    disagrees = sole_favourite & ~agrees
    values = np.column_stack((
        correct, np.eye(len(CLASS_NAMES))[picked], correct & agrees, correct & disagrees,
        correct - selected_probability,
    ))
    eligible = np.ones_like(values)
    eligible[:, 4:6] = np.column_stack((agrees, disagrees))

    def empty():
        return np.zeros((2, len(RATE_FEATURES)))

    population, by_author = empty(), defaultdict(empty)
    observations = np.stack((values, eligible), axis=1)
    keys = [key if isinstance(key, str) else None for key in author_keys]
    release_order = release.argsort(kind="stable").to_numpy()
    rows = np.empty((len(index), len(AUTHOR_FEATURES)))
    cursor = 0
    for i in prediction.argsort(kind="stable").to_numpy():
        while cursor < len(release_order) and release.iloc[release_order[cursor]] < prediction.iloc[i]:
            j = release_order[cursor]
            population += observations[j]
            if keys[j] is not None:
                by_author[keys[j]] += observations[j]
            cursor += 1
        key = keys[i]
        own = by_author[key] if key is not None else empty()
        sums, counts = own
        other_sums, other_counts = population - own
        prior = np.divide(other_sums, other_counts, out=NEUTRAL_RATES.copy(), where=other_counts > 0)
        rates = (sums + PRIOR_STRENGTH * prior) / (counts + PRIOR_STRENGTH)
        rows[i] = [counts[0], *rates, key is None, counts[0] == 0, counts[4] == 0, counts[5] == 0]
    return pd.DataFrame(rows, index=index, columns=AUTHOR_FEATURES)
