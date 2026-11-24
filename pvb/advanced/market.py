"""Football market probabilities in home, draw, away order (H=0, D=1, A=2)."""

import numpy as np
import pandas as pd

from .dataset import ODDS_COLUMNS, OUTCOME_CODES, Market

CLASS_CODES = tuple(sorted(OUTCOME_CODES, key=OUTCOME_CODES.__getitem__))
CLASS_NAMES = ("home", "draw", "away")
N_CLASSES = len(CLASS_CODES)


def validate_probabilities(values: np.ndarray, *, positive: bool = False) -> np.ndarray:
    """Validate a nonempty (matches, 3) array in H/D/A order."""
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES or len(probabilities) == 0:
        raise ValueError("Expected a nonempty probability array with three H/D/A columns")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if positive and (probabilities == 0).any():
        raise ValueError("Market probabilities for log features must be strictly positive")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Probability rows must sum to one")
    return probabilities


def decimal_to_implied(odds: np.ndarray) -> np.ndarray:
    """Return reciprocal decimal odds, including the market margin, in H/D/A order."""
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 2 or odds.shape[1] != N_CLASSES or len(odds) == 0:
        raise ValueError("Expected a nonempty odds array with three H/D/A columns")
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Decimal odds must be finite and greater than one")
    return np.reciprocal(odds)


def decimal_to_probabilities(odds: np.ndarray) -> np.ndarray:
    """Remove overround by dividing each implied probability by its row total."""
    implied = decimal_to_implied(odds)
    return implied / implied.sum(axis=1, keepdims=True)


def market_probabilities(numerical: pd.DataFrame, market: Market) -> np.ndarray:
    """Proportionally normalize implied probabilities for one market's average odds."""
    if market not in ODDS_COLUMNS:
        raise ValueError(f"Unknown market {market!r}; use 'early' or 'closing'")
    columns = ODDS_COLUMNS[market]
    missing = set(columns) - set(numerical.columns)
    if missing:
        raise ValueError(f"Missing {market} odds columns: {sorted(missing)}")
    return decimal_to_probabilities(numerical.loc[:, list(columns)].to_numpy(dtype=float))


def validate_targets(target: np.ndarray, n_rows: int) -> np.ndarray:
    """Validate one encoded H/D/A outcome per match."""
    target = np.asarray(target)
    if target.shape != (n_rows,) or not np.isin(target, np.arange(N_CLASSES)).all():
        raise ValueError("Expected one outcome coded 0, 1 or 2 per match")
    return target.astype(int)
