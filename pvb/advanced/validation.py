"""Expanding-window validation for football development seasons."""

import numpy as np
import pandas as pd

from ..config import SPLITS

FOOTBALL_SPLITS = SPLITS["football"]


def season_positions(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return positional indices for development, historical test and check."""
    seasons = metadata["season"].astype("string")
    partitions = {
        "development": FOOTBALL_SPLITS["train"],
        "test": FOOTBALL_SPLITS["test"],
        "check": FOOTBALL_SPLITS["check"],
    }
    known = {season for group in partitions.values() for season in group}
    unknown = ~seasons.isin(known)
    if unknown.any():
        raise ValueError(f"Unknown or missing football season on {unknown.sum()} rows")
    return {name: np.flatnonzero(seasons.isin(group)) for name, group in partitions.items()}


def chronological_folds(
    metadata: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Return scikit-learn-style (train, validation) positions and fold statistics.

    Positions are relative to the supplied row order. Before using an sklearn
    search with refit, subset all inputs to the development positions returned
    by season_positions and build folds on that same subset.

    Earlier-season training rows whose outcome was not available strictly
    before the earliest validation prediction are purged from that fold.
    """
    partitions = season_positions(metadata)
    development = metadata.iloc[partitions["development"]]
    seasons = development["season"].astype("string")
    required = {"article_id", "prediction_at", "kickoff", "outcome_available_at"}
    missing = sorted(required - set(development.columns))
    if missing:
        raise ValueError(f"Missing chronological metadata: {missing}")
    ids = development["article_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Development metadata must have unique, nonmissing article IDs")
    prediction = development["prediction_at"]
    kickoff = development["kickoff"]
    outcome = development["outcome_available_at"]
    for name, times in (("prediction_at", prediction), ("kickoff", kickoff), ("outcome_available_at", outcome)):
        if not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any():
            raise ValueError(f"{name} must contain nonmissing timezone-aware timestamps")
    if (prediction.gt(kickoff) | outcome.le(kickoff)).any():
        raise ValueError("Expected prediction_at <= kickoff < outcome_available_at")

    splits = []
    stats = []
    development_seasons = FOOTBALL_SPLITS["train"]
    for number, validation_season in enumerate(development_seasons[1:], start=1):
        training_seasons = development_seasons[:number]
        candidates = seasons.isin(training_seasons)
        validation = seasons.eq(validation_season)
        if not validation.any():
            raise ValueError(f"Fold {number}: no validation rows for {validation_season}")
        cutoff = prediction[validation].min()
        training = candidates & outcome.lt(cutoff)
        if not training.any():
            raise ValueError(f"Fold {number}: no training rows with outcomes available before {cutoff}")
        splits.append((
            partitions["development"][np.flatnonzero(training)],
            partitions["development"][np.flatnonzero(validation)],
        ))
        stats.append({
            "fold": number,
            "train_seasons": training_seasons,
            "validation_season": validation_season,
            "train_rows": int(training.sum()),
            "validation_rows": int(validation.sum()),
            "purged_train_rows": int((candidates & ~training).sum()),
            "validation_cutoff": cutoff,
            "last_training_outcome_available_at": outcome[training].max(),
        })
    return splits, pd.DataFrame(stats)
