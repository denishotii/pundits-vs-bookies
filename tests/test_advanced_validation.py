import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import check_cv

from pvb.advanced.validation import chronological_folds, season_positions


@pytest.fixture
def metadata():
    rows = [
        ("check", "2627", "2026-08-01"),
        ("s3", "2324", "2023-08-01"),
        ("s1", "2122", "2021-08-01"),
        ("test", "2526", "2025-08-01"),
        ("s4", "2425", "2024-08-01"),
        ("s2", "2223", "2022-08-01"),
        ("late", "2122", "2022-08-02"),
        ("tie", "2122", "2022-07-31 21:00"),
        ("during", "2122", "2022-07-31 22:00"),
        ("s2-later", "2223", "2022-08-03"),
    ]
    frame = pd.DataFrame(rows, columns=["article_id", "season", "prediction_at"])
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], format="mixed", utc=True)
    frame["kickoff"] = frame["prediction_at"]
    frame["outcome_available_at"] = frame["kickoff"] + pd.Timedelta(hours=3)
    return frame.set_axis([50, 7, 50, 3, 9, 81, 5, 6, 10, 11])


def test_expanding_folds_purge_unavailable_outcomes_and_keep_holdouts_reserved(metadata):
    folds, stats = chronological_folds(metadata)
    partitions = season_positions(metadata)
    assert partitions["test"].tolist() == [3]
    assert partitions["check"].tolist() == [0]
    assert partitions["development"].tolist() == [1, 2, 4, 5, 6, 7, 8, 9]
    assert [(train.tolist(), valid.tolist()) for train, valid in folds] == [
        ([2], [5, 9]),
        ([2, 5, 6, 7, 8, 9], [1]),
        ([1, 2, 5, 6, 7, 8, 9], [4]),
    ]
    assert stats["train_seasons"].tolist() == [("2122",), ("2122", "2223"), ("2122", "2223", "2324")]
    assert stats["validation_season"].tolist() == ["2223", "2324", "2425"]
    assert stats["purged_train_rows"].tolist() == [3, 0, 0]
    assert stats["train_rows"].tolist() == [1, 6, 7]
    assert stats["validation_rows"].tolist() == [2, 1, 1]
    for train, valid in folds:
        assert not np.intersect1d(train, valid).size
        assert set(np.r_[train, valid]) <= set(partitions["development"])
        cutoff = metadata.iloc[valid]["prediction_at"].min()
        assert metadata.iloc[train]["kickoff"].max() < cutoff
        assert metadata.iloc[train]["outcome_available_at"].max() < cutoff


def test_sklearn_split_positions_on_development_subset(metadata):
    development = metadata.iloc[season_positions(metadata)["development"]]
    folds, _ = chronological_folds(development)
    again, _ = chronological_folds(development)
    for (train, valid), (repeat_train, repeat_valid) in zip(check_cv(folds).split(development), again):
        np.testing.assert_array_equal(train, repeat_train)
        np.testing.assert_array_equal(valid, repeat_valid)
        assert set(development.iloc[valid]["article_id"]) <= {"s2", "s2-later", "s3", "s4"}


def test_invalid_chronological_metadata_is_rejected(metadata):
    with pytest.raises(ValueError, match="timezone-aware"):
        chronological_folds(metadata.assign(prediction_at=metadata["prediction_at"].dt.tz_localize(None)))
    with pytest.raises(ValueError, match="Expected prediction_at <= kickoff < outcome_available_at"):
        chronological_folds(metadata.assign(outcome_available_at=metadata["kickoff"]))
    with pytest.raises(ValueError, match="no validation rows for 2324"):
        chronological_folds(metadata.loc[metadata["season"] != "2324"])
