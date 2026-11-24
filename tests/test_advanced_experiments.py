import json

import numpy as np
import pandas as pd
import pytest

from pvb.advanced import experiments
from pvb.advanced.dataset import FootballDataset, ODDS_COLUMNS
from pvb.advanced.market import decimal_to_probabilities
from pvb.advanced.models import L2_GRID, MarketLogistic


def test_runner_isolates_folds_and_holdouts_and_pools_by_match(tmp_path, monkeypatch):
    rows = []
    outcomes = []
    odds = []
    seasons = ["2122", "2223", "2324", "2425", "2526", "2627"]
    for i, (season, size) in enumerate(zip(seasons, [4, 3, 4, 5, 1, 1], strict=True)):
        for number in range(size):
            kickoff = pd.Timestamp(f"{2021 + i}-08-{number + 1:02}T12:00:00Z")
            rows.append((f"{season}-{number}", season, kickoff))
            outcomes.append(number % 3 if i < 4 else 99)
            odds.append([2 + i / 10 + number / 5, 3, 4] if i < 4 else [np.nan] * 3)
    # The last 2122 result becomes available exactly at the first cutoff in 2223.
    rows[3] = ("2122-3", "2122", pd.Timestamp("2022-08-01T08:00:00Z"))
    metadata = pd.DataFrame(rows, columns=["article_id", "season", "kickoff"])
    metadata["prediction_at"] = metadata["kickoff"] - pd.Timedelta(hours=1)
    metadata["outcome_available_at"] = metadata["kickoff"] + pd.Timedelta(hours=3)
    inputs = pd.DataFrame(odds, columns=ODDS_COLUMNS["early"]).assign(
        closing_odds_home=np.nan, best_closing_odds_home=np.nan, home_goals=np.nan,
        pick_draw=np.nan, nb_home_vs_draw=np.nan,
    )
    data = FootballDataset(pd.Series(outcomes), metadata, pd.DataFrame({"text_preview": "body"}, index=metadata.index), inputs, {})

    def load(market, **kwargs):
        assert market == "early"
        return data

    fits = []

    class CheckedLogistic(MarketLogistic):
        def fit(self, market, y):
            fits.append((market.copy(), y.copy()))
            return super().fit(market, y)

    monkeypatch.setattr(experiments, "load_football", load)
    monkeypatch.setitem(experiments.MODEL_TYPES, "M1", CheckedLogistic)
    scores = experiments.run_experiment("early", tmp_path)

    assert len(fits) == 3 * len(L2_GRID)
    expected_training = [np.arange(3), np.arange(7), np.arange(11)]
    for i, (market, y) in enumerate(fits):
        positions = expected_training[i % 3]
        np.testing.assert_allclose(market, decimal_to_probabilities(np.asarray(odds)[positions]))
        np.testing.assert_array_equal(y, np.asarray(outcomes)[positions])

    output = tmp_path / "early"
    oof = pd.read_csv(output / "oof_predictions.csv", dtype={"season": str})
    assert oof["article_id"].tolist() == metadata.iloc[4:16]["article_id"].tolist()
    assert oof["article_id"].is_unique
    assert oof.groupby("season").size().to_dict() == {"2223": 3, "2324": 4, "2425": 5}
    assert len(oof.columns) == 15  # identifiers, target, nine probabilities, two paired differences
    splits = pd.read_csv(output / "folds.csv")
    assert splits["purged_train_rows"].tolist() == [1, 0, 0]
    for _, history in scores.groupby("model"):
        fold_scores = history.query("scope != 'pooled'")
        pooled = history.query("scope == 'pooled'").iloc[0]
        assert pooled.log_loss == pytest.approx(np.average(fold_scores.log_loss, weights=fold_scores.matches))
    manifest = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert manifest["independent_evaluation"] is False
    assert manifest["development_seasons"] == seasons[:4]
    search = pd.read_csv(output / "regularization_search.csv")
    for model, l2 in manifest["selected_l2"].items():
        eligible = search.query("model == @model and scope == 'pooled' and converged")
        assert l2 == eligible.sort_values(["log_loss", "l2"], ascending=[True, False]).iloc[0].l2
