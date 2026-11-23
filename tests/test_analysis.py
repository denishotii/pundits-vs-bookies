"""Analysis tests: the building blocks on numbers worked out by hand, the models on
simulated games where we know the answer, and the committed results/ against a fresh run."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvb import config
from pvb.analysis import (
    TEXT_FEATURES, backtest, brier, early_odds_time, implied_probabilities, log_loss, mean_with_ci,
    odds_movement, predictions, prepare_football, prepare_tennis, roi_with_ci, run, text_beyond_odds,
)

ROOT = Path(__file__).resolve().parents[1]
SATURDAY = pd.Timestamp("2025-03-08 15:00", tz="UTC")


def odds_columns(frame, kind, probabilities, outcomes=("home", "draw", "away")):
    """Odds of exactly 1/p, so the implied probabilities are p (no margin)."""
    for i, outcome in enumerate(outcomes):
        frame[f"{kind}_{outcome}"] = 1 / np.asarray(probabilities, float)[:, i]


def draw(rng, probabilities):
    """One outcome per row, drawn with the row's probabilities."""
    return np.minimum((rng.random(len(probabilities))[:, None] > probabilities.cumsum(axis=1)).sum(axis=1), 2)


def simulated_football(pick_knows_more: bool, n=6000, seed=7):
    """Games whose result follows the market, or the market plus what the pick knows.

    If the pick knows more, the true probabilities are 75% market and 25% the
    picked outcome. Two thirds of the games are training games.
    """
    rng = np.random.default_rng(seed)
    market = rng.dirichlet([5, 3, 3.5], size=n)
    pick = draw(rng, market)
    truth = 0.75 * market + 0.25 * np.eye(3)[pick] if pick_knows_more else market
    y = draw(rng, truth)
    letters, goals = np.array(["H", "D", "A"]), np.array([[1, 0], [1, 1], [0, 1]])
    games = pd.DataFrame({
        "article_id": np.arange(n), "season": np.where(np.arange(n) < 2 * n // 3, "2425", "2526"),
        "kickoff": SATURDAY, "result": letters[y], "pick_result": letters[pick],
        "pick_home_goals": goals[pick, 0], "pick_away_goals": goals[pick, 1],
    })
    for column in TEXT_FEATURES["football"]:
        games[column] = rng.random(n)  # noise
    odds_columns(games, "closing_odds", market * 1.05)  # a 5% margin
    odds_columns(games, "best_closing_odds", market * 1.01)
    odds_columns(games, "early_odds", market * 1.05)
    return prepare_football(games)


# Building blocks ---------------------------------------------------------------

def test_implied_probabilities_remove_the_margin():
    probabilities = implied_probabilities(np.array([[2.0, 4.0, 4.0], [1.9, 3.8, 3.8]]))
    assert probabilities == pytest.approx(np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]))


def test_log_loss_and_brier_per_game():
    p, y = np.array([[0.5, 0.25, 0.25], [0.2, 0.2, 0.6]]), np.array([0, 2])
    assert log_loss(p, y) == pytest.approx([np.log(2), -np.log(0.6)])
    assert brier(p, y) == pytest.approx([0.25 + 0.0625 + 0.0625, 0.04 + 0.04 + 0.16])


def test_mean_with_ci():
    result = mean_with_ci([1, 2, 3, 4])
    se = np.std([1, 2, 3, 4], ddof=1) / 2
    assert result["mean"] == 2.5
    assert (result["ci_low"], result["ci_high"]) == pytest.approx((2.5 - 1.96 * se, 2.5 + 1.96 * se))
    assert result["p_value"] < 0.001


def test_splits_cover_each_season_once():
    for sport, seasons in [("football", config.FOOTBALL_SEASONS), ("tennis", [str(y) for y in config.TENNIS_YEARS])]:
        assigned = [season for split in config.SPLITS[sport].values() for season in split]
        assert sorted(assigned) == sorted(seasons), sport  # every season once: no overlap between train and test


# Preparing the matched tables ---------------------------------------------------

def test_prepare_football_codes_result_and_pick():
    games = pd.DataFrame({
        "result": ["H", "A", "D"], "pick_result": ["D", "A", None],
        "pick_home_goals": [1, 0, None], "pick_away_goals": [1, 3, None], "season": ["2122", "2526", "2627"],
    })
    for column in TEXT_FEATURES["football"]:
        games[column] = 0.0
    games = prepare_football(games)
    assert len(games) == 2  # the preview without a pick is dropped
    assert games[["y", "pick", "pick_home", "pick_draw", "pick_goal_diff"]].values.tolist() == [[0, 1, 0, 1, 0], [2, 2, 0, 0, -3]]
    assert games["sample"].tolist() == ["train", "test"]


@pytest.mark.parametrize("pick_side, pick_sets, best_of, expected", [
    ("a", "two", 3, 1),       # two sets is straight sets in best of 3
    ("b", "two", 3, -1),
    ("a", "three", 3, 0),
    ("a", "three", 5, 1),     # three sets is straight sets in best of 5
    ("a", "four", 5, 0),
    ("b", "straight", 5, -1),
    ("a", None, 3, 0),
])
def test_prepare_tennis_codes_straight_sets_picks(pick_side, pick_sets, best_of, expected):
    games = prepare_tennis(pd.DataFrame({
        "date": [pd.Timestamp("2025-07-13")], "a_won": [False],
        "pick_side": [pick_side], "pick_sets": [pick_sets], "best_of": [best_of],
    }))
    assert games.loc[0, "pick_a_in_straight_sets"] == expected
    assert (games.loc[0, "y"], games.loc[0, "pick_a"]) == (1, int(pick_side == "a"))
    assert (games.loc[0, "season"], games.loc[0, "sample"]) == ("2025", "train")


# Test 1: the models find information the odds miss, and only then ------------------

def scores(games):
    table = text_beyond_odds(games, "football", predictions(games, "football"))
    return table[table["sample"] == "test"].set_index("model")


def test_pick_model_wins_when_the_pick_knows_more_than_the_odds():
    table = scores(simulated_football(pick_knows_more=True))
    assert table.loc["odds + pick", "ci_high"] < 0  # lower log loss than the odds alone, clearly


def test_text_models_do_not_win_when_the_text_knows_nothing_new():
    table = scores(simulated_football(pick_knows_more=False))
    assert table.loc["odds + pick", "ci_high"] > 0
    assert table.loc["odds + pick + text", "ci_high"] > 0
    # Here the market is the truth: fitting the odds model costs a little (estimating it), never much
    assert -0.005 < table.loc["market", "change_vs_odds"] <= 0


def test_models_never_see_the_test_results():
    games = simulated_football(pick_knows_more=True)
    shuffled = games.copy()
    test = shuffled["sample"] == "test"
    shuffled.loc[test, "y"] = shuffled.loc[test, "y"].sample(frac=1, random_state=1).to_numpy()
    for name, probabilities in predictions(games, "football").items():
        np.testing.assert_allclose(probabilities, predictions(shuffled, "football")[name], err_msg=name)


# Test 2: odds movement ------------------------------------------------------------

def movement(picks, early, closing, bands=1, published=("2025-03-06 12:00",)):
    """Saturday games; by default every preview came out on Thursday, before the early odds."""
    games = pd.DataFrame({"pick": picks, "kickoff": SATURDAY})
    games["published"] = pd.to_datetime(list(published) * (len(picks) // len(published)), utc=True)
    odds_columns(games, "early_odds", early)
    odds_columns(games, "closing_odds", closing)
    return odds_movement(games, bands=bands).set_index("games")


def test_odds_moving_towards_the_pick_show_as_excess():
    # two games picked home and two picked away, all starting at 50/25/25
    row = movement([0, 0, 2, 2], [[0.5, 0.25, 0.25]] * 4, [[0.54, 0.23, 0.23]] * 2 + [[0.5, 0.25, 0.25]] * 2).loc["All games"]
    # home picks: +4 points against 0 for the unpicked home sides; away picks: 0 against -2
    assert (row["pick_move_pp"], row["excess_pp"]) == pytest.approx((2, 3))


def test_odds_moving_the_same_way_for_every_game_are_not_excess():
    """The favourite shortened in every game, picked or not: nothing to do with the pick."""
    row = movement([0, 0, 2, 2], [[0.5, 0.25, 0.25]] * 4, [[0.54, 0.23, 0.23]] * 4).loc["All games"]
    assert row["pick_move_pp"] == pytest.approx(1)
    assert row["excess_pp"] == pytest.approx(0)


def test_bands_compare_picks_with_outcomes_of_similar_early_probability():
    """Strong favourites shorten and get picked more often, but within each band picking changes nothing."""
    strong, weak = [0.7, 0.15, 0.15], [0.4, 0.3, 0.3]
    picks = [0, 0, 0, 2] + [0, 2, 2, 2]
    early = [strong] * 4 + [weak] * 4
    closing = [[0.74, 0.13, 0.13]] * 4 + [weak] * 4
    assert movement(picks, early, closing, bands=1).loc["All games", "excess_pp"] > 0.5  # no bands: fooled
    assert movement(picks, early, closing, bands=2).loc["All games", "excess_pp"] == pytest.approx(0)


def test_previews_are_split_by_when_the_early_odds_were_taken():
    table = movement([0, 2], [[0.5, 0.25, 0.25]] * 2, [[0.54, 0.23, 0.23]] * 2,
                     published=("2025-03-06 12:00", "2025-03-07 18:00"))  # Thursday; Friday evening
    assert table.loc["Preview before the early odds", "n"] == 1
    assert table.loc["Preview after the early odds", "n"] == 1


@pytest.mark.parametrize("kickoff_uk, expected_uk", [
    ("2025-03-09 14:00", "2025-03-07 15:00"),  # Sunday -> Friday
    ("2025-03-10 20:00", "2025-03-07 15:00"),  # Monday -> Friday
    ("2025-03-07 20:00", "2025-03-07 15:00"),  # Friday -> same day
    ("2025-03-12 19:30", "2025-03-11 15:00"),  # Wednesday -> Tuesday
    ("2024-08-17 15:00", "2024-08-16 15:00"),  # Saturday in British summer time
])
def test_early_odds_time(kickoff_uk, expected_uk):
    kickoff = pd.Series([pd.Timestamp(kickoff_uk, tz="Europe/London").tz_convert("UTC")])
    assert early_odds_time(kickoff)[0] == pd.Timestamp(expected_uk, tz="Europe/London")


# Test 3: backtest -------------------------------------------------------------------

def backtest_rows(probabilities, average, best, y, pick):
    games = pd.DataFrame({"sample": "test", "y": y, "pick": pick})
    for kind, values in [("closing_odds", average), ("best_closing_odds", best)]:
        for i, outcome in enumerate(["home", "draw", "away"]):
            games[f"{kind}_{outcome}"] = np.asarray(values, float)[:, i]
    probabilities = np.asarray(probabilities, float)
    predicted = {name: probabilities for name in ["odds", "odds + pick", "odds + pick + text"]}
    table = backtest(games, "football", predicted)
    return table[table["long_shots"] == "left out"].set_index(["odds", "strategy"]), table


def test_backtest_bets_where_the_model_sees_value():
    table, _ = backtest_rows(
        probabilities=[[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]],
        average=[[1.6, 3.8, 5.5], [3.6, 2.9, 2.06]],
        best=[[1.9, 4.1, 6.0], [4.0, 3.0, 2.3]],
        y=[0, 1], pick=[1, 1],
    )
    # expected profit at the best odds: game 1 home +14%, draw +2.5%; game 2 away +15%
    rows = [("best", "odds model, edge > 0%"), ("best", "odds model, edge > 5%"),
            ("best", "back every pick"), ("best", "back every favourite"), ("average", "odds model, edge > 0%")]
    expected = [(3, -1.1), (2, -0.1), (2, 1.0), (2, -0.1), (1, -1.0)]  # (bets, profit)
    for row, (bets, profit) in zip(rows, expected):
        assert (table.loc[row, "bets"], table.loc[row, "profit"]) == pytest.approx((bets, profit)), row
    assert table.loc[("best", "odds model, edge > 5%"), "roi"] == pytest.approx(-0.05)


def test_backtest_skips_long_shots():
    main, everything = backtest_rows(probabilities=[[0.05, 0.15, 0.8]], average=[[24, 8, 1.15]], best=[[51, 8.2, 1.2]], y=[2], pick=[2])
    assert main.loc[("best", "odds model, edge > 0%"), "bets"] == 1  # the draw at 8.2, not the home win at 51
    included = everything[everything["long_shots"] == "included"].set_index(["odds", "strategy"])
    assert included.loc[("best", "odds model, edge > 0%"), "bets"] == 2


def test_roi_interval_counts_each_game_once():
    # game 1: two bets, one won at 3.0 (+2) and one lost (-1); game 2: one lost bet
    result = roi_with_ci(np.array([1.0, -1.0]), np.array([2, 1]))
    assert result["roi"] == 0
    assert result["ci_high"] == pytest.approx(1.96 * np.sqrt(2 * 2) / 3)
    # with one bet per game it is the usual interval for a mean
    returns = np.array([0.9, -1.0, 2.5, -1.0])
    single = roi_with_ci(returns, np.ones(4, int))
    assert (single["ci_low"], single["ci_high"]) == pytest.approx((mean_with_ci(returns)["ci_low"], mean_with_ci(returns)["ci_high"]))


# The real data ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def fresh_results(tmp_path_factory):
    if not (config.PROCESSED / "matched_football.csv").exists() or not (config.PROCESSED / "matched_tennis.csv").exists():
        pytest.skip("no matched tables yet; run `python -m pvb match`")
    folder = tmp_path_factory.mktemp("results")
    run(folder)
    return folder


def test_analyse_writes_every_table_and_chart(fresh_results):
    for name in ["picks_vs_favourite", "text_beyond_odds", "odds_movement", "backtest"]:
        assert not pd.read_csv(fresh_results / f"{name}.csv").empty
    for name in ["picks_vs_favourite", "text_beyond_odds", "odds_movement"]:
        assert (fresh_results / f"{name}.png").stat().st_size > 10_000


@pytest.mark.parametrize("name", ["picks_vs_favourite", "text_beyond_odds", "odds_movement", "backtest"])
def test_committed_results_are_up_to_date(fresh_results, name):
    """After a new `match` or `features`, run `python -m pvb analyse` again; this catches a forgotten one."""
    committed = ROOT / "results" / f"{name}.csv"
    if not committed.exists():
        pytest.skip(f"no results/{name}.csv yet; run `python -m pvb analyse`")
    pd.testing.assert_frame_equal(pd.read_csv(fresh_results / f"{name}.csv"), pd.read_csv(committed),
                                  check_exact=False, rtol=0, atol=1e-5, obj=name)
