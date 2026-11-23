"""Checks on the real matched tables in data/processed/ (they are on GitHub, so CI runs these too).

The unit tests in test_matching.py show the rules work on small examples;
these show the output on the full data obeys them.
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
FOOTBALL = PROCESSED / "matched_football.csv"
TENNIS = PROCESSED / "matched_tennis.csv"


@pytest.fixture(scope="module")
def football():
    if not FOOTBALL.exists():
        pytest.skip("no matched_football.csv yet; run `python -m pvb match`")
    return pd.read_csv(FOOTBALL, parse_dates=["published", "kickoff"])


@pytest.fixture(scope="module")
def tennis():
    if not TENNIS.exists():
        pytest.skip("no matched_tennis.csv yet; run `python -m pvb match`")
    return pd.read_csv(TENNIS, parse_dates=["published", "date"])


def test_football_one_row_per_preview_and_per_game(football):
    assert football["article_id"].is_unique
    assert not football.duplicated(["league", "season", "home_team", "away_team"]).any()


def test_football_previews_come_out_before_kickoff(football):
    assert (football["published"] < football["kickoff"]).all()
    assert football["hours_before_kickoff"].between(0, 5 * 24).all()


def test_football_each_sportsmole_name_is_always_the_same_team(football):
    """A wrong match would pair a Sports Mole name with a second football-data team."""
    names = pd.concat([
        football[["side_a", "home_team"]].set_axis(["sportsmole", "odds"], axis=1),
        football[["side_b", "away_team"]].set_axis(["sportsmole", "odds"], axis=1),
    ])
    teams_per_name = names.groupby("sportsmole")["odds"].nunique()
    assert (teams_per_name == 1).all(), teams_per_name[teams_per_name > 1]


def test_football_values_are_valid(football):
    assert football["result"].isin(["H", "D", "A"]).all()
    assert football["pick_result"].dropna().isin(["H", "D", "A"]).all()
    odds = football.filter(like="odds_").stack().dropna()  # football-data misses a few early odds
    assert (odds > 1).all()  # decimal odds are always above 1


def test_tennis_one_row_per_preview_and_per_game(tennis):
    assert tennis["article_id"].is_unique
    assert not tennis.duplicated(["date", "player_a", "player_b"]).any()


def test_tennis_previews_come_out_on_or_before_match_day(tennis):
    days = (tennis["date"] - tennis["published"].dt.tz_convert(None).dt.normalize()).dt.days
    assert days.between(0, 5).all()


def test_tennis_each_full_name_is_always_the_same_player(tennis):
    names = pd.concat([
        tennis[["side_a", "player_a"]].set_axis(["sportsmole", "odds"], axis=1),
        tennis[["side_b", "player_b"]].set_axis(["sportsmole", "odds"], axis=1),
    ])
    names = names[names["sportsmole"].str.contains(" ")]  # a bare surname like "Zverev" may be either brother
    # tennis-data writes some players two ways ("McDonald M." and "Mcdonald M."), so compare letters only
    letters = names["odds"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    players_per_name = letters.groupby(names["sportsmole"]).nunique()
    assert (players_per_name == 1).all(), players_per_name[players_per_name > 1]


def test_tennis_has_no_walkovers(tennis):
    assert set(tennis["status"]) <= {"Completed", "Retired"}


def test_tennis_winner_is_the_player_with_more_sets(tennis):
    """Checks the a/b swap: in finished matches the winner took more sets."""
    finished = tennis[tennis["status"] == "Completed"].dropna(subset=["sets_a", "sets_b"])
    assert ((finished["sets_a"] > finished["sets_b"]) == finished["a_won"]).all()


@pytest.mark.parametrize("name, columns", [
    ("matched_football.csv", ["side_a", "side_b", "pick_home_goals", "pick_away_goals", "pick_result"]),
    ("matched_tennis.csv", ["side_a", "side_b", "pick_winner", "pick_sets"]),
    ("features.csv", ["sport", "pick_result", "pick_winner"]),
])
def test_tables_agree_with_picks(name, columns):
    """After a new `parse`, `features` and `match` must be re-run too; this catches a forgotten one."""
    if not (PROCESSED / name).exists() or not (PROCESSED / "picks.csv").exists():
        pytest.skip(f"no {name} or picks.csv yet")
    picks = pd.read_csv(PROCESSED / "picks.csv").set_index("article_id")
    table = pd.read_csv(PROCESSED / name)
    assert table["article_id"].isin(picks.index).all()
    expected = picks.loc[table["article_id"], columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(table[columns], expected, check_dtype=False, obj=name)
