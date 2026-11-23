"""Matching tests on small invented tables shaped like picks.csv and the odds files."""

import pandas as pd
import pytest

from pvb.matching import FOOTBALL_COLUMNS, match_football, match_tennis, same_player, same_team, suggest_aliases


def previews(*rows, sport="football", competition="premier-league"):
    """rows: (article_id, side_a, side_b, published[, pick_winner])"""
    records = []
    for article_id, a, b, published, *pick in rows:
        url = f"https://www.sportsmole.co.uk/{sport}/{competition}/preview/a-vs-b-prediction_{article_id}.html"
        records.append({"article_id": article_id, "url": url, "sport": sport, "side_a": a, "side_b": b, "published": published,
                        "pick_home_goals": 1, "pick_away_goals": 0, "pick_result": "H",
                        "pick_winner": pick[0] if pick else None, "pick_sets": "three" if pick else None})
    return pd.DataFrame(records)


def football_games(*rows):
    """rows: (home, away, kick-off in UK time)"""
    games = pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "kickoff"])
    games["kickoff"] = pd.to_datetime(games["kickoff"]).dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    for column in FOOTBALL_COLUMNS:
        if column not in games:
            games[column] = 2.0
    games["FTR"], games["league"], games["season"] = "H", "E0", "2425"
    return games


def tennis_games(*rows):
    """rows: (winner, loser, date, odds for the winner, odds for the loser)"""
    games = pd.DataFrame(rows, columns=["Winner", "Loser", "Date", "AvgW", "AvgL"])
    games["Date"] = pd.to_datetime(games["Date"])
    games["MaxW"], games["MaxL"] = games["AvgW"] + 0.1, games["AvgL"] + 0.1
    games["Wsets"], games["Lsets"], games["WRank"], games["LRank"] = 3, 1, 1, 2
    games["tour"], games["Tournament"], games["Surface"], games["Round"] = "ATP", "Wimbledon", "Grass", "The Final"
    games["Best of"], games["Comment"] = 5, "Completed"
    return games


# Football ----------------------------------------------------------------------

def test_finds_the_game_in_the_right_season():
    matched = match_football(
        previews((1, "Manchester United", "Liverpool", "2025-03-07T10:00:00+00:00")),
        football_games(("Man United", "Liverpool", "2024-03-09 15:00"), ("Man United", "Liverpool", "2025-03-09 15:00")),
    )
    assert len(matched) == 1
    assert matched.loc[0, "kickoff"].year == 2025
    assert matched.loc[0, "hours_before_kickoff"] == 53.0  # 10:00 Friday to 15:00 Sunday, UK still on winter time


def test_preview_must_come_out_before_kickoff_and_at_most_5_days_before():
    games = football_games(("Arsenal", "Chelsea", "2025-03-09 15:00"))  # 15:00 UK = 15:00 UTC in March
    assert match_football(previews((1, "Arsenal", "Chelsea", "2025-03-09T16:00:00+00:00")), games).empty  # after
    assert len(match_football(previews((2, "Arsenal", "Chelsea", "2025-03-04T16:00:00+00:00")), games)) == 1  # 4 d 23 h
    assert match_football(previews((3, "Arsenal", "Chelsea", "2025-03-04T14:00:00+00:00")), games).empty  # 5 d 1 h


def test_football_odds_and_result_land_in_the_right_columns():
    games = football_games(("Arsenal", "Chelsea", "2025-03-09 15:00"))
    games[["AvgH", "AvgD", "AvgA"]] = [1.5, 4.0, 6.0]
    games[["AvgCH", "AvgCD", "AvgCA"]] = [1.6, 4.2, 5.5]
    games[["MaxCH", "MaxCD", "MaxCA"]] = [1.7, 4.5, 6.5]
    games["FTHG"], games["FTAG"], games["FTR"] = 0, 2, "A"
    row = match_football(previews((1, "Arsenal", "Chelsea", "2025-03-07T10:00:00+00:00")), games).iloc[0]
    assert (row["early_odds_home"], row["early_odds_draw"], row["early_odds_away"]) == (1.5, 4.0, 6.0)
    assert (row["closing_odds_home"], row["closing_odds_draw"], row["closing_odds_away"]) == (1.6, 4.2, 5.5)
    assert (row["best_closing_odds_home"], row["best_closing_odds_draw"], row["best_closing_odds_away"]) == (1.7, 4.5, 6.5)
    assert (row["home_goals"], row["away_goals"], row["result"]) == (0, 2, "A")


def test_cup_tie_preview_is_not_matched_to_the_league_game():
    matched = match_football(
        previews((1, "Arsenal", "Chelsea", "2025-03-05T10:00:00+00:00"), competition="arsenal/fa-cup"),
        football_games(("Arsenal", "Chelsea", "2025-03-09 15:00")),
    )
    assert matched.empty


def test_home_and_away_must_be_in_the_same_order():
    matched = match_football(
        previews((1, "Liverpool", "Manchester United", "2025-03-07T10:00:00+00:00")),
        football_games(("Man United", "Liverpool", "2025-03-09 15:00")),
    )
    assert matched.empty


def test_womens_team_is_not_matched_to_the_mens_game():
    matched = match_football(
        previews((1, "Arsenal Women", "Chelsea Women", "2025-03-07T10:00:00+00:00")),
        football_games(("Arsenal", "Chelsea", "2025-03-09 15:00")),
    )
    assert matched.empty


def test_inter_is_not_mistaken_for_milan():
    games = football_games(("Milan", "Juventus", "2025-03-09 19:45"))
    assert match_football(previews((1, "Inter Milan", "Juventus", "2025-03-07T10:00:00+00:00")), games).empty
    assert len(match_football(previews((2, "AC Milan", "Juventus", "2025-03-07T10:00:00+00:00")), games)) == 1


def test_preview_with_two_possible_games_is_left_out():
    matched = match_football(
        previews((1, "Arsenal", "Chelsea", "2025-03-07T10:00:00+00:00")),
        football_games(("Arsenal", "Chelsea", "2025-03-09 15:00"), ("Arsenal", "Chelsea", "2025-03-11 19:45")),
    )
    assert matched.empty


def test_second_preview_of_a_game_keeps_the_latest():
    matched = match_football(
        previews((1, "Arsenal", "Chelsea", "2025-03-05T10:00:00+00:00"), (2, "Arsenal", "Chelsea", "2025-03-08T10:00:00+00:00")),
        football_games(("Arsenal", "Chelsea", "2025-03-09 15:00")),
    )
    assert matched["article_id"].tolist() == [2]


def test_previews_without_team_names_are_skipped():
    matched = match_football(
        previews((1, None, None, "2025-03-07T10:00:00+00:00")),
        football_games(("Arsenal", "Chelsea", "2025-03-09 15:00")),
    )
    assert matched.empty


# Tennis ------------------------------------------------------------------------

def test_tennis_keeps_the_preview_order_of_players():
    matched = match_tennis(
        previews((1, "Carlos Alcaraz", "Jannik Sinner", "2025-07-12T18:00:00+01:00", "Sinner"), sport="tennis"),
        tennis_games(("Sinner J.", "Alcaraz C.", "2025-07-13", 2.1, 1.8)),
    )
    row = matched.iloc[0]
    assert (row["player_a"], row["player_b"]) == ("Alcaraz C.", "Sinner J.")
    assert not row["a_won"]
    assert (row["closing_odds_a"], row["closing_odds_b"]) == (1.8, 2.1)  # Alcaraz lost, so his odds are AvgL
    assert (row["sets_a"], row["sets_b"]) == (1, 3)
    assert (row["rank_a"], row["rank_b"]) == (2, 1)
    assert (row["best_closing_odds_a"], row["best_closing_odds_b"]) == pytest.approx((1.9, 2.2))
    assert row["pick_side"] == "b"


def test_tennis_window_runs_from_match_day_to_5_days_before():
    games = tennis_games(("Sinner J.", "Alcaraz C.", "2025-07-13", 2.1, 1.8))

    def matched(published):
        return len(match_tennis(previews((1, "Carlos Alcaraz", "Jannik Sinner", published), sport="tennis"), games))

    assert matched("2025-07-13T09:00:00+01:00") == 1  # on the day
    assert matched("2025-07-08T09:00:00+01:00") == 1  # 5 days before
    assert matched("2025-07-07T09:00:00+01:00") == 0  # 6 days before


def test_tennis_rematch_a_week_later_does_not_block_the_match():
    matched = match_tennis(
        previews((1, "Casper Ruud", "Stefanos Tsitsipas", "2024-04-13T12:00:00+01:00"), sport="tennis"),
        tennis_games(("Tsitsipas S.", "Ruud C.", "2024-04-14", 1.8, 2.0), ("Tsitsipas S.", "Ruud C.", "2024-04-21", 1.7, 2.1)),
    )
    assert matched["date"].tolist() == [pd.Timestamp("2024-04-14")]


def test_tennis_walkover_is_left_out():
    games = tennis_games(("Djokovic N.", "De Minaur A.", "2024-07-10", 1.3, 3.8))
    games["Comment"] = "Walkover"
    assert match_tennis(previews((1, "Alex de Minaur", "Novak Djokovic", "2024-07-09T12:00:00+01:00"), sport="tennis"), games).empty


def test_tennis_match_before_the_preview_is_not_matched():
    matched = match_tennis(
        previews((1, "Carlos Alcaraz", "Jannik Sinner", "2025-07-14T10:00:00+01:00"), sport="tennis"),
        tennis_games(("Sinner J.", "Alcaraz C.", "2025-07-13", 2.1, 1.8)),
    )
    assert matched.empty


# Names -------------------------------------------------------------------------

@pytest.mark.parametrize("odds_name, sportsmole_name, expected", [
    ("Brighton", "Brighton & Hove Albion", True),
    ("Atalanta", "Atalanta BC", True),
    ("Ath Madrid", "Atletico Madrid", True),
    ("Real Madrid", "Atletico Madrid", False),
    ("Sheffield United", "Sheffield Wednesday", False),
    ("Newcastle", "Newcastle United Under-21s", False),
    ("Tottenham", "Tottenham Hotspur Ladies", False),
    ("Wolves", "Wolverhampton Wanderers", True),
])
def test_same_team(odds_name, sportsmole_name, expected):
    assert same_team(odds_name, sportsmole_name) is expected


@pytest.mark.parametrize("odds_name, sportsmole_name, expected", [
    ("Sinner J.", "Jannik Sinner", True),
    ("De Minaur A.", "Alex de Minaur", True),
    ("Tseng C. H.", "Chun-hsin Tseng", True),
    ("Zheng Q.", "Qinwen Zheng", True),
    ("Zheng Q.", "Zheng Qinwen", True),  # family name first
    ("Pliskova Ka.", "Karolina Pliskova", True),
    ("Zverev A.", "Mischa Zverev", False),
    ("Sinner J.", "Jannik Sinnerr", False),
    ("Bouzas Maneiro J.", "Jessica Bouzas", True),  # via PLAYER_ALIASES
])
def test_same_player(odds_name, sportsmole_name, expected):
    assert same_player(odds_name, sportsmole_name) is expected


def test_unknown_names_get_a_suggestion():
    suggestions = suggest_aliases(["Arsenal", "Man United", "Spurs"], ["Arsenal", "Manchester United", "Tottenham Hotspur"])
    assert list(suggestions) == ["Spurs"]
