"""Match each preview to its game in the odds data.

Sports Mole and the odds files name the same team or player differently
("Manchester United" vs "Man United", "Jannik Sinner" vs "Sinner J."), and the
same fixture comes back every season: in our pilot, 24 of 60 games found by
name alone were from the wrong season. So a preview is matched to a game only if

1. both names agree (the rules are in `same_team` and `same_player`),
2. the game starts after the preview came out, at most 5 days later: previews
   come out one to three days ahead (median 50 hours, never more than 4.6 days
   in our data), and a longer window only adds the same teams' next meeting, and
3. exactly one game passes 1 and 2. Ambiguous previews are left out, never guessed.

Some previews are of cup ties between two league clubs. Their URL usually says
so (/fa-cup/preview/...), and we skip those. If two previews still point to the
same league game, we keep the one published last: the earlier one is about a
cup tie between the same clubs a few days before.

Why not fuzzy matching: rapidfuzz scores "Angers" against "Rangers" 92 out of
100, as high as most true pairs. Fuzzy scores only suggest missing aliases.

`python -m pvb match` writes data/processed/matched_football.csv and
matched_tennis.csv. They hold no article text, so they go on GitHub.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import cache

import pandas as pd
from rapidfuzz import process

from . import config

log = logging.getLogger(__name__)

FOOTBALL_CSV = config.PROCESSED / "matched_football.csv"
TENNIS_CSV = config.PROCESSED / "matched_tennis.csv"
MAX_DAYS_BEFORE = 5
# Previews of domestic cup ties, which are not in the league odds files
CUP_URL = re.compile(
    r"/(?:fa-cup|league-cup|efl-trophy|football-league-trophy|dfb-pokal|copa-del-rey|coppa-italia|coupe-de-france)"
    r"[\w-]*/preview/"
)

# football-data's abbreviations, written out the way Sports Mole writes the club
ALIASES = {
    "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Espanol": "Espanyol",
    "Hamburg": "Hamburger SV",
    "Inter": "Inter Milan",
    "M'gladbach": "Monchengladbach",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Milan": "AC Milan",  # or "Milan" would also match "Inter Milan"
    "Nott'm Forest": "Nottingham Forest",
    "Paris SG": "Paris Saint-Germain",
    "Peterboro": "Peterborough",
    "QPR": "Queens Park Rangers",
    "Sheffield Weds": "Sheffield Wednesday",
    "St Etienne": "Saint-Etienne",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton",
}
# Sports Mole names with one of these words are women's or youth teams
NOT_FIRST_TEAM = {"women", "ladies", "femenino", "w", "under", "u18", "u19", "u21", "u23"}
# Sports Mole player names that tennis-data writes differently (the last two are Sports Mole typos)
PLAYER_ALIASES = {
    "Camila Osorio": "Maria Camila Osorio",
    "Jessica Bouzas": "Jessica Bouzas Maneiro",
    "Aryna Sabaleka": "Aryna Sabalenka",
    "Zachary Zvajda": "Zachary Svajda",
}

FOOTBALL_COLUMNS = {
    "league": "league", "season": "season", "kickoff": "kickoff",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    # "early" odds are taken on the Friday (weekend games) or Tuesday (midweek games) before
    "AvgH": "early_odds_home", "AvgD": "early_odds_draw", "AvgA": "early_odds_away",
    # closing odds are the last ones before kick-off: the average and the best bookmaker
    "AvgCH": "closing_odds_home", "AvgCD": "closing_odds_draw", "AvgCA": "closing_odds_away",
    "MaxCH": "best_closing_odds_home", "MaxCD": "best_closing_odds_draw", "MaxCA": "best_closing_odds_away",
}


# Names -------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """Lower case, no accents, no punctuation: "Atlético Madrid" -> "atletico madrid"."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


@cache
def _words(name: str) -> frozenset[str]:
    return frozenset(normalise_name(name).split())


def same_team(odds_name: str, sportsmole_name: str) -> bool:
    """Every word of football-data's name, abbreviations written out, is in Sports Mole's name.

    >>> same_team("Brighton", "Brighton & Hove Albion")
    True
    >>> same_team("Man United", "Manchester United")
    True
    >>> same_team("Man United", "Manchester United Women")  # never women's or youth teams
    False
    >>> same_team("Milan", "Inter Milan")
    False
    """
    full = _words(sportsmole_name)
    return _words(ALIASES.get(odds_name, odds_name)) <= full and not full & NOT_FIRST_TEAM


def same_player(odds_name: str, sportsmole_name: str) -> bool:
    """tennis-data writes "Surname I.": the surname must be in Sports Mole's name, the initial must fit.

    Sports Mole names that tennis-data spells differently go through PLAYER_ALIASES first.

    >>> same_player("Auger-Aliassime F.", "Felix Auger-Aliassime")
    True
    >>> same_player("Pliskova Ka.", "Kristyna Pliskova")  # Karolina is "Ka.", Kristyna "Kr."
    False
    >>> same_player("Sinner J.", "Sinner")  # some titles give only the surname
    True
    """
    parts = odds_name.split()
    initials = [p for p in parts if p.endswith(".")]
    surname = _words(" ".join(p for p in parts if not p.endswith(".")))
    full = _words(PLAYER_ALIASES.get(sportsmole_name, sportsmole_name))
    if not surname or not surname <= full:
        return False
    first_names = full - surname
    if not initials or not first_names:
        return True  # nothing to compare
    initial = normalise_name(initials[0].split(".")[0])  # "Ka." -> "ka", "J.M." -> "j"
    return any(word.startswith(initial) for word in first_names)


def name_table(sportsmole_names, odds_names, same) -> pd.DataFrame:
    """Every (Sports Mole name, odds-file name) pair that `same` says is the same team or player."""
    pairs = [(sm, od) for sm in set(sportsmole_names) for od in set(odds_names) if same(od, sm)]
    return pd.DataFrame(pairs, columns=["sportsmole_name", "odds_name"])


def suggest_aliases(odds_names, sportsmole_names) -> dict[str, str]:
    """For odds-file names that no Sports Mole name matches, the closest Sports Mole name.

    Only a suggestion for ALIASES (fuzzy matching is too loose to decide on its
    own: it scores "Rangers" and "Angers" 92 out of 100). Check each one by hand.
    """
    choices = list(set(sportsmole_names))
    return {
        od: process.extractOne(od, choices, processor=normalise_name)[0]
        for od in sorted(set(odds_names))
        if not any(same_team(od, sm) for sm in choices)
    }


# Matching ----------------------------------------------------------------------

def _keep_unique(candidates: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Leave out previews with more than one candidate, then keep the last preview of each game."""
    candidates_per_preview = candidates.groupby("article_id")["article_id"].transform("size")
    unique = candidates[candidates_per_preview == 1]
    kept = unique.sort_values(["published", "article_id"], kind="stable").drop_duplicates("game", keep="last")
    log.info("%s: %d previews matched; %d left out because two games fit; "
             "%d dropped because a later preview names the same game",
             sport, len(kept), candidates.loc[candidates_per_preview > 1, "article_id"].nunique(), len(unique) - len(kept))
    return kept


def match_football(previews: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per matched preview: the pick next to the result and the odds.

    previews: picks.csv; games: odds.load_football(). Sports Mole names the home team first.
    """
    previews = previews[(previews["sport"] == "football") & previews["side_a"].notna() & previews["side_b"].notna()]
    previews = previews[~previews["url"].str.contains(CUP_URL)]
    previews = previews.assign(published=pd.to_datetime(previews["published"], utc=True))
    games = games.reset_index(drop=True).rename_axis("game").reset_index()
    names = name_table(pd.concat([previews["side_a"], previews["side_b"]]),
                       pd.concat([games["HomeTeam"], games["AwayTeam"]]).dropna(), same_team)

    candidates = (
        previews
        .merge(names.set_axis(["side_a", "HomeTeam"], axis=1), on="side_a")
        .merge(names.set_axis(["side_b", "AwayTeam"], axis=1), on="side_b")
        .merge(games, on=["HomeTeam", "AwayTeam"])
    )
    lead = candidates["kickoff"] - candidates["published"]
    candidates = candidates[(lead > pd.Timedelta(0)) & (lead <= pd.Timedelta(days=MAX_DAYS_BEFORE))]
    matched = _keep_unique(candidates, "football").rename(columns=FOOTBALL_COLUMNS)
    matched["hours_before_kickoff"] = ((matched["kickoff"] - matched["published"]).dt.total_seconds() / 3600).round(1)
    columns = ["article_id", "published", "hours_before_kickoff", "side_a", "side_b",
               "pick_home_goals", "pick_away_goals", "pick_result", *FOOTBALL_COLUMNS.values()]
    return matched[columns].sort_values(["kickoff", "article_id"], kind="stable").reset_index(drop=True)


def match_tennis(previews: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per matched preview. Players stay in the preview's order: a = side_a, b = side_b.

    previews: picks.csv; games: odds.load_tennis(). tennis-data gives only a date
    for each match, so a preview counts if it came out on or before that date.
    For some US and Asian events the date is one day after the real match day.
    Walkovers are left out (no match was played, so bets on it are void);
    retirements stay in, with status "Retired".
    """
    previews = previews[(previews["sport"] == "tennis") & previews["side_a"].notna() & previews["side_b"].notna()]
    previews = previews.assign(published=pd.to_datetime(previews["published"], utc=True))
    games = games.reset_index(drop=True).rename_axis("game").reset_index()
    names = name_table(pd.concat([previews["side_a"], previews["side_b"]]),
                       pd.concat([games["Winner"], games["Loser"]]).dropna(), same_player)

    # side_a may have won or lost, so try both ways round
    pairs = previews.merge(names.set_axis(["side_a", "player_a"], axis=1), on="side_a")
    pairs = pairs.merge(names.set_axis(["side_b", "player_b"], axis=1), on="side_b")
    a_won = pairs.merge(games, left_on=["player_a", "player_b"], right_on=["Winner", "Loser"]).assign(a_won=True)
    b_won = pairs.merge(games, left_on=["player_a", "player_b"], right_on=["Loser", "Winner"]).assign(a_won=False)
    candidates = pd.concat([a_won, b_won], ignore_index=True)

    days = (candidates["Date"] - candidates["published"].dt.tz_convert(None).dt.normalize()).dt.days
    candidates = candidates[(days >= 0) & (days <= MAX_DAYS_BEFORE)]
    m = _keep_unique(candidates, "tennis")
    walkover = m["Comment"] == "Walkover"
    m = m[~walkover]
    log.info("tennis: %d walkovers left out", walkover.sum())

    recent = previews[previews["published"].dt.tz_convert(None) >= games["Date"].min()]
    unknown = sorted((set(recent["side_a"]) | set(recent["side_b"])) - set(names["sportsmole_name"]))
    if unknown:
        log.warning("tennis: no tennis-data player for %s (add to PLAYER_ALIASES if only the spelling differs)",
                    ", ".join(unknown))

    def for_a(winner_column: str, loser_column: str) -> pd.Series:
        """Player a's value: from the winner's column if a won, else from the loser's."""
        return m[winner_column].where(m["a_won"], m[loser_column])

    out = pd.DataFrame({
        "article_id": m["article_id"], "published": m["published"],
        "side_a": m["side_a"], "side_b": m["side_b"], "player_a": m["player_a"], "player_b": m["player_b"],
        "pick_winner": m["pick_winner"],
        "pick_side": [_pick_side(pick, a, b) for pick, a, b in zip(m["pick_winner"], m["side_a"], m["side_b"])],
        "pick_sets": m["pick_sets"],
        "date": m["Date"], "tour": m["tour"], "tournament": m["Tournament"], "surface": m["Surface"],
        "round": m["Round"], "best_of": m["Best of"], "status": m["Comment"],
        "a_won": m["a_won"],
        "sets_a": for_a("Wsets", "Lsets"), "sets_b": for_a("Lsets", "Wsets"),
        "rank_a": for_a("WRank", "LRank"), "rank_b": for_a("LRank", "WRank"),
        # tennis-data's odds are closing odds: the average and the best bookmaker
        "closing_odds_a": for_a("AvgW", "AvgL"), "closing_odds_b": for_a("AvgL", "AvgW"),
        "best_closing_odds_a": for_a("MaxW", "MaxL"), "best_closing_odds_b": for_a("MaxL", "MaxW"),
    })
    return out.sort_values(["date", "article_id"], kind="stable").reset_index(drop=True)


def _pick_side(pick: object, side_a: str, side_b: str) -> str | None:
    """Which player the "We say" pick names: "a", "b", or None if it is unclear.

    >>> _pick_side("Alcaraz", "Carlos Alcaraz", "Jannik Sinner")
    'a'
    """
    if not isinstance(pick, str):
        return None
    in_a, in_b = _words(pick) <= _words(side_a), _words(pick) <= _words(side_b)
    if in_a != in_b:
        return "a" if in_a else "b"
    return None


# Pipeline step -----------------------------------------------------------------

def build(sports=config.SPORTS) -> dict[str, pd.DataFrame]:
    """Match picks.csv to the saved odds files and write one table per sport."""
    from . import odds
    from .parse import PICKS_CSV

    previews = pd.read_csv(PICKS_CSV)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    results = {}
    if "football" in sports:
        games = odds.load_football()
        games = games[games["date"] <= config.STUDY_END]
        for name, guess in suggest_aliases(pd.concat([games["HomeTeam"], games["AwayTeam"]]),
                                           pd.concat([previews["side_a"], previews["side_b"]]).dropna()).items():
            log.warning("no Sports Mole name matches %r; closest is %r (add it to ALIASES if right)", name, guess)
        results["football"] = match_football(previews, games)
        results["football"].to_csv(FOOTBALL_CSV, index=False)
    if "tennis" in sports:
        games = odds.load_tennis()
        results["tennis"] = match_tennis(previews, games[games["Date"] <= config.STUDY_END])
        results["tennis"].to_csv(TENNIS_CSV, index=False)
    return results
