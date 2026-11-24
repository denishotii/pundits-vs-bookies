"""Build development-only historical football features at market prediction times."""

from bisect import insort
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import config
from .dataset import DAY, MATCH_DURATION_BUFFER, _utc
from .diagnostic_data import DEVELOPMENT

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 60.0
ELO_K = 20.0
ELO_SCALE = 400.0
WINDOWS = (5, 10)
FIXTURE_KEY = ("season", "league", "home_team", "away_team")
GOALS = ("home_goals", "away_goals")
HISTORY_COLUMNS = (*FIXTURE_KEY, "kickoff", *GOALS)
RAW_ODDS_DIR = config.ODDS_RAW / "football"
ROLLING_STATS = ("points", "goals_for", "goals_against", "goal_difference")
SIDE_FEATURES = (
    "elo", "observed_matches", "observed_venue_matches", "venue_points_per_match", "days_since_known_match",
    *(f"{stat}_last_{window}" for window in WINDOWS for stat in ROLLING_STATS),
)
HISTORY_FEATURES = (*(f"{side}_{name}" for side in ("home", "away") for name in SIDE_FEATURES), "elo_difference")
WITH_MISSINGNESS = tuple(name for name in HISTORY_FEATURES
                         if name not in ("home_elo", "away_elo", "elo_difference") and "_observed_" not in name)


def _raw_results(path, season, league):
    columns = {"Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    table = pd.read_csv(path, usecols=lambda name: name in columns, dtype="string",
                        encoding="utf-8-sig", encoding_errors="replace")
    dates = pd.to_datetime(table["Date"], dayfirst=True, format="mixed", errors="coerce")
    time = table.get("Time", pd.Series("", index=table.index, dtype="string")).fillna("")
    exact = pd.to_datetime(table["Date"] + " " + time, dayfirst=True, format="mixed", errors="coerce")
    kickoff = exact.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    kickoff = kickoff.where(time.str.strip().ne(""), dates.dt.tz_localize("UTC"))
    return table.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team",
                                 "FTHG": "home_goals", "FTAG": "away_goals"}).assign(
        season=season, league=league, kickoff=kickoff, source="football_data",
    ).loc[:, [*HISTORY_COLUMNS, "source"]]


def _observed(table):
    table = table.copy()
    for name in FIXTURE_KEY:
        table[name] = table[name].str.strip()
    goals = table[list(GOALS)].apply(pd.to_numeric, errors="coerce").astype(float)
    valid = (np.isfinite(goals) & goals.ge(0) & goals.mod(1).eq(0)).all(axis=1)
    valid &= table["kickoff"].notna() & table[list(FIXTURE_KEY)].fillna("").ne("").all(axis=1)
    valid &= table["home_team"].ne(table["away_team"]).fillna(False)
    table = table.loc[valid].assign(**goals.loc[valid].astype(int))
    table = table.drop_duplicates()
    if table.duplicated(list(FIXTURE_KEY)).any():
        raise ValueError("Expected one observed league fixture per ordered pair and season")
    return table


def load_history(matches_path: Path, raw_dir: Path = RAW_ODDS_DIR):
    """Prefer valid local football-data rows, supplemented by matched-only results."""
    matched = pd.read_csv(matches_path, usecols=list(HISTORY_COLUMNS), dtype="string")
    matched = matched.loc[matched["season"].isin(DEVELOPMENT) & matched["league"].isin(config.LEAGUES)]
    matched = _observed(matched.assign(kickoff=_utc(matched["kickoff"]), source="matched_export"))
    raw = []
    for season in DEVELOPMENT:
        for league in config.LEAGUES:
            path = raw_dir / f"{season}_{league}.csv"
            if path.is_file():
                raw.append(_observed(_raw_results(path, season, league)))
    history = matched
    if raw:
        primary = pd.concat(raw, ignore_index=True).set_index(list(FIXTURE_KEY))
        fallback = matched.set_index(list(FIXTURE_KEY))
        overlap = primary.index.intersection(fallback.index)
        if not primary.loc[overlap, list(GOALS)].equals(fallback.loc[overlap, list(GOALS)]):
            raise ValueError("Local result sources disagree on observed scores")
        raw_times = primary.loc[overlap, "kickoff"]
        matched_times = fallback.loc[overlap, "kickoff"]
        raw_days, matched_days = raw_times.dt.normalize(), matched_times.dt.normalize()
        raw_date_only = raw_times.eq(raw_days)
        matched_date_only = matched_times.eq(matched_days)
        compatible = raw_times.eq(matched_times) | (
            (raw_date_only | matched_date_only) & raw_days.eq(matched_days)
        )
        if not compatible.all():
            raise ValueError("Local result sources disagree on observed kickoffs")
        use_matched = overlap[raw_date_only & ~matched_date_only]
        primary.loc[use_matched, "kickoff"] = matched_times.loc[use_matched]
        history = pd.concat([primary, fallback.loc[~fallback.index.isin(primary.index)]]).reset_index()

    kickoff = history["kickoff"]
    date_only = kickoff.eq(kickoff.dt.normalize())
    history = history.assign(
        date_only=date_only,
        result_available_at=kickoff + MATCH_DURATION_BUFFER + date_only.astype(int) * DAY,
    )
    return history.reset_index(drop=True)


def source_coverage(history):
    rows = []
    for (season, league), frame in history.groupby(["season", "league"], sort=True):
        teams = len(set(frame["home_team"]) | set(frame["away_team"]))
        rows.append({
            "season": season, "league": league, "observed_fixtures": len(frame),
            "matched_export": int(frame["source"].eq("matched_export").sum()),
            "football_data": int(frame["source"].eq("football_data").sum()),
            "date_only_kickoff": int(frame["date_only"].sum()),
            "observed_teams": teams, "double_round_robin_given_observed_teams": teams * (teams - 1),
        })
    return pd.DataFrame(rows)


def elo_change(home_rating, away_rating, home_goals, away_goals):
    """Home Elo increase from a match. Apply its negative to the away club."""
    expected = 1 / (1 + 10 ** ((away_rating - home_rating - HOME_ADVANTAGE) / ELO_SCALE))
    achieved = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5
    return ELO_K * (achieved - expected)


def _club(league, name):
    # The two English divisions use the same provider names. Carry clubs between them.
    return ("England" if league in ("E0", "E1") else league, name)


@dataclass
class ClubHistory:
    elo: float = INITIAL_ELO
    matches: list = field(default_factory=list)
    venue_points: list = field(default_factory=lambda: [0, 0])
    venue_matches: list = field(default_factory=lambda: [0, 0])

    def record(self, kickoff, key, venue, goals_for, goals_against):
        points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
        insort(self.matches, (kickoff, key, points, goals_for, goals_against))
        self.venue_points[venue] += points
        self.venue_matches[venue] += 1

    def features(self, cutoff, venue):
        count = self.venue_matches[venue]
        rest = (cutoff - self.matches[-1][0]) / DAY if self.matches else np.nan
        values = [self.elo, len(self.matches), count, self.venue_points[venue] / count if count else np.nan, rest]
        for window in WINDOWS:
            if self.matches:
                points, goals_for, goals_against = np.mean([match[2:] for match in self.matches[-window:]], axis=0)
                values.extend((points, goals_for, goals_against, goals_for - goals_against))
            else:
                values.extend((np.nan,) * len(ROLLING_STATS))
        return values


def historical_features(fixtures, history):
    """Replay completed results; each row sees only results available strictly before its cutoff."""
    index = fixtures.index
    if not index.is_unique or index.to_frame(index=False).isna().any().any():
        raise ValueError("Prediction rows must have unique match IDs")
    if not fixtures["season"].isin(DEVELOPMENT).all():
        raise ValueError("Historical features accept development predictions only")
    required = [*FIXTURE_KEY, "kickoff", "prediction_at"]
    if fixtures[required].isna().any().any() or fixtures["prediction_at"].gt(fixtures["kickoff"]).any():
        raise ValueError("Predictions require known fixture keys and cutoffs at or before kickoff")

    history = history.loc[history["season"].isin(DEVELOPMENT), [*HISTORY_COLUMNS, "result_available_at"]]
    if (history.isna().any().any() or history["result_available_at"].le(history["kickoff"]).any()
            or history.duplicated(list(FIXTURE_KEY)).any()):
        raise ValueError("Results require unique fixture keys and availability strictly after kickoff")
    by_fixture = history.set_index(list(FIXTURE_KEY))["result_available_at"]
    current = by_fixture.reindex(pd.MultiIndex.from_frame(fixtures[list(FIXTURE_KEY)]))
    # Also guard against a mismatched kickoff across data sources for the same fixture.
    if current.set_axis(fixtures.index).lt(fixtures["prediction_at"]).any():
        raise ValueError("The current fixture result is already marked available at prediction time")

    states = defaultdict(ClubHistory)
    events = history.sort_values(["result_available_at", *FIXTURE_KEY]).itertuples(index=False)
    batches = iter(groupby(events, key=lambda row: row.result_available_at))
    batch = next(batches, None)
    rows = {}
    fixtures = fixtures[required].sort_values("prediction_at", kind="stable")
    for fixture in fixtures.itertuples():
        cutoff = fixture.prediction_at
        while batch is not None and batch[0] < cutoff:
            deltas = defaultdict(float)
            for event in batch[1]:
                home = _club(event.league, event.home_team)
                away = _club(event.league, event.away_team)
                change = elo_change(states[home].elo, states[away].elo, event.home_goals, event.away_goals)
                deltas[home] += change
                deltas[away] -= change
                key = (event.season, event.league, event.home_team, event.away_team)
                states[home].record(event.kickoff, key, 0, event.home_goals, event.away_goals)
                states[away].record(event.kickoff, key, 1, event.away_goals, event.home_goals)
            for club, change in deltas.items():
                states[club].elo += change
            batch = next(batches, None)
        home = states[_club(fixture.league, fixture.home_team)]
        away = states[_club(fixture.league, fixture.away_team)]
        rows[fixture.Index] = [*home.features(cutoff, 0), *away.features(cutoff, 1), home.elo - away.elo]
    return pd.DataFrame([rows[key] for key in index], index=index, columns=HISTORY_FEATURES)


def historical_fold(features, train, valid):
    """Median-impute and scale historical values on training rows, then append raw missing flags."""
    values = features.loc[:, list(HISTORY_FEATURES)]
    flags = values.loc[:, list(WITH_MISSINGNESS)].isna().to_numpy(dtype=float)
    fill = values.iloc[train].median().fillna(0)
    training, validation = (values.iloc[positions].fillna(fill) for positions in (train, valid))
    scaler = StandardScaler()
    return (np.column_stack((scaler.fit_transform(training), flags[train])),
            np.column_stack((scaler.transform(validation), flags[valid])))
