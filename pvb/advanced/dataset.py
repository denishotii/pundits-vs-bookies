"""Load football inputs for a single market, with publication-time eligibility.

Publication dates do not establish which version of the text the odds market
could have seen. Article modification history is unavailable and remains an
unresolved source of leakage for both markets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .. import config
from ..analysis import early_odds_time

Market = Literal["early", "closing"]
OUTCOME_CODES = {"H": 0, "D": 1, "A": 2}
ODDS_COLUMNS = {
    "early": ("early_odds_home", "early_odds_draw", "early_odds_away"),
    "closing": ("closing_odds_home", "closing_odds_draw", "closing_odds_away"),
}
TEXT_COLUMNS = (
    "title", "text_intro", "text_preview", "text_team_news", "text_head_to_head",
    "text_verdict", "text_other",
)
FIXTURE_COLUMNS = ("season", "league", "home_team", "away_team", "kickoff")
PICK_SCORES = ("pick_home_goals", "pick_away_goals")
DAY = pd.Timedelta(days=1)
MATCH_DURATION_BUFFER = pd.Timedelta(hours=3)


@dataclass
class FootballDataset:
    """Aligned rows for one market, with each component using the same index."""

    target: pd.Series
    metadata: pd.DataFrame
    text: pd.DataFrame
    numerical: pd.DataFrame
    counts: dict[str, int]


def _read_source(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    source = pd.read_csv(path, usecols=lambda col: col in columns, dtype="string")
    missing = sorted(set(columns) - set(source.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    ids = source["article_id"].str.strip()
    missing_ids = ids.isna() | ids.eq("")
    if missing_ids.any():
        raise ValueError(f"{path.name}: missing article_id on {missing_ids.sum()} rows")
    duplicates = ids.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(f"{path.name}: duplicate article_id on {duplicates.sum()} rows")
    return source.assign(article_id=ids)


def _utc(series: pd.Series) -> pd.Series:
    # The project stores UTC dates and timestamps. An explicit offset takes precedence.
    return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")


def load_football(
    market: Market,
    *,
    matches_path: Path = config.PROCESSED / "matched_football.csv",
    previews_path: Path = config.INTERIM / "previews.csv",
) -> FootballDataset:
    """Load records eligible at one market's prediction time.

    Early odds reuse the baseline's approximate 15:00 UK snapshot on Tuesday
    (Tue–Thu fixtures) or Friday (Fri–Mon fixtures). Closing odds use kickoff
    as a proxy for the unsupplied closing timestamp. Midnight UTC values are
    treated as date-only. For these publication dates, require an earlier UTC
    day than the cutoff. For date-only kickoffs, use the earliest possible
    kickoff for eligibility and the start of the next day plus three hours
    as a proxy for outcome availability. Exact kickoff times use a three-hour
    buffer. Exact result release times are unavailable.

    Malformed schemas or ambiguous IDs raise. Row-level exclusion reasons
    can overlap. The excluded_rows total counts each dropped row once.
    """
    if market not in ODDS_COLUMNS:
        raise ValueError(f"Unknown market {market!r}; use 'early' or 'closing'")

    odds_columns = ODDS_COLUMNS[market]
    matches = _read_source(
        matches_path,
        ("article_id", *FIXTURE_COLUMNS, "published", "result", "pick_result", *PICK_SCORES, *odds_columns),
    ).rename(columns={"published": "matched_published"})
    previews = _read_source(previews_path, ("article_id", "sport", "published", *TEXT_COLUMNS))
    games = matches.merge(previews, on="article_id", how="left", validate="one_to_one", indicator=True)

    kickoff = _utc(games["kickoff"])
    published = _utc(games["published"])
    matched_published = _utc(games["matched_published"])
    has_article = games["_merge"].eq("both")
    fixture_names = games[["league", "home_team", "away_team"]].apply(lambda col: col.str.strip())
    missing_fixture = (fixture_names.isna() | fixture_names.eq("")).any(axis=1)
    fixture = fixture_names.assign(season=games["season"], kickoff=kickoff)
    identified = fixture.loc[~missing_fixture].dropna()
    duplicate_fixtures = identified.duplicated(keep=False)
    if duplicate_fixtures.any():
        raise ValueError(f"{matches_path.name}: duplicate football fixture on {duplicate_fixtures.sum()} rows")

    date_only_kickoff = kickoff.eq(kickoff.dt.normalize())
    date_only_publication = published.eq(published.dt.normalize())
    cutoff = kickoff.copy()
    if market == "early":
        cutoff.loc[kickoff.notna()] = early_odds_time(kickoff.dropna())
    publication_cutoff = cutoff.where(~date_only_publication, cutoff.dt.normalize())
    outcome_available_at = kickoff.where(~date_only_kickoff, kickoff + DAY) + MATCH_DURATION_BUFFER

    raw_odds = games[list(odds_columns)]
    odds = raw_odds.apply(pd.to_numeric, errors="coerce").astype(float)
    scores = games[list(PICK_SCORES)].apply(pd.to_numeric, errors="coerce").astype(float)
    valid_scores = np.isfinite(scores) & scores.ge(0) & scores.mod(1).eq(0)
    body = games[list(TEXT_COLUMNS[1:])]
    has_body = body.apply(lambda col: col.str.strip().fillna("").ne("")).any(axis=1)
    reasons = pd.DataFrame({
        "missing_article": ~has_article,
        "wrong_article_sport": has_article & ~games["sport"].fillna("").eq("football"),
        "missing_article_body": has_article & ~has_body,
        "invalid_season": ~games["season"].isin(config.FOOTBALL_SEASONS),
        "missing_fixture_metadata": missing_fixture,
        "invalid_result": ~games["result"].isin(OUTCOME_CODES),
        "missing_pick": games["pick_result"].isna(),
        "invalid_pick": games["pick_result"].notna() & ~games["pick_result"].isin(OUTCOME_CODES),
        "invalid_pick_score": games["pick_result"].isin(OUTCOME_CODES) & ~valid_scores.all(axis=1),
        "missing_odds": raw_odds.isna().any(axis=1),
        "invalid_odds": (raw_odds.notna() & ~(np.isfinite(odds) & odds.gt(1))).any(axis=1),
        "invalid_kickoff": kickoff.isna(),
        "invalid_match_publication": matched_published.isna(),
        "invalid_article_publication": has_article & published.isna(),
        "publication_mismatch": published.notna() & matched_published.notna() & published.ne(matched_published),
        "early_snapshot_not_before_kickoff": (market == "early") & cutoff.ge(kickoff),
        "article_not_before_cutoff": published.ge(publication_cutoff),
    })
    games = games.assign(
        published=published, kickoff=kickoff, prediction_at=cutoff, outcome_available_at=outcome_available_at,
    )
    eligible = games.loc[~reasons.any(axis=1)].sort_values(["kickoff", "article_id"]).index
    numerical = odds.loc[eligible].assign(
        pick_home=games.loc[eligible, "pick_result"].eq("H").astype(int),
        pick_draw=games.loc[eligible, "pick_result"].eq("D").astype(int),
        pick_goal_diff=scores.loc[eligible, "pick_home_goals"] - scores.loc[eligible, "pick_away_goals"],
    )
    return FootballDataset(
        target=games.loc[eligible, "result"].map(OUTCOME_CODES).astype(int).rename("y").reset_index(drop=True),
        metadata=games.loc[eligible, [
            "article_id", *FIXTURE_COLUMNS, "published", "prediction_at", "outcome_available_at",
        ]].reset_index(drop=True),
        text=games.loc[eligible, list(TEXT_COLUMNS)].fillna("").reset_index(drop=True),
        numerical=numerical.reset_index(drop=True),
        counts={
            "input_rows": len(games), "eligible_rows": len(eligible), "excluded_rows": len(games) - len(eligible),
            **reasons.sum().astype(int).to_dict(),
        },
    )
