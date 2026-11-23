"""The three tests from docs/plan.md, plus a first look at how good the picks are.

`python -m pvb analyse` reads only the tables in data/processed/ (no article text
needed) and writes every table and chart to results/:

0. picks_vs_favourite: how often the pick and the bookmaker favourite are right.
1. text_beyond_odds: does the text say more than the closing odds? A logistic
   regression predicts the result from the odds alone, and from the odds plus
   the pick (plus the text features). Both are fitted on the training seasons,
   then compared game by game on the test season.
2. odds_movement (football only): between the early and the closing odds, did
   the picked outcome become more likely than outcomes nobody picked?
3. backtest: bet whenever a model's probability times the odds promises a profit.

Choices, and why:

- Odds become probabilities as 1/odds, scaled to sum to 1, which removes the
  bookmaker's margin.
- The text models are compared with an odds-only model fitted the same way, not
  with the raw odds. Otherwise they could win only by correcting known biases in
  the odds (such as the favourite-longshot bias), which has nothing to do with text.
- Models learn from the training seasons only and are evaluated once on the test
  season, then once more on 2026/27 so far (config.SPLITS).
- Uncertainty: 95% normal intervals, treating games as independent. Test 1 uses
  per-game differences; tests 2 and 3 cluster by game, because the outcomes
  (or bets) of one game are not independent of each other.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES_CSV
from .matching import FOOTBALL_CSV, TENNIS_CSV

RESULTS = config.ROOT / "results"
OUTCOMES = {"football": ["home", "draw", "away"], "tennis": ["a", "b"]}  # the order of y = 0, 1, 2
PICK_FEATURES = {
    "football": ["pick_home", "pick_draw", "pick_goal_diff"],
    "tennis": ["pick_a", "pick_a_in_straight_sets"],
}
# Columns from features.csv. Text analysis role: add tone_diff and the Naive Bayes probabilities.
# Tennis needs features with a direction (a minus b): a word count for the whole preview
# can't say which of the two players it favours.
TEXT_FEATURES = {"football": ["uncertainty_per_1000", "absence_terms"], "tennis": []}
EDGES = (0.0, 0.05)  # bet when the expected profit per unit staked is above this
MAX_ODDS = 10  # average closing odds above this make a long shot (see `backtest`)
SETS = {"two": 2, "three": 3, "four": 4, "five": 5}


# Loading -----------------------------------------------------------------------

def load(sport: str) -> pd.DataFrame:
    """The matched table for `sport` with the text features, ready for the tests."""
    path = FOOTBALL_CSV if sport == "football" else TENNIS_CSV
    if not path.exists():
        raise FileNotFoundError(f"No {path.name} yet. Run `python -m pvb match` first.")
    games = pd.read_csv(path, dtype={"season": str}, parse_dates=["kickoff", "published"] if sport == "football" else ["date"])
    if TEXT_FEATURES[sport]:
        features = pd.read_csv(FEATURES_CSV, usecols=["article_id", *TEXT_FEATURES[sport]])
        games = games.merge(features, on="article_id", how="left", validate="one_to_one")
    return prepare_football(games) if sport == "football" else prepare_tennis(games)


def prepare_football(games: pd.DataFrame) -> pd.DataFrame:
    """Code the result and the pick as numbers. Previews without a pick are dropped."""
    games = games.dropna(subset=["pick_result"]).reset_index(drop=True)
    code = {"H": 0, "D": 1, "A": 2}
    games["y"], games["pick"] = games["result"].map(code), games["pick_result"].map(code)
    games["pick_home"] = (games["pick_result"] == "H").astype(int)
    games["pick_draw"] = (games["pick_result"] == "D").astype(int)
    games["pick_goal_diff"] = games["pick_home_goals"] - games["pick_away_goals"]  # 3-0 is surer than 1-0
    return add_sample(games, "football")


def prepare_tennis(games: pd.DataFrame) -> pd.DataFrame:
    games = games.dropna(subset=["pick_side"]).reset_index(drop=True)
    games["season"] = games["date"].dt.year.astype(str)
    games["y"] = np.where(games["a_won"], 0, 1)
    games["pick"] = np.where(games["pick_side"] == "a", 0, 1)
    games["pick_a"] = 1 - games["pick"]
    # "in straight sets" is the surest pick: two sets in best of 3, three in best of 5
    straight = (games["pick_sets"] == "straight") | (games["pick_sets"].map(SETS) == games["best_of"] // 2 + 1)
    games["pick_a_in_straight_sets"] = straight.astype(int) * np.where(games["pick_a"] == 1, 1, -1)
    return add_sample(games, "tennis")


def add_sample(games: pd.DataFrame, sport: str) -> pd.DataFrame:
    """train, test or check, from config.SPLITS."""
    sample_of = {season: sample for sample, seasons in config.SPLITS[sport].items() for season in seasons}
    games["sample"] = games["season"].map(sample_of)
    missing = [c for c in PICK_FEATURES[sport] + TEXT_FEATURES[sport] if games[c].isna().any()]
    if missing:
        raise ValueError(f"{sport}: missing values in {missing}; re-run `python -m pvb features` and `match`")
    return games


def odds(games: pd.DataFrame, sport: str, kind: str = "closing_odds") -> np.ndarray:
    """One row per game, one column per outcome. kind: closing_odds, best_closing_odds or early_odds."""
    return games[[f"{kind}_{outcome}" for outcome in OUTCOMES[sport]]].to_numpy(float)


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """1/odds, scaled so each row sums to 1. This removes the bookmaker's margin."""
    raw = 1 / odds
    return raw / raw.sum(axis=1, keepdims=True)


def season_label(season: str, sport: str) -> str:
    return f"20{season[:2]}/{season[2:]}" if sport == "football" else season


# Scores ------------------------------------------------------------------------

def log_loss(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per game: minus the log of the probability given to what happened. Lower is better."""
    return -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1))


def brier(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per game: squared distance between the probabilities and what happened. Lower is better."""
    return ((probabilities - np.eye(probabilities.shape[1])[y]) ** 2).sum(axis=1)


def mean_with_ci(values) -> dict:
    """Mean, 95% interval and two-sided p-value (is the mean different from 0?)."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return {"mean": values.mean() if len(values) else np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    p_value = 2 * (1 - NormalDist().cdf(abs(mean / se))) if se > 0 else np.nan
    return {"mean": mean, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se, "p_value": p_value}


# 0. How good are the picks ------------------------------------------------------

def picks_vs_favourite(games: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Per season: how often the pick and the favourite are right, and what backing every pick returns."""
    rows = np.arange(len(games))
    y, pick = games["y"].to_numpy(), games["pick"].to_numpy()
    average, best = odds(games, sport), odds(games, sport, "best_closing_odds")
    favourite = average.argmin(axis=1)
    per_game = pd.DataFrame({
        "pick_right": pick == y,
        "favourite_right": favourite == y,
        "pick_is_favourite": pick == favourite,
        "roi_pick_average_odds": np.where(pick == y, average[rows, pick] - 1, -1),
        "roi_pick_best_odds": np.where(pick == y, best[rows, pick] - 1, -1),
    })
    if sport == "football":
        per_game["draw_picks"], per_game["draws"] = pick == 1, y == 1
    seasons = games["season"].map(lambda s: season_label(s, sport)).to_numpy()  # per_game has its own index
    table = pd.concat([per_game.groupby(seasons).mean(), per_game.mean().to_frame("all").T])
    table.insert(0, "games", [*per_game.groupby(seasons).size(), len(per_game)])
    table = table.rename_axis("season").reset_index()
    table.insert(0, "sport", sport)
    return table


# 1. Text beyond odds ------------------------------------------------------------

def models(sport: str) -> dict[str, list[str]]:
    """Model name -> the columns it uses on top of the odds."""
    found = {"odds": [], "odds + pick": PICK_FEATURES[sport]}
    if TEXT_FEATURES[sport]:
        found["odds + pick + text"] = PICK_FEATURES[sport] + TEXT_FEATURES[sport]
    return found


def fit_and_predict(games: pd.DataFrame, sport: str, columns: list[str]) -> np.ndarray:
    """Fit on the training seasons only, then give each game's outcome probabilities.

    The inputs are the log market probabilities plus `columns`. The regression
    has no penalty, so it is the plain (multinomial) logit from econometrics.
    It is solved to a tight tolerance, so the result doesn't depend on where the
    solver stops; the scaler only helps the solver and doesn't change the predictions.
    """
    X = np.column_stack([np.log(implied_probabilities(odds(games, sport))), games[columns].to_numpy(float)])
    train = (games["sample"] == "train").to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000))
    model.fit(X[train], games["y"].to_numpy()[train])
    return model.predict_proba(X)


def predictions(games: pd.DataFrame, sport: str) -> dict[str, np.ndarray]:
    """'market' is the odds as they are; the others are fitted (see `models`)."""
    found = {"market": implied_probabilities(odds(games, sport))}
    for name, columns in models(sport).items():
        found[name] = fit_and_predict(games, sport, columns)
    return found


def text_beyond_odds(games: pd.DataFrame, sport: str, predicted: dict[str, np.ndarray]) -> pd.DataFrame:
    """Scores per model on each evaluation sample.

    change_vs_odds is the mean per-game log loss minus that of the odds-only
    model on the same games: below 0 means the model knows something the odds don't.
    """
    rows = []
    for sample, seasons in config.SPLITS[sport].items():
        here = (games["sample"] == sample).to_numpy()
        if sample == "train" or not here.any():
            continue
        y = games["y"].to_numpy()[here]
        baseline = log_loss(predicted["odds"][here], y)
        for name, probabilities in predicted.items():
            p = probabilities[here]
            loss = log_loss(p, y)
            change = mean_with_ci(loss - baseline) if name != "odds" else {}
            rows.append({
                "sport": sport, "sample": sample, "seasons": ", ".join(season_label(s, sport) for s in seasons),
                "model": name, "games": here.sum(),
                "log_loss": loss.mean(), "brier": brier(p, y).mean(), "accuracy": (p.argmax(axis=1) == y).mean(),
                "change_vs_odds": change.get("mean", np.nan), "ci_low": change.get("ci_low", np.nan),
                "ci_high": change.get("ci_high", np.nan), "p_value": change.get("p_value", np.nan),
            })
    return pd.DataFrame(rows)


# 2. Odds movement ---------------------------------------------------------------

def odds_movement(games: pd.DataFrame, bands: int = 10) -> pd.DataFrame:
    """Football: did the picked outcome gain probability between the early and the closing odds?

    Favourites may shorten anyway, and the pick is usually the favourite. So the
    picked outcomes are compared with unpicked outcomes of the same kind (home,
    draw or away) and a similar early probability (the same tenth): see
    `picked_effect`. excess_pp is that difference in percentage points; above 0
    means the odds moved towards the pundit. With 5, 20 or 50 bands instead of
    10 the estimate barely changes.

    The games are also split by whether the preview came out before the early
    odds were taken (`early_odds_time`). If it came out after, the move includes
    the market's reaction to the preview itself.
    """
    games = games.dropna(subset=[f"early_odds_{o}" for o in OUTCOMES["football"]]).reset_index(drop=True)
    early, closing = implied_probabilities(odds(games, "football", "early_odds")), implied_probabilities(odds(games, "football"))
    n = len(games)
    outcomes = pd.DataFrame({
        "game": np.repeat(np.arange(n), 3),
        "outcome": np.tile([0, 1, 2], n),
        "early": early.ravel(),
        "move": 100 * (closing - early).ravel(),
        "picked": (np.eye(3)[games["pick"]] == 1).ravel(),
    })
    outcomes["band"] = outcomes.groupby("outcome")["early"].transform(lambda p: pd.qcut(p, bands, labels=False))

    before = (games["published"] < early_odds_time(games["kickoff"])).to_numpy()
    groups = {
        "All games": np.full(n, True),
        "Preview before the early odds": before,
        "Preview after the early odds": ~before,
    }
    rows = []
    for name, keep in groups.items():
        if not keep.any():
            continue
        part = outcomes[np.repeat(keep, 3)]
        moves = part.loc[part["picked"], "move"]
        effect = picked_effect(part)
        rows.append({
            "games": name, "n": keep.sum(), "pick_move_pp": moves.mean(),
            "excess_pp": effect["mean"], "ci_low": effect["ci_low"], "ci_high": effect["ci_high"], "p_value": effect["p_value"],
            "pick_gained": (moves > 0).mean(), "pick_lost": (moves < 0).mean(),
        })
    return pd.DataFrame(rows)


def picked_effect(outcomes: pd.DataFrame) -> dict:
    """Regression of the move on `picked`, with one constant per outcome kind and band.

    This compares picked and unpicked outcomes within each band and averages the
    differences, giving more weight to bands with plenty of both. The standard
    error is clustered by game: a game's three probabilities add up to 1, so
    they move together.
    """
    cell = outcomes.groupby(["outcome", "band"]).ngroup()
    x = outcomes["picked"].astype(float)
    x = x - x.groupby(cell).transform("mean")
    y = outcomes["move"] - outcomes["move"].groupby(cell).transform("mean")
    if (x ** 2).sum() == 0:  # no band has both picked and unpicked outcomes
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    estimate = (x * y).sum() / (x ** 2).sum()
    per_game = (x * (y - estimate * x)).groupby(outcomes["game"]).sum()
    clusters = len(per_game)
    se = np.sqrt((per_game ** 2).sum() * clusters / (clusters - 1)) / (x ** 2).sum()
    p_value = 2 * (1 - NormalDist().cdf(abs(estimate / se))) if se > 0 else np.nan
    return {"mean": estimate, "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se, "p_value": p_value}


def early_odds_time(kickoff: pd.Series) -> pd.Series:
    """Roughly when football-data took the early odds, in UTC.

    Their notes say Friday afternoon for weekend games and Tuesday afternoon for
    midweek games. We take Friday for Friday to Monday games, Tuesday for Tuesday
    to Thursday games, and 15:00 UK time for "afternoon" (a guess).
    """
    local = kickoff.dt.tz_convert("Europe/London").dt.tz_localize(None)
    weekday = local.dt.day_of_week.to_numpy()  # Monday = 0
    days_back = np.where(np.isin(weekday, [1, 2, 3]), weekday - 1, (weekday - 4) % 7)
    day = local.dt.normalize() - pd.to_timedelta(days_back, unit="D").to_numpy()
    return (day + pd.Timedelta(hours=15)).dt.tz_localize("Europe/London").dt.tz_convert("UTC")


# 3. Betting backtest ------------------------------------------------------------

def backtest(games: pd.DataFrame, sport: str, predicted: dict[str, np.ndarray]) -> pd.DataFrame:
    """Stake 1 on each bet, on the evaluation samples only.

    Two simple strategies come first for comparison. A model bets on an outcome
    when probability x odds - 1 (the expected profit per unit) is above the edge.
    The odds-only model shows what fitting the odds earns without any text.

    Average closing odds are what a bettor could get; the best closing odds (the
    highest quote of any bookmaker, stale ones included) are an optimistic upper
    bound. Long shots (average closing odds above MAX_ODDS) are left out of the
    main rows, because their best quotes are the least reliable (51 where the
    average is 23.5) and a few of them can decide the result. The rows with
    long_shots = "included" show what happens with them.
    """
    rows = []
    for sample in config.SPLITS[sport]:
        here = (games["sample"] == sample).to_numpy()
        if sample == "train" or not here.any():
            continue
        part = games[here]
        won = np.eye(len(OUTCOMES[sport]), dtype=bool)[part["y"].to_numpy()]
        long_shot = odds(part, sport) > MAX_ODDS
        for odds_used, kind in [("average", "closing_odds"), ("best", "best_closing_odds")]:
            price = odds(part, sport, kind)
            strategies = {
                "back every pick": np.eye(price.shape[1], dtype=bool)[part["pick"].to_numpy()],
                "back every favourite": np.eye(price.shape[1], dtype=bool)[odds(part, sport).argmin(axis=1)],
            }
            for name in models(sport):
                for edge in EDGES:
                    strategies[f"{name} model, edge > {edge:.0%}"] = predicted[name][here] * price - 1 > edge
            for long_shots in ["left out", "included"]:
                for strategy, bets in strategies.items():
                    if long_shots == "left out":
                        bets = bets & ~long_shot
                    profit = np.where(won, price - 1, -1) * bets  # per game and outcome, 0 without a bet
                    roi = roi_with_ci(profit.sum(axis=1), bets.sum(axis=1))
                    rows.append({
                        "sport": sport, "sample": sample, "odds": odds_used, "long_shots": long_shots,
                        "strategy": strategy, "bets": bets.sum(), "hit_rate": won[bets].mean() if bets.any() else np.nan,
                        "profit": profit.sum(), **roi,
                    })
    return pd.DataFrame(rows)


def roi_with_ci(profit_per_game: np.ndarray, bets_per_game: np.ndarray) -> dict:
    """Profit per unit staked, with a 95% interval clustered by game (a game can carry two bets).

    With one bet per game this is the usual interval for a mean.
    """
    keep = bets_per_game > 0
    profit, bets = profit_per_game[keep], bets_per_game[keep]
    if len(bets) == 0:
        return {"roi": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    roi = profit.sum() / bets.sum()
    if len(bets) < 2:
        return {"roi": roi, "ci_low": np.nan, "ci_high": np.nan}
    se = np.sqrt(((profit - roi * bets) ** 2).sum() * len(bets) / (len(bets) - 1)) / bets.sum()
    return {"roi": roi, "ci_low": roi - 1.96 * se, "ci_high": roi + 1.96 * se}


# Running it all -----------------------------------------------------------------

def make_tables(football: pd.DataFrame, tennis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predicted = {"football": predictions(football, "football"), "tennis": predictions(tennis, "tennis")}
    both = {"football": football, "tennis": tennis}
    return {
        "picks_vs_favourite": pd.concat([picks_vs_favourite(g, s) for s, g in both.items()], ignore_index=True),
        "text_beyond_odds": pd.concat([text_beyond_odds(g, s, predicted[s]) for s, g in both.items()], ignore_index=True),
        "odds_movement": odds_movement(football),
        "backtest": pd.concat([backtest(g, s, predicted[s]) for s, g in both.items()], ignore_index=True),
    }


def run(results: Path = RESULTS) -> dict[str, pd.DataFrame]:
    """Load data/processed/, run every test, write the tables and charts to `results`."""
    tables = make_tables(load("football"), load("tennis"))
    results.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(6).to_csv(results / f"{name}.csv", index=False)
    chart_picks_vs_favourite(tables["picks_vs_favourite"], results / "picks_vs_favourite.png")
    chart_text_beyond_odds(tables["text_beyond_odds"], results / "text_beyond_odds.png")
    chart_odds_movement(tables["odds_movement"], results / "odds_movement.png")
    return tables


# Charts (static PNGs for the slides) --------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def new_chart(width: float, height: float, title: str):
    figure = Figure(figsize=(width, height), facecolor="white")
    ax = figure.subplots()
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.set_axisbelow(True)
    return figure, ax


def chart_picks_vs_favourite(table: pd.DataFrame, path: Path) -> None:
    seasons = table[(table["sport"] == "football") & (table["season"] != "all")]
    x = np.arange(len(seasons))
    figure, ax = new_chart(8, 4, "Football: how often the pick is right")
    for offset, column, colour, label in [(-0.18, "pick_right", BLUE, "Sports Mole's pick"),
                                          (0.18, "favourite_right", ORANGE, "Bookmaker favourite")]:
        bars = ax.bar(x + offset, 100 * seasons[column], width=0.34, color=colour, label=label)
        ax.bar_label(bars, fmt="%.0f%%", padding=3, color=MUTED, fontsize=8)
    ax.set_xticks(x, seasons["season"])
    ax.set_ylim(0, 65)
    ax.yaxis.grid(color=GRID)
    ax.set_ylabel("Share of games, %", color=MUTED)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    figure.savefig(path, dpi=200, bbox_inches="tight")


def forest_chart(labels, means, lows, highs, title: str, xlabel: str, path: Path) -> None:
    """One dot with its 95% interval per row, and a line at 0 (no effect)."""
    y = np.arange(len(labels))[::-1]
    figure, ax = new_chart(8, 0.45 * len(labels) + 1.6, title)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.hlines(y, lows, highs, color=BLUE, linewidth=2)
    ax.plot(means, y, "o", color=BLUE, markersize=7)
    ax.set_yticks(y, labels)
    ax.xaxis.grid(color=GRID)
    ax.set_xlabel(xlabel, color=MUTED)
    figure.savefig(path, dpi=200, bbox_inches="tight")


def chart_text_beyond_odds(table: pd.DataFrame, path: Path) -> None:
    rows = table[~table["model"].isin(["odds", "market"])]
    labels = rows["sport"].str.capitalize() + " " + rows["seasons"] + " (" + rows["sample"] + "): " + rows["model"]
    forest_chart(labels, rows["change_vs_odds"], rows["ci_low"], rows["ci_high"],
                 "Does the text improve on the odds?",
                 "Change in log loss against the odds-only model (below 0 = the text helps)", path)


def chart_odds_movement(table: pd.DataFrame, path: Path) -> None:
    forest_chart(table["games"], table["excess_pp"], table["ci_low"], table["ci_high"],
                 "Football: did the odds move towards the pick?",
                 "Extra change in the pick's probability, percentage points (above 0 = towards the pick)", path)
