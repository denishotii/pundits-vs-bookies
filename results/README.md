# Results

Made by `python -m pvb analyse` (code in `pvb/analysis.py`) from the tables in `data/processed/` alone. Don't edit these files by hand; run the command again. A test fails if they don't match a fresh run.

The models learn from football 2021/22 to 2024/25 and tennis 2024 to 2025. They are tested once on football 2025/26 and tennis 2026 (`test`). The same football model is then checked on 2026/27 up to 14 November 2026 (`check`).

Every interval is a 95% interval. A p-value tests whether the mean is 0.

## `picks_vs_favourite.csv` and `.png`

One row per sport and season, plus `all`.

- `pick_right`, `favourite_right`: the share of games where Sports Mole's pick, or the bookmaker favourite (the lowest average closing odds), was right.
- `pick_is_favourite`: the share of games where the pick was the favourite.
- `roi_pick_average_odds`, `roi_pick_best_odds`: the profit per unit staked from backing every pick. −0.05 means losing 5 cents per euro.
- `draw_picks`, `draws` (football): the share of draws picked, and the share of games that ended in a draw.

## `text_beyond_odds.csv` and `.png` (test 1)

Each model predicts the probability of each result (home, draw, away; or player a, player b):

- `market`: the closing odds, turned into probabilities (1/odds, scaled to add up to 1). Nothing is fitted.
- `odds`: a logistic regression on the odds alone. It can correct biases in the odds, such as long shots winning less often than their odds say.
- `odds + pick`: also uses the pick: which result, and the goal difference of the predicted score (football) or whether it says straight sets (tennis).
- `odds + pick + text`: also uses the text features from `features.csv`. These are starter features for now (uncertainty words, absence words).

`log_loss` and `brier` measure how far the probabilities were from what happened, so lower is better. `change_vs_odds` is the model's log loss minus the `odds` model's, on the same games. **Below 0, with the whole interval below 0, means the text knows something the odds don't.**

## `odds_movement.csv` and `.png` (test 2, football)

This test asks whether the picked result became more likely between the early odds and the closing odds.

- `pick_move_pp` is the average change in the pick's probability, in percentage points.
- `excess_pp` compares that change with results of the same kind (home, draw or away) and a similar early probability (the same tenth) that were not picked. It is a regression with one constant per kind and tenth, with standard errors clustered by game. Above 0 means the odds moved towards the pick. The comparison is needed because favourites can move on their own, and the pick is usually the favourite. With 5, 20 or 50 bands instead of 10, the estimate barely changes.
- `pick_gained`, `pick_lost`: the share of picks whose probability went up, or down.

football-data took the early odds on Friday afternoon for weekend games and on Tuesday afternoon for midweek games. We assume 15:00 UK time. The rows split the games by whether the preview came out before that snapshot (8 in 10 games) or after it. If it came out after, the move also includes the market's reaction to the preview itself.

## `backtest.csv` (test 3)

The backtest stakes 1 on each bet on the test and check seasons, at the average and at the best closing odds.

- `back every pick` and `back every favourite` are simple strategies, shown for comparison.
- `... model, edge > 5%` bets when the model's probability × the odds − 1 is above 5%.
- `roi` is the profit per unit staked. Its interval is clustered by game, because one game can carry two bets.
- `long_shots`: the main rows leave out bets on long shots, meaning average odds above 10. Their best quotes are the least reliable (51 where the average is 23.5), and a few of them can decide the result. The `included` rows keep them.

The best odds are the highest quote from any bookmaker, stale quotes included, so treat them as an optimistic upper bound. Retired tennis matches count as played, although bookmakers' rules on retirements differ. With few bets, the intervals are wide.
