"""The three tests from docs/plan.md.

Status: TODO (analysis and writing role). Train on 2021/22-2024/25, test once
on 2025/26.

1. Text beyond odds: predict the result from the closing-odds probability
   alone, then from the odds plus the text features (logistic regression).
   Compare log loss / Brier score on 2025/26.
2. Odds movement: previews come out about 2 days before kick-off. Did the
   odds on the pundit's pick shorten between the earlier and closing odds?
3. Betting backtest: a simple rule ("bet when the text model's probability
   beats the odds by more than X") on 2025/26 only, after the bookmaker's margin.
"""


def text_beyond_odds():
    raise NotImplementedError


def odds_movement():
    raise NotImplementedError


def betting_backtest():
    raise NotImplementedError
