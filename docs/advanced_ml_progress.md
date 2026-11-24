# Advanced ML handover

## Phase 1 complete

- `pvb.advanced.dataset.load_football("early" | "closing")` uses configured paths
  with optional overrides. It joins records by unique article ID, checks the
  fixture and source data, and returns aligned `target`, `metadata`, `text`,
  `numerical`, and exclusion `counts`. Counts by reason can overlap.
- The numerical allowlist contains the selected market's average home, draw,
  and away odds plus three baseline expert-pick features. The target uses
  H = 0, D = 1, A = 2. Actual scores and precomputed features, including the
  supervised Naive Bayes values, are excluded from model inputs.
- `pvb.advanced.validation.season_positions(metadata)` returns development,
  test and check positions. `chronological_folds(metadata)` returns positional
  train/validation pairs and statistics. For scikit-learn search and refit,
  subset all inputs to development first and compute folds on that subset.
  A training outcome must be available strictly before the first validation
  prediction in its fold. Fold 3 treats the requested `2323` as `2324`.

### Phase 1 cohort verification

Both markets start with 12,303 matched records.

| Market | Eligible | Excluded | Development | 2526 test | 2627 check |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closing | 12,302 | 1 | 9,355 | 2,271 | 676 |
| Early | 9,696 | 2,607 | 7,364 | 1,784 | 548 |

Closing excludes one missing pick and preserves the baseline sample. Early
exclusion counts are 2,528 articles without established pre-snapshot
availability, 78 snapshots at or after kickoff, one missing odds record,
and one missing pick. The reason counts overlap.

| Train seasons | Validation season | Closing train / validation | Early train / validation |
| --- | --- | ---: | ---: |
| 2122 | 2223 | 2,376 / 2,376 | 1,880 / 1,868 |
| 2122–2223 | 2324 | 4,752 / 2,303 | 3,748 / 1,792 |
| 2122–2324 | 2425 | 7,055 / 2,300 | 5,540 / 1,824 |

Zero outcomes were purged in the real-data folds. The 26 Phase 1 tests and
all 164 suite tests passed at the end of Phase 1.

## Phase 3 complete

- `pvb.advanced.market` validates H/D/A arrays and proportionally normalizes
  reciprocal decimal odds. `market_probabilities(numerical, market)` selects
  only the named market's average odds.
- `pvb.advanced.models` provides `RawMarket` (M0), `MarketLogistic(l2)` (M1),
  and `MarketResidual(l2)` (M2). M1 uses L2-regularized multinomial logistic
  regression on standardized log market probabilities. M2 adds a linear
  correction to the fixed log market offset and uses sum-to-zero class
  coordinates. `fit(market, y, features=None)` and `predict_proba(market,
  features=None)` on M2 accept optional dense or CSR correction features.
  With no features passed, M2 uses log market probabilities for its correction.
  Scaling is learned inside each fold. Both models penalize slopes relative
  to mean training loss and expose `convergence_`.
- `pvb.advanced.evaluation.evaluate_predictions(target, predictions)` aligns
  by match ID and returns log loss, Brier, accuracy, class metrics, calibration
  bins, and paired per-match log-loss changes versus M0. Log loss uses the
  original 1e-15 floor and Brier sums squared errors across classes per match.
- `pvb.advanced.experiments.run_experiment(market)` selects L2 from
  `0.0001, 0.001, 0.01, 0.1` by pooled development log loss, weighting each
  match equally. `results/advanced/{early,closing}/` contains `metrics.csv`,
  `folds.csv`, `regularization_search.csv`, `class_metrics.csv`,
  `calibration.csv`, `oof_predictions.csv`, and `metadata.json`. The manifest
  records a fingerprint of the effective development inputs and package versions.

```powershell
.\.venv\Scripts\python.exe -m pvb.advanced.experiments --market early
.\.venv\Scripts\python.exe -m pvb.advanced.experiments --market closing
```

### Pooled development out-of-fold results

| Market | Matches | Model | L2 | Log loss | Brier | Accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Early | 5,484 | M0 | — | 0.980928 | 0.584944 | 52.90% |
| Early | 5,484 | M1 | 0.0001 | 0.981686 | 0.585587 | 52.95% |
| Early | 5,484 | M2 | 0.1 | 0.981282 | 0.585268 | 52.86% |
| Closing | 6,979 | M0 | — | 0.979448 | 0.583735 | 53.22% |
| Closing | 6,979 | M1 | 0.001 | 0.979540 | 0.583846 | 52.99% |
| Closing | 6,979 | M2 | 0.1 | 0.979348 | 0.583718 | 53.10% |

All 48 candidate fits converged. The 52 focused advanced tests and all 190
suite tests passed under Python 3.11.9. Saved OOF predictions reproduce the
pooled metrics and paired losses. The same folds selected L2 and supplied
these scores, so the scores describe development selection. Seasons 2526
and 2627 remain reserved for independent evaluation.

The runner uses Phase 1 eligibility for reproducible comparisons with future
NLP experiments. The loader checks article-body presence and valid picks, then
the runner discards text and pick features. In particular, the early cohort is
conditional on estimated pre-snapshot article availability and is not
representative of all matches with early odds.

## Timing limitations and next step

- The baseline's approximate early snapshot is 15:00 Europe/London on
  Tuesday for Tuesday through Thursday fixtures and Friday for Friday
  through Monday fixtures. Closing uses kickoff as the prediction cutoff
  proxy. The actual odds capture times are unavailable.
- UTC midnight values are treated as date-only. This policy excludes five
  additional same-day articles from the early sample. Outcome availability
  is estimated as kickoff plus three hours, or the next UTC day plus three
  hours when kickoff is date-only. Actual result release times are unavailable.
- Phase 2 remains deferred until raw HTML is available. Then audit article
  versions, publication timestamps, and market timestamp provenance before
  drawing conclusions from text experiments. Original publication times alone
  cannot establish when the scraped text or expert pick became available.
