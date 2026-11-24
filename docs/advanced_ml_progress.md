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

Both markets start with 12,303 matched records. Closing excludes one missing
pick. Early exclusion reasons overlap, including 2,528 articles without
established pre-snapshot availability and 78 snapshots at or after kickoff.

| Market | Eligible | Excluded | Development | 2526 test | 2627 check |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closing | 12,302 | 1 | 9,355 | 2,271 | 676 |
| Early | 9,696 | 2,607 | 7,364 | 1,784 | 548 |

| Train seasons | Validation season | Closing train / validation | Early train / validation |
| --- | --- | ---: | ---: |
| 2122 | 2223 | 2,376 / 2,376 | 1,880 / 1,868 |
| 2122–2223 | 2324 | 4,752 / 2,303 | 3,748 / 1,792 |
| 2122–2324 | 2425 | 7,055 / 2,300 | 5,540 / 1,824 |

The real-data folds purged zero outcomes. All 164 tests passed at Phase 1.

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
  0.0001, 0.001, 0.01 and 0.1 by pooled per-match development log loss. Scores,
  calibration, predictions and the input manifest are in
  `results/advanced/{early,closing}/`.

Run `.\.venv\Scripts\python.exe -m pvb.advanced.experiments --market early` or use `closing`.

### Pooled development out-of-fold results

| Market | Matches | Model | L2 | Log loss | Brier | Accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Early | 5,484 | M0 | — | 0.980928 | 0.584944 | 52.90% |
| Early | 5,484 | M1 | 0.0001 | 0.981686 | 0.585587 | 52.95% |
| Early | 5,484 | M2 | 0.1 | 0.981282 | 0.585268 | 52.86% |
| Closing | 6,979 | M0 | — | 0.979448 | 0.583735 | 53.22% |
| Closing | 6,979 | M1 | 0.001 | 0.979540 | 0.583846 | 52.99% |
| Closing | 6,979 | M2 | 0.1 | 0.979348 | 0.583718 | 53.10% |

All 48 fits converged and all 190 tests passed. The same folds select L2 and
supply the scores. The original project's historical aggregates are already
known. The early cohort is conditional on estimated pre-snapshot article
availability, so it represents a subset of matches with early odds.

## Phase 4 complete — preliminary classical NLP

- `pvb.advanced.preprocessing.prepare_documents` joins NFC-normalized sections
  with collapsed whitespace. Primary uses preview, team news and head-to-head.
  Full article adds title, intro, verdict and other. Primary can still contain
  explicit predictions. Scoreline and phrase diagnostics are heuristic counts.
- `fit_tfidf` fits word unigrams/bigrams per training fold and representation.
  It lowercases, retains single digits and contractions, uses sublinear TF,
  smoothed IDF and L2 document normalization, and sets min_df=5, max_df=0.95,
  max_features=10000. Training and validation text matrices stay in CSR.
- `MarketResidual(l2, standardize=False)` accepts preprocessed correction
  features. Phase 3 keeps the original default. The runner standardizes log
  market probabilities and expert-pick columns on each training fold, then
  combines them as needed with TF-IDF weighted by 0.5. M3 adds `pick_home`,
  `pick_draw` and `pick_goal_diff` to M2. M4 adds primary TF-IDF to M2. M5
  combines picks and primary TF-IDF. The residual logit offset is `log(p_market)`.
  The penalty is L2/2 times the squared slope norm relative to mean log loss,
  with unpenalized intercepts and a 4× effective penalty on unweighted text slopes.
- `pvb.advanced.nlp_experiments.run_experiment(market)` uses the Phase 3 folds
  and L2 grid, requiring convergence in every fold. The full-article ablation
  uses M5's selected L2. Reports are in `results/advanced/nlp/{early,closing}/`.
- With 5,484 early and 6,979 closing validation matches, M5 minus M3 log loss
  was +0.000009784 and +0.000000875 respectively. M3 minus M2 was +0.000537727
  and +0.000214891. M4 minus M2 was +0.000009871 and +0.000000854. Full-article
  M5 also scored slightly worse than M3 in both markets. M2–M5 selected L2=0.1
  for both markets. All 63 optimizer fits per market converged. Each text
  configuration had 10,000 features per fold. All 196 suite tests passed.
- Phase 4 used the same development folds for tuning and scoring. It excluded
  both historical holdout seasons.

Run with `.\.venv\Scripts\python.exe -m pvb.advanced.nlp_experiments --market early`
or substitute `closing`.

## Phase 5 complete — block penalties and contextual NLP

- `MarketResidual(l2, text_l2=..., standardize=False).fit(..., numerical_width=...)`
  penalizes numerical and text slopes separately. The original scalar L2
  interface, sum-to-zero class contrasts and fixed log-market offset still apply.
- `pvb.advanced.contextual_experiments.run_experiment` selects L2 for M2 and M3
  first, then freezes each penalty for its text comparisons. M4/M5 use primary
  TF-IDF and M6/M7 use frozen MiniLM embeddings, each without/with picks.
- `pvb.advanced.embeddings.embed_articles` chunks primary sections by token ID
  and weights normalized chunk vectors by token count before article normalization.
  An empty input yields zeros. The private SQLite cache under `data/cache/contextual/`
  keys on article ID, normalized section content, encoder revision and configuration.
- MiniLM was [first uploaded on 2021-08-30](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commit/acbb28c8aa70f5503c85d6b90e8cd65606993a20).
  Historical availability of the pinned revision and overlap with its pretraining
  corpora are unverified. Global pooling can lose team attribution.
- Install `requirements-contextual.txt` with `.\.venv\Scripts\python.exe -m pip install -r requirements-contextual.txt`.
  Run `.\.venv\Scripts\python.exe -m pvb.advanced.contextual_experiments --market both`.
  Add `--offline` after caching the checkpoint, or `--tfidf-only` to skip MiniLM.
- Reports in `results/advanced/contextual/{early,closing}/` cover 5,484 and
  6,979 development validation matches. The two TF-IDF models chose L2=1 in
  both markets and trailed their numerical references. MiniLM slightly worsened
  early log loss and reduced closing log loss by 0.0000083 without picks and
  0.0000106 with picks. Closing generated 9,355 vectors on CUDA and early
  reused all 7,364 it needed. All selected fits converged and all 200 tests passed.

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
