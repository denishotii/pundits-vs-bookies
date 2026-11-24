# Advanced ML handover

## Phase 1 complete

- `pvb.advanced.dataset.load_football("early" | "closing")` joins unique article IDs,
  validates fixtures and returns aligned `target`, `metadata`, `text`, `numerical`
  and overlapping exclusion `counts`. The allowlist uses market-specific average H/D/A
  odds and three expert-pick features. Targets use H = 0, D = 1, A = 2.
- `season_positions(metadata)` returns development, test and check positions.
  Compute `chronological_folds(metadata)` on development only for search and refit.
  Training outcomes must precede the first validation prediction. Fold 3 uses `2324`.

Both markets start with 12,303 matches. Closing excludes one missing pick. Early
exclusions overlap, including 2,528 without established pre-snapshot availability
and 78 with snapshots at or after kickoff.

The early development/test/check counts are 7,364/1,784/548 and closing counts
are 9,355/2,271/676. Across the three expanding folds, early validates on
1,868/1,792/1,824 fixtures and closing on 2,376/2,303/2,300. The folds purged zero
training outcomes at validation boundaries. All 164 tests passed at Phase 1.

## Phase 2 complete — pilot HTML integrity audit

- Salted SHA-256 picked 25 of 9,356 matched articles per development season. The consented download
  checked robots.txt and fetched 100/100 at two-second intervals. Private files are in `data/pilot_html_integrity/`.
- `html_integrity.audit_article(page, reference, fetched_at=())` compares a saved page to the CSV.
  `html_audit.audit_saved` reads `<article_id>.html` and optional `manifest.csv` and writes private diagnostics
  under `results/advanced/html_integrity/` while preserving timestamp provenance and precision.
- All 100 pages parsed. Publication and modification sources agreed across JSON-LD, Open Graph and HTML.
  Timestamps had second precision, and publication agreed with both CSVs. All publications preceded kickoff
  and 28 followed the early cutoff. All modification markers followed kickoff and 95 shared a UTC date.
- Normalized sections, titles, verdicts and picks agreed with the historical CSV for all 100. There were zero
  conservative post-match text flags. The original HTML and fetch manifest are still needed for version verification.
  Download took 224.6 s, offline analysis 24.8 s, and all 210 tests passed.

Replay with `.\.venv\Scripts\python.exe -m pvb.advanced.html_audit audit --version pilot_download`.
Use `audit --version original --html-dir <archive_dir>` for the original archive, with optional `--sample` or `--manifest`.

## Phase 3 complete

- `market_probabilities(numerical, market)` proportionally normalizes inverse decimal
  odds for one named market. Models are `RawMarket` (M0), `MarketLogistic(l2)` (M1)
  and `MarketResidual(l2)` (M2). M1 fits L2 multinomial logistic regression on log
  market probabilities. M2 adds a linear correction with sum-to-zero classes to
  the log market offset. M2 accepts optional dense or CSR correction features in
  `fit` and `predict_proba`, defaulting to log market probabilities. Scaling is fit
  inside each fold. Penalized models expose `convergence_`.
- `evaluate_predictions(target, predictions)` aligns by match ID. It reports log
  loss with a 1e-15 floor, multiclass Brier, accuracy, class metrics, calibration
  and paired log-loss changes versus M0. `experiments.run_experiment(market)`
  selects L2 from 0.0001, 0.001, 0.01 and 0.1 by pooled development log loss.
  Scores, calibration, predictions and manifests are in `results/advanced/{early,closing}/`.

Run `.\.venv\Scripts\python.exe -m pvb.advanced.experiments --market early` or use `closing`.

Pooled log loss for M0/M1/M2 was 0.980928/0.981686/0.981282 early and
0.979448/0.979540/0.979348 closing. All 48 fits converged and all 190 tests passed.
The early cohort is conditional on estimated pre-snapshot article availability.

## Phase 4 complete — preliminary classical NLP

- `preprocessing.prepare_documents` normalizes NFC and whitespace. Primary uses
  preview, team news and head-to-head. Full adds title, intro, verdict and other.
  Primary can contain picks. Scoreline and phrase diagnostics are heuristic.
- `fit_tfidf` fits lowercase word unigrams/bigrams per training fold in CSR, with
  sublinear TF, smoothed IDF, L2 normalization, min_df=5, max_df=0.95 and 10,000
  features. It retains single digits and contractions.
- `MarketResidual(l2, standardize=False)` accepts preprocessed features. The runner
  standardizes numerical inputs per fold and weights TF-IDF by 0.5, equivalent to
  a 4× text slope penalty. M3 adds three pick features to M2, M4 adds primary TF-IDF
  to M2, and M5 uses both. `nlp_experiments.run_experiment(market)` uses the Phase 3
  folds and grid with full convergence. The full-article ablation reuses M5's L2.
  Reports are in `results/advanced/nlp/{early,closing}/`.
- With 5,484 early and 6,979 closing validation matches, M5 minus M3 log loss
  was +0.000009784 and +0.000000875 respectively. M3 minus M2 was +0.000537727
  and +0.000214891. M4 minus M2 was +0.000009871 and +0.000000854. Full-article
  M5 also scored slightly worse than M3 in both markets. M2–M5 selected L2=0.1
  for both markets. All 63 optimizer fits per market converged. Each text
  configuration had 10,000 features per fold. All 196 suite tests passed.
- Phase 4 tuned and scored on the same development folds and excluded both holdouts.

Run with `.\.venv\Scripts\python.exe -m pvb.advanced.nlp_experiments --market early`
or substitute `closing`.

## Phase 5 complete — block penalties and contextual NLP

- `MarketResidual(l2, text_l2=..., standardize=False).fit(..., numerical_width=...)`
  penalizes numerical and text slopes separately with the fixed log-market offset.
- `contextual_experiments.run_experiment` freezes selected M2/M3 penalties for text
  comparisons. M4/M5 use primary TF-IDF and M6/M7 use frozen MiniLM, each without/with picks.
- `embeddings.embed_articles` token-chunks primary sections, pools normalized chunk
  vectors by token count and normalizes the article. Empty inputs yield zeros. The
  private SQLite cache keys on ID, normalized content, encoder revision and settings.
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

## Phase 06A complete — signal diagnostics

- Run `pvb.advanced.diagnostics` offline. See `results/advanced/diagnostics/`. Eight gradient checks
  passed for the shared residual objective, five seeds recovered planted signal and Phase 05 replayed.
- MiniLM had weak pick recovery and no consistent gain on odds movement. Picks recovered 62–64%
  of market KL and global text 1.6–2.4%. Legacy aliases learn from the whole corpus and are
  excluded from advanced models. The ignored 50-article attribution review has no human labels.

## Phase 06B complete — team-aware NLP and observed football history

- Run `pvb.advanced.team_historical` with the private Phase 06A review and local MiniLM cache.
  Reports in `results/advanced/team_historical/` retain 5,484 early and 6,979 closing validation fixtures.
- `team_text` builds fixture-oriented, name-masked side documents. Side TF-IDF led pick recovery
  at 52.33% early and 52.92% closing, and recovered 31.86% and 32.47% of market KL.
- `historical_features.load_history` accepts local football-data files with matched-export fallback.
  The as-of replay carries Elo and form across seasons and the two English divisions.
- Residual `block_penalties` separate numerical, historical and text regularization. History and
  team text gave no consistent incremental gain. All 528 fits and 239 tests passed.

## Phase 07 complete — author reliability and nonlinear models

- `author_features.author_features` replays each market and fold with strict result-release cutoffs,
  shrinking rates by 20 previous population observations excluding the current author.
- `pvb.advanced.nonlinear_experiments` reuses Phase 06B cohorts, the side cache and prior penalties.
  CatBoost uses two fixed-iteration candidates and 32 training-fold SVD components plus two flags.
  Install optional `requirements-nonlinear.txt` in the existing Python 3.11 environment.
- Fixed exact validation-cohort checks and overlapping source kickoff validation. The matched
  history reproduces 9,356 fixtures, eight short of a double round robin for the observed teams.
- Author reliability worsened pooled linear loss in both markets. Every CatBoost comparison
  trailed raw odds and selected alpha zero. All 114 candidate fits completed and 251 tests passed.
- See `results/advanced/nonlinear/report.md` and `frozen_models.json` for the freeze inputs.

## Phase 08 complete — final freeze and evaluation preparation

- `results/advanced/final/` contains the frozen F0–F3 specifications, protocol, research summary and verification.
- `pvb.advanced.final_evaluation` plans by default. `--verify-development` matched both Phase 07 data
  fingerprints and the attribution fingerprint. All 257 tests passed, including six new tests.
- The future runner refits once per market on released 2122–2425 rows for both 2526 and 2627.
  Scoring requires `--execute-holdouts` and a frozen, approved article audit policy. Neither holdout was scored.
- Next obtain original HTML and contemporaneous article versions, verify attribution and odds/result times,
  and approve screened-cohort rules. Early uses 15:00 Europe/London Tuesday or Friday, closing uses kickoff,
  and release uses kickoff plus three hours or 03:00 UTC the next day for date-only fixtures.
