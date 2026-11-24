# Phase 07 — Author reliability and nonlinear market modeling

Author reliability increased pooled log loss in both linear comparisons and both markets. Every unblended CatBoost model lost to raw probabilities in all three development folds. Every selected blend put zero weight on CatBoost. Freeze four models for later evaluation once article-version integrity has been checked.

## Focused implementation review

- Phase 06A and 06B use the development seasons 2122–2425. The first validation prediction follows every training target release in each fold. Standardization, TF-IDF vocabulary and IDF, and historical median imputation are fitted only on fold-training rows. Frozen embeddings and the fixture-based attribution cache carry no learned fold statistics. The legacy whole-corpus alias learner remains outside these models. Phase 06A's post-snapshot movement diagnostic is retrospective. It takes closing probabilities as targets and uses a conservative release proxy.
- Verified 9,356 realized H/D/A labels against the ordered match scores and 9,355 expert H/D/A labels against the ordered predicted scores. All agreed. Phase 06B also checked source-side orientation and the separate verdict-heading control. Phase 07 matched both Phase 06B market input and validation-ID fingerprints. All models in a market share every validation fixture.
- Phase 06A could silently discard extra forecast IDs before scoring. The categorical, soft-target and uncertainty paths now check the exact out-of-fold cohort before slicing. Phase 06B and Phase 07 share that check. The old artifacts were preserved.
- The historical result loader compared scores when raw and matched sources overlapped, but could accept conflicting kickoffs. That could admit a result too early for another fixture's historical features. It now rejects conflicting kickoffs and retains an exact kickoff when the other source only supplies an agreeing date. Phase 07's matched-only history is unaffected.
- The configured raw football results directory is absent, and the adjacent previews archive contains a CSV only. The matched export reproduces all 24 Phase 06B league-season coverage rows. There are 9,356 observed fixtures and 9,364 expected from the observed team counts under a double round robin. Missing fixtures by season and league are 2122 E1 and F1 (one each), 2223 E1 (one), 2324 I1 (one), and 2425 F1, I1 and SP1 (one, two and one). The estimate is based on the observed leagues and teams. Cups, continental competitions, unobserved divisions and pre-2122 results are absent.

## Design and coverage

Expanding folds validate on 2223, 2324 and 2425, with 5,484 early matches (1,868, 1,792, 1,824) and 6,979 closing matches (2,376, 2,303, 2,300). The training, selection, history and scoring scope is limited to the four development seasons.

Author observations come from eligible fixtures for the same market. Phase 07 reads bylines from the private previews CSV and matches them by article ID. Normalization uses Unicode NFKC, case folding, whitespace collapse and simple punctuation normalization. Generic bylines, single tokens and ambiguous formats are treated as missing. Unmatched variants remain separate. The author key is used internally to group history.

Each fold replays matches in release order and uses only releases strictly before each prediction cutoff. Earlier validation results can enter a later row's history. The training features, fitted scaler and model parameters are fixed before the validation period. Author features are count, accuracy, H/D/A pick frequencies, accuracy agreeing or disagreeing with the sole market favourite, excess accuracy, and four missingness indicators. Excess accuracy averages the correctness indicator minus the contemporaneous market probability of the selected result. Market ties enter the overall quantities only. Each rate is smoothed with 20 equivalent observations from earlier released fixtures excluding the current author's own records. If that population has no support for a rate, accuracy and pick probabilities start at 1/3 and excess accuracy starts at zero. Missing bylines can contribute to population history. The scaler uses the training count and rates. The four missingness flags remain binary.

Every validation match had a usable byline. Prior author history was absent on 40 early and 34 closing matches. Another 94 matches per market had one to four prior predictions. Median counts were 340 early and 411 closing. Historical team features had zero prior observed fixtures for at least one side on 24 early and 29 closing matches. Side TF-IDF covered both sides for every validation fixture under the existing attribution heuristic. All blocks retain every fixture.

Linear models use the same log-market offset and regularized multinomial residual as Phase 06B. Numerical and history penalties are reused from the corresponding models. The author block tries L2 values 0.1, 1 and 10 and selects 10 in both markets, with and without team history. Team history uses training medians and scaling plus raw missingness flags.

CatBoost fits H/D/A probabilities directly. A uses raw and log market probabilities. B adds the three expert-pick features, C adds history, D adds author features and E adds team-aware text. E uses 32 components of randomized TruncatedSVD from the training-fold side TF-IDF difference and mean matrices, plus two side-missing flags. The sparse vocabulary and IDF are fitted on training side documents only. CatBoost uses 34 text dimensions and numeric allowlisted features. It uses fixed iterations with CPU MultiClass loss and seed 70107. The two candidates use depth/iterations/learning rate/leaf L2 of 2/400/0.035/30 and 4/500/0.025/45. Candidate 1 won for every block in both markets.

## Development results

Each cell is pooled log loss / multiclass Brier score / accuracy. Fold-level metrics and one-vs-rest ten-bin calibration are in the market-specific metrics and calibration CSVs. Brier sums squared errors across the three classes.

| Model | Early | Closing |
| --- | ---: | ---: |
| Raw market | 0.980928 / 0.584944 / 52.90% | 0.979448 / 0.583735 / 53.22% |
| Linear market residual | 0.981171 / 0.585147 / 52.86% | 0.979348 / 0.583718 / 53.10% |
| Linear market + picks | 0.981222 / 0.585154 / 52.95% | 0.979467 / 0.583773 / 53.13% |
| Linear market + picks + author | 0.981279 / 0.585194 / 52.95% | 0.979488 / 0.583784 / 53.12% |
| Linear market + picks + history | 0.981176 / 0.585129 / 53.08% | 0.979497 / 0.583787 / 53.19% |
| Linear market + picks + history + author | 0.981232 / 0.585168 / 53.05% | 0.979520 / 0.583798 / 53.20% |
| Unblended CatBoost A | 0.984623 / 0.587143 / 52.75% | 0.982114 / 0.585264 / 52.79% |
| Unblended CatBoost B | 0.984519 / 0.587122 / 52.52% | 0.982112 / 0.585208 / 52.90% |
| Unblended CatBoost C | 0.986570 / 0.588181 / 52.74% | 0.983549 / 0.585879 / 53.03% |
| Unblended CatBoost D | 0.986805 / 0.588387 / 52.90% | 0.984782 / 0.586694 / 53.00% |
| Unblended CatBoost E | 0.988479 / 0.589374 / 52.88% | 0.984297 / 0.586314 / 52.84% |
| Selected market–CatBoost blends A–E | 0.980928 / 0.584944 / 52.90% | 0.979448 / 0.583735 / 53.22% |

Pooling weights matches equally. The fit for each CatBoost block was selected on unblended log loss, then evaluated at alpha values 0, 0.1, 0.25, 0.5 and 1 in `(1 − alpha) × market + alpha × CatBoost`. All five blocks chose alpha 0 in both markets. It reproduces raw probabilities exactly and yields zero paired difference in every fold. Even alpha 0.1 worsened pooled loss in every block. Calibration was mixed for unblended B. Its pooled bin-weighted absolute calibration error fell for away probabilities from 0.0203 to 0.0127 early and 0.0165 to 0.0105 closing, while draw errors rose from 0.0027 to 0.0067 and from 0.0051 to 0.0085.

## Paired log loss and uncertainty

Differences are model minus baseline. Intervals use 3,000 occupied seven-day UTC fixture-date block resamples within season, preserving pooled season weights. Fourteen-day intervals are in the bootstrap CSVs and support the same overall interpretation. These are descriptive intervals for fitted development forecasts. The same folds are used for selection and evaluation, so independent confirmation remains pending.

| Increment or comparison | Early Δ [95% interval] | Closing Δ [95% interval] |
| --- | ---: | ---: |
| Linear market vs raw | +0.000244 [−0.000164, +0.000657] | −0.000100 [−0.000510, +0.000312] |
| Linear picks vs linear market | +0.000050 [−0.000100, +0.000196] | +0.000119 [−0.000198, +0.000423] |
| Add author to linear market and picks | +0.000057 [−0.000028, +0.000147] | +0.000021 [−0.000060, +0.000101] |
| Add history to linear market and picks | −0.000046 [−0.000325, +0.000220] | +0.000030 [−0.000191, +0.000246] |
| Add author to linear market, picks and history | +0.000056 [−0.000026, +0.000144] | +0.000022 [−0.000055, +0.000099] |
| Best unblended CatBoost B vs raw | +0.003591 [+0.001725, +0.005513] | +0.002664 [+0.001112, +0.004339] |
| Add picks to CatBoost (B−A) | −0.000104 [−0.000918, +0.000740] | −0.000003 [−0.000655, +0.000675] |
| Add history to CatBoost (C−B) | +0.002051 [+0.000336, +0.003702] | +0.001438 [−0.000136, +0.002905] |
| Add author to CatBoost (D−C) | +0.000236 [−0.000882, +0.001414] | +0.001233 [+0.000382, +0.002143] |
| Add team text to CatBoost (E−D) | +0.001674 [+0.000041, +0.003371] | −0.000485 [−0.001813, +0.000772] |

The linear author block improved one of three validation folds in each comparison. The nonlinear author block improved zero of three in both markets. Team-aware text improved closing CatBoost in all three folds and early CatBoost in none. Both the seven- and fourteen-day closing text intervals include zero, and closing E still trailed the raw market by 0.004849. The fixed-configuration nonlinear contrasts agree with the selected contrasts because the same candidate won throughout.

## Freeze for later holdout evaluation

The [frozen configuration](frozen_models.json) proposes raw bookmaker probabilities, the linear market-only residual, the linear market-plus-picks residual and the Phase 06B market-plus-picks residual with side TF-IDF. The NLP representation had the strongest development pick recovery among the tested team-aware choices. Freeze these settings separately for early and closing. Refit learned transforms and coefficients once on released development rows, then keep them fixed for both held-out seasons.

The results support a four-model set. Historical gains vary by market and the unblended nonlinear comparisons lost to raw odds. Alpha zero yields the original raw baseline, which is already in the set.

## Remaining integrity limits

Original HTML and contemporaneous versions are still unavailable. Phase 02's pilot provides a consistency check on downloaded pages. The mostly post-kickoff modification markers leave the text, byline and pick visible at prediction unverified. Attribution has no human-labeled precision/recall estimate. Early odds rely on a Tuesday or Friday 15:00 Europe/London proxy, closing uses kickoff, and actual capture times are unknown. History uses a three-hour outcome-release buffer or the next UTC day plus three hours for date-only kickoffs. Actual result-release times and complete fixture feeds are unavailable. Author history is conditional on the matched, market-eligible sample and the parser's single recorded byline.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-nonlinear.txt
.\.venv\Scripts\python.exe -m pvb.advanced.nonlinear_experiments
.\.venv\Scripts\python.exe -m pytest -q
```

The runner uses the private previews CSV, the local matched export and the Phase 06B attribution cache. It runs offline and uses no pretrained encoder. All 54 linear candidate fits converged and all 60 CatBoost fits completed. The warm-cache run took 75.7 seconds in Python 3.11.9 with CatBoost 1.2.10. The full test suite passed 251 tests in 9.55 seconds. Per-market execution times are in `metadata.json`. Forecasts, fitted models and author histories stay in memory. Caches are Git-ignored.
