# Phase 06A signal diagnostics

Explicit expert picks predict outcomes and recover much of the variation in bookmaker probabilities. Global article text alone yields a small standalone log-loss gain, carries limited directional information, and produces no consistent improvement over an odds-based baseline. The best next experiment is a small team-aware representation comparison, conditional on manual verification of sentence attribution.

## Scope and implementation verification

The experiment uses expanding development folds validating on 2223, 2324 and 2425. The outcome, pick and market-recovery tasks use 5,484 early and 6,979 closing validation matches. Hyperparameters use pooled development validation loss. The same folds supply descriptive scores. Both historical holdout seasons were excluded. All learned scalers and TF-IDF vocabularies/IDF values are fit on training rows. Frozen MiniLM uses the Phase 05 checkpoint and primary cache. The existing article-version risk and uncertainty about checkpoint and pretraining provenance carry forward.

The audit verified H/D/A ordering, proportional margin removal, the fixed log-market offset, row alignment by article ID, training-only preprocessing, the mean-loss scaling of L2, independent block penalties and paired loss alignment. Training target availability precedes the first prediction in each validation season. Eight central finite-difference checks of the residual objective passed across dense/CSR, single/block L2 and hard/soft labels. Maximum absolute gradient error was **2.0 × 10⁻¹⁰**. The development input fingerprints matched Phase 05, and its six residual scores per market replayed within **7.1 × 10⁻¹⁴** log loss. The legacy feature exporter learns aliases using the entire football article corpus, including within-season articles. Chronological use of those aliases would leak future text. The Phase 03–05 models use market odds, picks and article representations directly, so the alias issue has no effect on their scores.

Across five seeds with 2,500 training and 5,000 independent validation rows each, a correctly specified synthetic market plus 96 pure-noise features saw mean excess log loss fall from **+0.04835** to **+0.00388** as block L2 increased from 0.0001 to 1. The noise slope norm fell from 0.569 to 0.115. At L2 = 0.03, the planted omitted feature improved log loss by **0.12733** alone and **0.09264** among 96 noise features. Both improvements held against the known data-generating probabilities. All 45 fits converged. These controls verify recovery of a planted signal of the tested strength.

## Standalone outcomes and pick direction

Lower log loss is better. The outcome models use standardized explicit pick features or fixed normalized text features. Only the raw-market reference uses odds.

| Model | Early outcome log loss | Closing outcome log loss |
|---|---:|---:|
| Training class frequency | 1.070227 | 1.072960 |
| Expert picks | 1.015966 | 1.018244 |
| TF-IDF | 1.070215 | 1.072681 |
| MiniLM | 1.069751 | 1.072314 |
| MiniLM + picks | 1.015865 | 1.018077 |
| Raw market | 0.980928 | 0.979448 |

MiniLM improves on class-frequency log loss in all three seasons on both cohorts, by 0.00048 early and 0.00065 closing pooled. Standalone text accuracy stays at the frequency baseline, 44.2% early and 43.7% closing. Adding MiniLM to expert picks improves pooled log loss by 0.00010 and 0.00017, and improves two of three validation seasons in each market. Pooled accuracy is unchanged. The raw market leads on log loss, Brier and accuracy. The outcome CSVs also record classwise recall, precision and conditional loss.

| Pick-recovery model | Early log loss / accuracy / macro F1 | Closing log loss / accuracy / macro F1 |
|---|---:|---:|
| Training class frequency | 1.083 / 41.8% / .196 | 1.082 / 41.9% / .197 |
| TF-IDF primary | 1.062 / 43.8% / .309 | 1.062 / 43.8% / .298 |
| MiniLM primary | 1.073 / 41.6% / .298 | 1.072 / 41.9% / .291 |
| TF-IDF full, verdict control | .937 / 55.1% / .522 | .929 / 56.5% / .537 |
| MiniLM full, verdict control | 1.065 / 43.1% / .311 | 1.061 / 44.2% / .372 |

The primary representation excludes the title, introduction and verdict. TF-IDF recovers some direction from the remaining body. Global MiniLM improves log loss and macro F1 with accuracy near the class-frequency baseline. Full-article controls intentionally expose the verdict and also add the other excluded sections. These scores diagnose representation and pooling. The improvement from the full article is larger for TF-IDF, and the comparison leaves the contribution of each added section unresolved. The pick CSVs include fold-level and pooled confusion matrices.

## Market overlap and closing-odds movement

Mean-market baseline KL was 0.07684 early and 0.07773 closing. Expert picks reduced KL by **64.1% early and 62.0% closing**, with home and away correlations near 0.8. TF-IDF alone reduced KL by **2.4% and 2.3%**, MiniLM by **1.6% and 1.7%**, and MiniLM plus picks by **64.8% and 62.8%**. Text-only models recovered more of the variation in draw probability (correlations 0.43–0.55) than in either named side (0.07–0.13). For MiniLM plus picks, probability MAE/RMSE were 0.0578/0.0787 early and 0.0599/0.0814 closing. These are measures of information overlap. The source and direction of information flow remain unidentified.

| Publication cohort | Development / validation | No-movement KL | Historical market-only KL | Market + picks + MiniLM KL |
|---|---:|---:|---:|---:|
| Before estimated snapshot | 7,364 / 5,484 | .00187293 | .00186204 | .00186253 |
| At least a 3-hour buffer | 6,752 / 5,042 | .00182753 | .00181806 | .00181771 |
| After snapshot, before kickoff (retrospective) | 1,926 / 1,449 | .00227737 | .00225246 | .00225184 |

There were 9,354 matched development rows with valid dual snapshots. The three rows above cover 78.7%, 72.2% and 20.6% of that base. The buffered cohort is a subset of the first. Fifty-six matches had a snapshot at or after kickoff and eight had uncertain publication timing relative to the snapshot. Market calibration improved on no movement by 0.58% in the primary cohort. On 8,438 material class movements of at least one percentage point, market calibration called 55.75% of signs correctly and the text-plus-pick model called 55.66%. The latter improved on calibration in one of three folds in each timing cohort. Primary market-plus-picks-plus-MiniLM change correlations were 0.036 home, 0.100 draw and 0.038 away. No movement abstains on direction and therefore scores zero on this diagnostic. The retrospective comparison uses text after the snapshot. The actual odds capture times, available article versions and executable prices are unobserved, and the associations support no causal inference.

## Training size, attribution and precision

- With the 2425 validation set held fixed, recent training sizes were 1,385–5,540 early and 1,764–7,055 closing. TF-IDF plus picks trailed the corresponding numerical baseline at every size. Early MiniLM plus picks also trailed at every size. Closing MiniLM plus picks improved by 0.000017 at 25%, then trailed by 0.000049–0.000097 at the larger sizes. The numerical baselines were non-monotonic. Going back further adds older matches, so the curves confound training size and period composition.
- Source and matched-export side order agreed with home/away in all 9,355 audited development articles. The original matcher expands curated odds-file abbreviations and requires the resulting tokens to appear in the source team name, excluding youth and women's teams. This audit checks consistency on the fixtures already selected by that rule.
- Legacy attribution learns capitalized words for a team-season with at least five previews, requiring the word to appear in 25% of that team's previews and at least ten times as often as in the full corpus. It drops recognized team-name words and words shared by more than two teams in a season. Its sentence credit carries the last identified side within a section. Reusing the learner would require a training-only corpus, a rule for carrying aliases forward, and validation of stale or ambiguous aliases.
- Using literal fixture names, curated spellings and explicit role phrases, 43,743 of 188,335 sentences were home-only (23.2%), 42,780 away-only (22.7%), 8,232 both (4.4%) and 93,580 neither (49.7%). All articles had at least one proposed exclusive sentence for each side. There were 497 sentences with an ambiguous shared-name cue, 893 with naming and role cues pointing to opposite sides, and 39,220 unassigned sentences with a generic reference. The heuristic uses no learned aliases, player linking or sentence carryover. Coverage measures cue availability. The private review file has 1,008 sentences from 50 deterministically sampled articles and empty human labels.

Paired differences are text-model minus numerical-baseline log loss. The table uses the corresponding market-plus-picks baseline and 3,000 within-season resamples of occupied 7-day fixture blocks. The CSVs also include 14-day intervals, fold signs, standard deviations, quantiles, and models without picks.

| Cohort and text model | Mean delta | Matches improved | 95% descriptive interval | Conditional detectable mean magnitude |
|---|---:|---:|---:|---:|
| Early TF-IDF | +.0000041 | 49.4% | [−.0000003, +.0000084] | .0000063 |
| Early MiniLM | +.0000019 | 52.1% | [−.0000161, +.0000205] | .0000264 |
| Closing TF-IDF | +.0000003 | 51.2% | [−.0000030, +.0000037] | .0000048 |
| Closing MiniLM | −.0000106 | 53.9% | [−.0001436, +.0001264] | .0001911 |

Fourteen-day sensitivity estimates reach 0.000029 early and 0.000200 closing for MiniLM. MiniLM improved pooled closing loss while improving just one validation season, 2324. The detectable magnitudes are bootstrap SE × 2.80, a normal approximation at nominal two-sided 5% size and 80% sensitivity conditional on the selected fitted forecasts. The small TF-IDF magnitudes reflect very small probability perturbations. Development selection, refitting variation and future-season drift fall outside these descriptive intervals.

## Ranked assessment of the six hypotheses

| Rank and hypothesis | Supporting evidence | Counterevidence | Remaining uncertainty and next experiment |
|---|---|---|---|
| 1. Predictive information is already reflected in odds | Picks predict outcomes, recover 62–64% of mean-market KL and add almost nothing after conditioning on odds. | Global text alone recovers only 1.6–2.4% of market KL, so the overlap evidence is strongest for explicit picks. | Measure incremental value of independently sourced, timestamped team covariates at fixed odds snapshots. |
| 2. Global NLP loses team-specific or directional information | Primary MiniLM barely improves pick accuracy. Full TF-IDF improves pick accuracy markedly, and most articles contain cues for both named sides. | Primary representations recover some direction and market variation, especially for draws. Even full MiniLM recovers direction weakly. | Label the 50-article sample, then compare separate home/away pooling against global pooling on pick recovery and fixed residual models. |
| 3. Primary article text contains little usable predictive signal | Text-only accuracy is unchanged and standalone log-loss gains are small. | MiniLM yields small standalone gains across all folds. Explicit picks carry clear standalone outcome signal. | Separate sparse directional cues from weak article content using a supervised team-aware diagnostic. |
| 4. The sample is too small for a weak incremental signal | Paired closing MiniLM changes vary by season and its conditional resolution is around 0.0002. | Training with more existing observations did not deliver a consistent improvement in text increments. | If a team-aware signal appears, test it on more prospectively archived fixtures with season composition tracked. |
| 5. The temporal question hides a signal | Pick and article availability vary relative to an approximate early quote, and article-version integrity is unresolved. | Pre-snapshot, buffered and retrospective comparisons all show tiny and inconsistent text gains over historical calibration. | Capture timestamped odds and contemporaneous article versions before revisiting the movement question. |
| 6. Fitting fails to recover an available signal | Weak, sparse, team-dependent real signals could still be missed by the chosen linear specifications. The legacy same-corpus alias learner would leak if reused as-is. | The gradient and alignment checks passed, the Phase 05 comparison replayed, and the residual optimizer recovered the planted signal with noise present. | Use a planted weak, team-signed feature as a control for the planned team-aware representation. |

Prioritize the manually checked team-aware comparison. Use historical team statistics as the next numerical benchmark if more modeling is warranted. Larger prospective datasets matter if a directional signal survives this comparison. Author reliability and nonlinear global text models have lower priority given current results. Snapshot provenance is a prerequisite for a stronger odds-movement study.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pvb.advanced.diagnostics
.\.venv\Scripts\python.exe -m pytest -q
```

`metadata.json` records settings, input fingerprints, runtime, the prior penalties and embedding cache statistics. All development experiments completed, all candidate model fits converged, and all 221 tests passed. Aggregate CSVs contain fold-level and pooled scores. `private/attribution_review.csv` and `private/*_residual_oof.csv` stay Git-ignored along with the embedding caches. The original HTML archive remains unavailable.
