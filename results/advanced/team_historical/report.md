# Phase 06B — Team-aware text and historical football features

## Cohorts and attribution

The experiments use the Phase 06A expanding development folds validating on 2223, 2324 and 2425. Early has 7,364 eligible development fixtures and 5,484 validation fixtures. Closing has 9,355 and 6,979. Every model retains the original market's eligible match IDs. The 2526 and 2627 seasons were excluded. Log loss is the primary metric. Brier score is the sum across the three outcome classes, averaged over matches.

Article-side names in the source and the matched export agree with fixture order for all 9,355 closing development articles. The procedure aligns article sides to fixture home and away before attributing sentences. It uses fixed fixture names, curated spellings and explicit football roles. Overlapping names use the longest match, and isolated shared tokens are ambiguous. Sentences with both teams or conflicting cues do not enter either side document. Pronouns receive no credit on their own, and no attribution carries between sentences. There are no corpus-learned aliases.

The private stratified review has 192 sentences from 49 of the 50 Phase 06A review articles. Proposals include 48 home-only, 48 away-only, 37 both-team and 59 unattributed sentences across all four development seasons. The subset contains no flagged shared-name ambiguity cases. No human labels were available, so attribution precision and recall remain unverified.

| Validation cohort | Home-only | Away-only | Both-team | Unattributed | Shared-name flags | Missing either side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early | 25,942 | 25,165 | 4,814 | 52,406 | 366 | 0 |
| Closing | 33,086 | 32,083 | 6,266 | 66,917 | 434 | 0 |

The primary representation uses preview, team news and head-to-head text. Accepted team-name mentions become `TEAM` in side documents. Side TF-IDF fits one vocabulary across the training-fold home and away documents and combines the sparse home-minus-away and mean vectors. Team MiniLM encodes each side separately using the frozen Phase 05 encoder with the same token-aware section chunking and pooling. Both paired representations add two missing-attribution flags. Empty side documents have zero vectors, and their matches remain eligible. Global TF-IDF, global MiniLM, side TF-IDF and team MiniLM have 10,000, 384, 20,002 and 770 features across all folds. The private caches include configuration and content fingerprints.

## Representation diagnostics

| Representation | Outcome log loss early / closing | Pick log loss early / closing | Pick accuracy early / closing | Share of mean-market KL recovered early / closing |
| --- | ---: | ---: | ---: | ---: |
| Global TF-IDF | 1.070215 / 1.072681 | 1.061542 / 1.062390 | 43.80% / 43.76% | 2.44% / 2.27% |
| Global MiniLM | 1.069751 / 1.072314 | 1.073423 / 1.072206 | 41.61% / 41.91% | 1.64% / 1.66% |
| Side TF-IDF | 1.061454 / 1.062173 | 0.990616 / 0.986884 | 52.33% / 52.92% | 31.86% / 32.47% |
| Team MiniLM | 1.067331 / 1.069611 | 1.030507 / 1.025178 | 46.99% / 47.77% | 14.02% / 14.48% |
| Class frequency | 1.070227 / 1.072960 | 1.082660 / 1.081775 | 41.78% / 41.94% | — |
| Explicit picks | 1.015966 / 1.018244 | — | — | 64.12% / 61.98% |
| Raw market | 0.980928 / 0.979448 | — | — | — |
| Verdict parser control | — | 0.020203 / 0.020203 | 100% / 100% | — |

Both side-aware representations improve expert-pick recovery and standalone outcome prediction over their global counterparts. Side TF-IDF has pick macro F1 of 0.468 early and 0.470 closing, versus 0.309 and 0.298 for global TF-IDF. Its home and away probability correlations with the market are 0.577 and 0.582 early and 0.583 and 0.586 closing. Global TF-IDF correlations for the two named sides are 0.105 and 0.125 early and 0.102 and 0.121 closing. Side TF-IDF also improves standalone outcome log loss by 0.00876 early and 0.01051 closing, while remaining well behind explicit picks and the market. This supports weak directional extraction as one explanation for Phase 06A.

The deterministic positive control parses the separate `pick_text` scoreline heading, checks fixture orientation and assigns 0.98 probability to the parsed class. It agrees with every evaluated pick label. The archived labels originate from the same heading, so this checks consistency of the stored target and confirms the heading carries the direction. Only this diagnostic accesses the heading. The model text comparisons use primary sections alone. Side TF-IDF was selected for the final hybrid by the predefined rule of lowest pooled closing-cohort expert-pick recovery log loss. Fold-level scores, class metrics and confusion counts are in the aggregate CSVs.

## Historical results and availability

No local raw football-data files were available. The run uses 9,356 observed results from `data/processed/matched_football.csv`, covering all six study leagues. Given the team counts observed in each league-season, a double round robin would have 9,364 fixtures. The eight unmatched fixtures comprise two in the Championship, two in Ligue 1, three in Serie A and one in La Liga. This comparison cannot reveal clubs entirely absent from the source. The source also omits cup, European and lower-division matches and pre-2122 history. Observed history and rest intervals are therefore incomplete even in league-seasons with full expected counts.

Historical features are computed as of each market prediction. The replay assumes results are available three hours after exact kickoff. Date-only fixtures would be released at 03:00 UTC the next day, with no date-only kickoffs in this run. Only releases strictly before prediction enter features. Closing uses kickoff as a proxy, and early uses the approximate Tuesday or Friday 15:00 Europe/London snapshot. Current-fixture outcomes are excluded, and the implementation rejects source timing that would make one available before its own prediction. Results released simultaneously update Elo from a common pre-batch state. Earlier results in validation seasons can enter later histories. Model coefficients, TF-IDF vocabulary, imputation and scaling stay frozen within each validation fold.

| Historical feature | Input and availability |
| --- | --- |
| Home and away Elo and their difference | Released match scores determine win, draw or loss. Initial rating 1500, K 20, rating scale 400 and 60 points of home advantage. Update both teams by equal and opposite amounts using actual minus expected result, where a draw is 0.5. |
| Prior 5 and 10 points, goals for, goals against and goal difference | Means over the most recent observed and released fixtures in kickoff order, using available history when shorter than the window. Points are 3 for a win, 1 for a draw and 0 for a loss. |
| Home club's home PPG and away club's away PPG | Cumulative points per released match at the relevant venue. |
| Days since previous known match | Elapsed days from the most recent released match's kickoff to the prediction timestamp. |
| Available match counts | Total released fixtures per club, plus released fixtures at the relevant venue. |
| Missingness | Defined by zero eligible history overall or at the relevant venue. Training medians fill missing values, with zero for columns entirely missing in training. Twenty explicit missing flags complement 27 historical values. |

Club histories carry through season boundaries without regression or reset. Clubs moving between the two observed English divisions retain their history. First-seen clubs begin at Elo 1500 with zero observed matches. Within the early and closing validation cohorts, 24 and 29 fixtures lack any observed history for at least one side. The prior-five window is short for at least one side in 116 and 143 fixtures, and the prior-ten window in 216 and 280. Venue history is absent on at least one side in 48 and 58. Neither market drops fixtures due to missing history.

## Residual models

All residuals use the fixed log-market offset and the existing multinomial optimizer. The numerical block includes the log market probabilities and, when specified, the three explicit pick features. Historical features and text each have independent slope penalties. Numerical and history L2 candidates are 0.01, 0.1, 1 and 10. Text candidates are 0.001, 0.01, 0.1 and 1. The market-only and market-plus-picks references select numerical penalties separately. Each history or text addition freezes its reference penalties while tuning the added block. Every candidate reuses its fold's preprocessed matrices. Standalone diagnostics use Phase 06A's 0.0001, 0.001, 0.01 and 0.1 grid.

Paired changes use model minus reference log loss, so negative values favor the addition. Intervals resample 3,000 occupied seven-day calendar blocks within validation seasons and retain season weights. The CSVs include 14-day sensitivity intervals. Development folds serve both selection and scoring, so the intervals describe the selected development forecasts and are exploratory.

| Model | Early log loss / Brier / accuracy | Closing log loss / Brier / accuracy |
| --- | ---: | ---: |
| Raw market | 0.980928 / 0.584944 / 52.90% | 0.979448 / 0.583735 / 53.22% |
| Market residual | 0.981171 / 0.585147 / 52.86% | 0.979348 / 0.583718 / 53.10% |
| + picks | 0.981222 / 0.585154 / 52.95% | 0.979467 / 0.583773 / 53.13% |
| + picks + global MiniLM | 0.981224 / 0.585155 / 52.99% | 0.979457 / 0.583775 / 53.15% |
| + picks + side TF-IDF | 0.981223 / 0.585155 / 52.99% | 0.979468 / 0.583774 / 53.15% |
| + picks + team MiniLM | 0.981223 / 0.585155 / 53.01% | 0.979472 / 0.583777 / 53.13% |
| + history | 0.981127 / 0.585123 / 53.03% | 0.979368 / 0.583724 / 53.03% |
| + history + picks | 0.981176 / 0.585129 / 53.08% | 0.979497 / 0.583787 / 53.19% |
| + history + picks + side TF-IDF | 0.981178 / 0.585130 / 53.08% | 0.979499 / 0.583788 / 53.17% |

| Increment or contrast | Early Δ and 95% interval, ×10⁻⁶ | Closing Δ and 95% interval, ×10⁻⁶ |
| --- | ---: | ---: |
| Market residual vs raw market | +243.6 [−164.0, +657.2] | −100.0 [−509.5, +312.0] |
| History residual vs raw market | +198.9 [−326.0, +714.7] | −80.0 [−568.4, +388.9] |
| Add history to market residual | −44.7 [−324.2, +220.3] | +19.9 [−204.8, +236.0] |
| Add history to market and picks | −45.7 [−324.6, +219.8] | +30.3 [−191.1, +245.6] |
| Add side TF-IDF to market and picks | +1.7 [−5.5, +8.5] | +1.0 [−3.9, +6.0] |
| Add team MiniLM to market and picks | +1.5 [−16.5, +19.8] | +4.8 [−12.2, +20.9] |
| Add side TF-IDF to market, picks and history | +1.7 [−5.5, +8.5] | +1.1 [−3.8, +6.0] |

History improved on the market-only residual in one of the three validation seasons for each market. It improved pooled early log loss against that residual by 0.000045, while raw early probabilities still led. It worsened pooled closing log loss against the calibrated closing baseline by 0.000020. The hybrid with side TF-IDF improved one of three folds in each market, with a slight pooled loss increase in both. Pooled and fold-level scores for every model, raw and nested paired comparisons, and the 14-day intervals are available in the market-specific CSVs.

The market-only numerical L2 was 1 early and 0.1 closing. The market-plus-picks numerical L2 was 10 early and 1 closing. History selected L2 10 in all comparisons, and both team text blocks selected L2 1. Global MiniLM selected 1 early and 0.1 closing. History's pooled logit RMS in the final hybrid was 0.01249 early and 0.01122 closing. The numerical block contributed 0.00199 and 0.01150, and side TF-IDF contributed 0.000298 and 0.000264. The table of block corrections also records fold-level RMS, mean absolute logits and intercepts.

## Conclusion

Separating the teams recovers substantially more directional signal from article text. This improvement carries to standalone outcome prediction, and the market remains much stronger. The predefined history block and team-aware text show no reliable incremental log-loss improvement on bookmaker probabilities. The market-overlap diagnostic and the residual comparisons are consistent with substantial shared information. They leave room for useful signals that this attribution rule or these linear models could miss.

Further semantic modeling or nonlinear ML is premature on these data. The more informative next phase would label the private attribution sample and obtain timestamped odds, contemporaneous article versions and fuller result histories before specifying another development-only test. Attribution is currently unverified. Incomplete observed history affects form and rest covariates, and the odds and result availability times are proxies. The original HTML archive remains unavailable, so publication timestamps cannot establish which text version was available at prediction. Potential team identifiers outside the confidently matched names and pretraining corpus overlap also remain unverified.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pvb.advanced.team_historical
.\.venv\Scripts\python.exe -m pytest -q tests/test_advanced_team_historical.py tests/test_advanced_diagnostics.py tests/test_advanced_contextual.py
.\.venv\Scripts\python.exe -m pytest -q
```

The runner requires the local Phase 06A private review and the pinned checkpoint already in the evaluation cache. It makes no network requests for model files. The private review is stored under `results/advanced/team_historical/private/`. Attribution and embedding caches stay under `data/cache/contextual/`. Forecasts are evaluated in memory. The final run took 331.2 seconds with warm caches. MiniLM used CUDA for the initial side encoding. Early and closing diagnostics took 93.0 and 97.9 seconds. Their residual comparisons took 43.6 and 41.2 seconds. All 336 diagnostic and 192 residual candidate fits converged. The focused suite passed 33 tests and the full suite passed 239.
