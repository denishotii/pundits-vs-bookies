# Phase 08 historical evaluation protocol

## Fixed comparisons

The primary comparison is **F3 minus F2** in multiclass log loss. It tests the increment from side-aware article text conditional on market probabilities and explicit picks. Secondary comparisons are F2 minus F1, F1 minus F0, and F3 minus F0. Lower loss and negative differences favor the first model. The original eligible cohort is primary. Report the approved integrity-screened sensitivity cohort alongside it.

| Model | Correction to log market probabilities | Early L2 | Closing L2 |
| --- | --- | --- | --- |
| F0 | Raw probabilities from proportional margin removal | — | — |
| F1 | Standardized log market probabilities | 1 | 0.1 |
| F2 | Standardized log market probabilities and explicit picks | 10 | 1 |
| F3 | F2 inputs plus side TF-IDF | 10 numerical, 1 text | 1 numerical, 1 text |

## Refit and score

1. Verify the freeze, exact dependency versions, Phase 07 development input and attribution fingerprints, and approved audit policy. Lock the input file hashes before fitting.
2. Apply the existing market-specific football eligibility rules. Early uses the Tuesday/Friday 15:00 Europe/London proxy and closing uses kickoff. Publication must precede the cutoff under the source precision rules. F0–F3 share the market-specific eligible sample, including the article and pick requirements.
3. Separately by market, refit F1–F3 once on eligible 2122–2425 rows with an outcome release strictly before the earliest feature-eligible 2526 prediction. A release at the cutoff is purged. Check that 2627 follows 2526 and reuse the fitted transforms and coefficients for both. Standardization, vocabulary and IDF use the retained development rows alone. The release check uses the frozen proxy of kickoff plus three hours, or 03:00 UTC the next day for date-only kickoffs. Actual availability requires independent verification.
4. Generate fixed forecasts for both seasons and markets before reading holdout outcomes. Join valid H/D/A outcomes by unique article ID and report missing or invalid outcomes. Within each market, season and cohort, score the four models on exactly the same IDs. Use the same forecasts for the original and screened cohorts. Report counts without scores for an empty screened cohort.
5. Report 2526 as historical test and 2627 as historical check, separately for each market. Report the first and last scored kickoffs and indicate if 2627 coverage is incomplete. Never refit, select a winner for deployment, change thresholds, or revise the cohort rules using these scores.

Primary loss is the mean natural-log loss with a 1e-15 realized-probability floor. Report mean multiclass Brier score (sum across H/D/A), accuracy (ties in H/D/A order), class counts, recall, precision, per-class conditional log loss and one-vs-rest Brier. Report calibration in ten equal-width one-vs-rest bins, including the count, mean forecast and observed rate in each occupied bin. Undefined class metrics remain missing.

Report paired per-match log-loss differences privately and summarize their mean, dispersion, quantiles and fraction improved. Use 3,000 resamples of occupied, non-overlapping seven-day blocks of UTC fixture dates with replacement, grouped within season using the existing Monday 1970-01-05 origin and seed 60108. Weight matches within each replicate by their multiplicity and retain original season weights. Report 2.5th and 97.5th percentiles and repeat with fourteen-day blocks. These intervals are conditional on the fixed forecasts and observed cohorts. Interpret secondary contrasts, markets and screened analyses with their multiplicity in mind. There is no pooled holdout result or holdout-driven selection.

Aggregate holdout results from the original project are already known. This protocol is prespecified for the newly frozen comparisons, and both seasons remain historical assessments. A prospective study would lock the same choices before outcomes arise and archive the exact odds, article versions, picks and outcome-release times as they become available.

## Article-integrity sensitivity

The original HTML archive is unavailable. The original cohort retains the original eligibility rules. The sensitivity cohort removes a fixture when any of its market's approved exclusion rules applies. Freeze the evidence table and rule lists before examining either season's scores. Confirmed outcome information requires evidence and is always excluded from the screened cohort. A later `dateModified` alone is insufficient to flag a content modification or exclude a fixture. Text matching across later downloads does not establish availability at prediction time.

The separate UTF-8 audit CSV has one row per `(article_id, market)`. Use `early` or `closing` for market and adjudicate against that market's prediction time. Required fields are:

| Field | Accepted values |
| --- | --- |
| `article_id`, `market` | Nonempty matched ID and named market |
| `metadata_status` | `verified`, `unverifiable`, `missing` |
| `timestamp_conflict` | `yes`, `no`, `unknown` |
| `potential_post_prediction_content_modification` | `yes`, `no`, `unknown` |
| `confirmed_outcome_information_contamination` | `yes`, `no`, `unknown` |
| `evidence_reference` | Private provenance reference, required for confirmed contamination |

An absent row has missing metadata and unknown flags. Unknown flags never equal affirmative evidence. Policy choices beyond confirmed contamination remain pending approval and may include `metadata_missing`, `metadata_unverifiable`, `timestamp_conflict`, and `potential_post_prediction_content_modification`. The manifest records approval, overlapping reason counts, unique excluded matches, audit coverage, and original and screened sample fingerprints for each market and season. Preserve the original inputs.

## Commands

Run from the repository root in PowerShell.

```powershell
.\.venv\Scripts\python.exe -m pvb.advanced.final_evaluation
.\.venv\Scripts\python.exe -m pvb.advanced.final_evaluation --verify-development
.\.venv\Scripts\python.exe -m pvb.advanced.final_evaluation --integrity-audit data/final_integrity/article_audit.csv
```

The third command validates the audit schema and prints the freeze, audit and input hashes. Once approved, save `data/final_integrity/approval.json` with `status` set to `approved`, `approved_by`, a timezone-aware ISO 8601 `approved_at`, `freeze_sha256`, `audit_sha256`, the `input_sha256` object containing `matches` and `previews`, and `exclude_if_any` with an explicit rule array for each of `early` and `closing`. Include `confirmed_outcome_information_contamination` in each array. The hashes must match the plan exactly. A changed snapshot requires a new plan and approval before seeing scores.

```powershell
.\.venv\Scripts\python.exe -m pvb.advanced.final_evaluation --integrity-audit data/final_integrity/article_audit.csv --integrity-approval data/final_integrity/approval.json
.\.venv\Scripts\python.exe -m pvb.advanced.final_evaluation --execute-holdouts --integrity-audit data/final_integrity/article_audit.csv --integrity-approval data/final_integrity/approval.json
```

Only the final command scores historical outcomes. It writes a new ignored directory at `results/advanced/final/evaluation/` with aggregate CSVs, private per-match differences and a manifest. Raw forecasts and fitted weights remain in memory.
