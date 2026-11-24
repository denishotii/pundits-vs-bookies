# Project Instructions

## Project objective

Extend the existing Pundits vs Bookies research project with advanced NLP and market-aware machine learning. Investigate whether textual information improves bookmaker-implied probability forecasts.

Scientific validity, reproducibility, and leakage prevention take priority over achieving better numerical results.

## Context and token efficiency

- Read only files directly relevant to the current task.
- Do not recursively inspect the entire repository unless necessary.
- Never print or load complete CSV datasets into the model context.
- Never read the full contents of `data/interim/previews.csv` through terminal output or file inspection tools.
- Process large datasets programmatically using pandas or NumPy.
- Inspect datasets using column names, shapes, dtypes, aggregate statistics, and at most 3–5 sampled rows.
- Truncate long text fields when displaying samples.
- Never inspect all raw HTML files individually. Use Python scripts to extract metadata and aggregate results.
- Limit command output to relevant summaries and errors.
- Avoid repeatedly reading unchanged files.
- Do not generate unnecessary documentation or verbose comments.

## Development constraints

- Use Python 3.11 compatibility.
- Implement new research components under `pvb/advanced/`.
- Keep new experimental results under `results/advanced/`.
- Preserve the existing baseline implementation.
- Do not modify existing modules unless explicitly requested.
- Do not install additional dependencies without justification.
- Do not commit, push, merge, or switch branches without explicit permission.
- Never publish raw scraped text, HTML files, or private datasets.
- Do not start web scraping unless explicitly requested.

## Scientific methodology

- Use chronological train/validation/test splits.
- Never use future observations during preprocessing or training.
- Fit all learned preprocessing transformations on training data only.
- Use explicit feature allowlists.
- Never use actual match scores or outcomes as input features.
- Keep early-odds and closing-odds experiments separate.
- Select models and hyperparameters using training-period validation only.
- Do not optimize against the historical test or check seasons.
- Treat article modification timestamps as an unresolved leakage risk until audited.
- Compare all models on identical evaluation samples.
- Use Log Loss as the primary evaluation metric and Brier Score as a secondary metric.
- Never fabricate experimental results.

## Task completion

- Implement only the requested scope.
- Add focused unit tests for new functionality.
- Run relevant tests.
- Report changed files, test results, assumptions, and unresolved issues concisely.

## Context handover

Maintain `docs/advanced_ml_progress.md` as a concise project handover document.

At the beginning of each task, read this document if it exists.

At the end of each completed phase, update it with:
- Completed components and their interfaces.
- Important architectural decisions.
- Test and experiment results.
- Known limitations and unresolved issues.
- Recommended next step.

Keep this document under 150 lines. Update existing information instead of accumulating repetitive entries.

Use programmatic dataset inspection. Never dump large CSV files, raw HTML, full articles, embeddings, or complete prediction tables into the conversation context. Limit terminal output to compact summaries.

Run commands using the existing local Python virtual environment. Do not recreate the environment or reinstall dependencies unnecessarily.