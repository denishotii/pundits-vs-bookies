# Data

Everything in this folder stays on your own computer. Git ignores it, apart from this file.

| Folder | What goes in it | Made by |
| --- | --- | --- |
| `raw/sportsmole/` | One HTML file per preview (`<article_id>.html`) and `manifest.csv`, the download log | `python -m pvb scrape` |
| `raw/odds/football/` | football-data.co.uk CSVs, one per season and league (`2526_E0.csv`) | `python -m pvb odds football` |
| `raw/odds/tennis/` | tennis-data.co.uk files, one per year and tour (`2025_ATP.xlsx`) | `python -m pvb odds tennis` |
| `raw/polymarket/` | Polymarket tennis prices (part 2) | `pvb/polymarket.py` |
| `interim/` | `preview_urls.csv` and `previews.csv` (one row per preview) | `urls`, `parse` |
| `processed/` | `features.csv`, `matched.csv`, the tables used in the analysis | `features`, `match` |

Rules:

- Never edit files in `raw/`. If something is wrong, fix the parser and run it again.
- Keep all data files inside `data/`. Git ignores this folder, so scraped text never reaches GitHub. Sports Mole's terms forbid republishing its content, so we publish only numbers and results.
- Only the scraper runs the full scrape. Everyone else copies `data/` from the team's shared drive.
