# Data

Most of this folder stays on your own computer. Git ignores it, apart from this file and the text-free tables in `processed/`.

| Folder | What goes in it | Made by | On GitHub |
| --- | --- | --- | --- |
| `raw/sportsmole/` | One HTML file per preview (`<article_id>.html`) and `manifest.csv`, the download log | `python -m pvb scrape` | No |
| `raw/odds/football/` | football-data.co.uk CSVs, one per season and league (`2526_E0.csv`) | `python -m pvb odds football` | No |
| `raw/odds/tennis/` | tennis-data.co.uk files, one per year and tour (`2025_ATP.xlsx`) | `python -m pvb odds tennis` | No |
| `raw/polymarket/` | Polymarket tennis prices (part 2) | `pvb/polymarket.py` | No |
| `interim/` | `preview_urls.csv` and `previews.csv` (one row per preview, with the full text) | `urls`, `parse` | No |
| `processed/` | `picks.csv`, `features.csv`, `matched.csv`: the tables used in the analysis, numbers only | `parse`, `features`, `match` | Yes |

Rules:

- Never edit files in `raw/`. If something is wrong, fix the parser and run it again.
- Keep all data files inside `data/`, so scraped text never reaches GitHub. Sports Mole's terms forbid republishing its content, so we publish only numbers and results.
- Never put text in `processed/`: everything there is public. `tests/test_shared_data.py` fails if a table there has a text column or a cell longer than ten words.
- Only the scraper runs the full scrape. Everyone else copies `data/` from the team's shared drive.
- The full text (`previews_full_text.zip`) is shared privately, only with the team and the examiner. German copyright law allows sharing a text-mining corpus with a closed research group and with someone checking the research (§ 60d UrhG), but not publishing it.
