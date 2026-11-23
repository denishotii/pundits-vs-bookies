# Data

Most of this folder stays on your own computer. Git ignores it, apart from this file and the text-free tables in `processed/`.

| Folder | What goes in it | Made by | On GitHub |
| --- | --- | --- | --- |
| `raw/sportsmole/` | One HTML file per preview (`<article_id>.html`) and `manifest.csv`, the download log | `python -m pvb scrape` | No |
| `raw/odds/football/` | football-data.co.uk CSVs, one per season and league (`2526_E0.csv`) | `python -m pvb odds football` | No |
| `raw/odds/tennis/` | tennis-data.co.uk files, one per year and tour (`2025_ATP.xlsx`) | `python -m pvb odds tennis` | No |
| `raw/polymarket/` | Polymarket tennis prices (part 2) | `pvb/polymarket.py` (to do: it downloads but doesn't save yet) | No |
| `interim/` | `preview_urls.csv` and `previews.csv` (one row per preview, with the full text) | `urls`, `parse` | No |
| `processed/` | `picks.csv`, `features.csv`, `matched_football.csv`, `matched_tennis.csv`: the tables used in the analysis, with no article text | `parse`, `features`, `match` | Yes |

Rules:

- Never edit files in `raw/`. If something is wrong, fix the parser and run it again.
- Keep all data files inside `data/`, so scraped text never reaches GitHub. Sports Mole's terms forbid republishing its content, so we publish no article text, only facts and numbers taken from it: IDs, URLs, dates, author and team or player names, the parsed pick, counts and odds.
- Never put text in `processed/`: everything there is public. `tests/test_shared_data.py` fails if a table there has a text column or a cell longer than ten words.
- Only the scraper runs the full scrape. Everyone else copies `data/` from the team's shared drive.
- The full text (`previews_full_text.zip`) is shared privately, only with the team and the examiner. German copyright law allows sharing a text-mining corpus with a closed research group and with someone checking the research, but not publishing it, and only until the research and its check are finished (§ 60d(4) UrhG). We keep our own copies secured for as long as the research needs them (§ 60d(5)).

## The matched tables

One row per preview that was matched to a game. They join to `picks.csv` and `features.csv` on `article_id`.

- `published`, `kickoff`: UTC. `date` (tennis) is tennis-data's date, which for some US and Asian events is one day after the real match day.
- `side_a`, `side_b`: the names as Sports Mole writes them. `home_team`, `away_team`, `player_a`, `player_b`: the names in the odds files.
- `result`, `pick_result`: H, D or A. `a_won`: player a won. `pick_side`: the player the pick names, a or b.
- `early_odds_*`: the average odds on the Friday afternoon (weekend games) or Tuesday afternoon (midweek games) before the game. For most previews this is after the preview came out.
- `closing_odds_*`: the average of the last odds before the start. `best_closing_odds_*`: the highest of them across bookmakers.
- `status` (tennis): Completed or Retired. Walkovers are left out, because no match was played.
- The odds files were downloaded on 16 November 2026. The study ends on 14 November 2026 (`STUDY_END` in `pvb/config.py`).
