# Data

Most of this folder stays on your own computer. Git ignores it, apart from this file and the text-free tables in `processed/`.

| Folder | What goes in it | Made by | On GitHub |
| --- | --- | --- | --- |
| `raw/sportsmole/` | One HTML file per preview (`<article_id>.html`) and `manifest.csv`, the download log | `python -m pvb scrape` | No |
| `raw/odds/football/` | football-data.co.uk CSVs, one per season and league (`2526_E0.csv`) | `python -m pvb odds football` | No |
| `raw/odds/tennis/` | tennis-data.co.uk files, one per year and tour (`2025_ATP.xlsx`) | `python -m pvb odds tennis` | No |
| `raw/dictionaries/` | `inqtabs.txt`, the Harvard IV dictionary (General Inquirer) | Download by hand from [inquirer.sites.fas.harvard.edu](https://inquirer.sites.fas.harvard.edu/inqtabs.txt) | No |
| `raw/polymarket/` | Polymarket tennis prices (part 2) | `pvb/polymarket.py` (to do: it downloads but doesn't save yet) | No |
| `interim/` | `preview_urls.csv` and `previews.csv` (one row per preview, with the full text) | `urls`, `parse` | No |
| `processed/` | `picks.csv`, `matched_football.csv`, `matched_tennis.csv`, `features.csv`, `dictionary_review.csv`: the tables used in the analysis, with no article text | `parse`, `match`, `features` | Yes |

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

## The text features

`features.csv` has one row per preview and joins on `article_id`. Side a is the home team in football and the first-named player in tennis. Each column is explained in `pvb/features.py`:

- `words_a`, `words_b`: how many lower-case words are about each side. About a quarter of the words are about neither side (see `pvb/features.py`).
- `tone_a`, `tone_b`, `tone_diff`: (positive − negative) / (positive + negative) words about each side, from −1 to 1. The dictionary is Harvard IV adapted to sport (`pvb/sentiment.py`).
- `absence_a`, `absence_b`, `absence_diff`: absence words (injured, suspended, without, ...) per 100 words of team news about each side. Football only.
- `uncertainty_per_1000`: hedging words ("could", "tight") per 1,000 words.
- `nb_home_vs_draw`, `nb_away_vs_draw` (football), `nb_a_vs_b` (tennis): Naive Bayes log odds, such as log P(home win) − log P(draw), given the words. Empty for previews that weren't matched to a game, because the model learns from results.

`dictionary_review.csv` is the review behind the tone dictionary: every Harvard IV word (and every word we added) found in at least 1% of the previews before the 2025/26 season, with Harvard's sign, ours, and the reason for any change.
