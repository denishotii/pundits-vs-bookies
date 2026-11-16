# Can Pundits Beat the Bookies?

Do expert match previews know something the betting odds don't? We scrape thousands of football and tennis previews from [Sports Mole](https://www.sportsmole.co.uk), turn their text into numbers, and test whether they add anything to what the closing odds (the last odds before kick-off) already say.

MSc seminar project for *Webscraping & Textual Analysis in Python*, KU Eichstätt-Ingolstadt, winter term 2026/27.

## The idea

Every Sports Mole preview ends with a prediction, such as "We say: Brighton & Hove Albion 2-2 Leeds United" or "We say: Medvedev to win in three sets". Betting odds show what the market expects for the same match.

- If the text adds information beyond the odds, the market missed something.
- If it doesn't, the market already knows what the pundits know.

Both answers are a result. Part 2 asks whether Polymarket, a prediction market, overreacts when a tennis favourite loses the first set.

The full plan, with scope, strategy and timeline, is in [docs/plan.md](docs/plan.md).

## Pipeline

| Step | Command | Output | Status |
| --- | --- | --- | --- |
| 1. Find previews in the sitemaps | `python -m pvb urls` | `data/interim/preview_urls.csv` | Working |
| 2. Download the pages | `python -m pvb scrape` | `data/raw/sportsmole/*.html` | Working |
| 3. Parse pick, sections, date | `python -m pvb parse` | `data/interim/previews.csv` | Working |
| 4. Download odds | `python -m pvb odds football` (or `tennis`) | `data/raw/odds/` | Working |
| 5. Text features | `python -m pvb features` | `data/processed/features.csv` | Started |
| 6. Match previews to games | `python -m pvb match` | `data/processed/matched.csv` | To do |
| 7. Analysis | `python -m pvb analyse` | Tables and charts | To do |
| Part 2: Polymarket prices | `pvb/polymarket.py` | | Downloads work, analysis to do |

## Getting started

Open Anaconda Prompt (Windows) or Terminal (Mac) and get the code:

```bash
git clone https://github.com/<owner>/pundits-vs-bookies.git
cd pundits-vs-bookies
```

No git? Use [GitHub Desktop](https://desktop.github.com): File > Clone repository.

Create the environment once, and set a contact for the scraper's User-Agent:

```bash
conda create -n pvb python=3.12 spyder
conda activate pvb
pip install -r requirements.txt
conda env config vars set PVB_CONTACT="our-team-email@example.com"
conda activate pvb
```

Without Anaconda: `python3 -m venv .venv`, then `source .venv/bin/activate` and `pip install -r requirements.txt`.

Check that everything works, then try the pipeline on a small sample:

```bash
python -m pytest
python -m pvb urls --years 2026
python -m pvb scrape --sport football --limit 20
python -m pvb parse
python -m pvb odds football --seasons 2526
```

Run every command from the project folder.

**Spyder:** start it from the activated environment (`conda activate pvb`, then `spyder`). Add the project folder under Tools > PYTHONPATH manager. Then you can write, for example, `from pvb.parse import parse_all`.

**The full scrape:** only the scraper runs it. Sports Mole previews every competition, not just our six leagues, so 2021 to 2026 is about 45,000 pages. At one page every 2 seconds that is about 25 hours. You can stop and restart it at any time; pages already saved are skipped. Start with `--years 2025-2026` so the others can begin. Everyone else copies `data/` from the shared drive.

## Project structure

```
pundits-vs-bookies/
├── pvb/                  the Python package
│   ├── __main__.py       command line: python -m pvb ...
│   ├── config.py         folders, sources, seasons, scraping settings
│   ├── http.py           polite downloads: delay, retries, robots.txt
│   ├── sportsmole.py     find previews in the sitemaps, save raw HTML
│   ├── parse.py          raw HTML -> one row per preview
│   ├── odds.py           football-data and tennis-data downloads
│   ├── features.py       text features (started)
│   ├── matching.py       previews -> games (to do)
│   ├── polymarket.py     part 2: Polymarket tennis prices
│   └── analysis.py       the three tests (to do)
├── tests/                tests, no internet needed
├── data/                 not on GitHub, see data/README.md
├── docs/plan.md          the project plan
└── notebooks/            exploration, as .py files with # %% cells
```

## Scraping rules

- One request every 2 seconds per site, and nothing that robots.txt disallows.
- The scraper names itself in its User-Agent, using `PVB_CONTACT`.
- Each page is downloaded once and saved. Parsing always runs on the saved copy.
- Scraped text never goes on GitHub. Sports Mole's terms forbid republishing its content, so we publish only numbers and results.

## Team

| Role | Owns | Hands over | Person |
| --- | --- | --- | --- |
| Scraper | `sportsmole.py`, `parse.py` | `previews.csv` | |
| Odds and matching | `odds.py`, `matching.py` | `matched.csv` | |
| Text analysis | `features.py` | `features.csv` | |
| Part 2 (Polymarket) | `polymarket.py` | Part 2 results | |
| Analysis and writing | `analysis.py`, proposal, slides | Proposal and slides | |

## Working together

- Pull before you start. Work on a branch named after your part, such as `matching/football`, and open a pull request.
- Run `python -m pytest` before you push. GitHub runs the tests on every push too.
- Keep all data files inside `data/`; git ignores that folder. To use a shared copy instead, set `PVB_DATA` to its path.
- New to git? GitHub Desktop does all of this with buttons. Don't use GitHub's "Upload files" button: it ignores `.gitignore` and could publish scraped text.

Deadlines: submission on 13 December 2026, presentation on 18 December 2026.

## License

The code is MIT licensed.
