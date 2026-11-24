# Project plan

**Question: does the text of expert match previews know something the betting odds don't?**

If it does, a bettor could use the text to beat the bookies. If it doesn't, the market already prices in what experts write. Both answers are a valid result. It is the same test Tetlock (2007) ran with news articles and stock prices.

## Scope

- **Football (core):** Premier League, Championship, Bundesliga, La Liga, Serie A, Ligue 1, seasons 2021/22 to 2025/26, plus 2026/27 up to 14 November 2026. About 2,300 league games per season; 99% of them have a matched preview (2,271 of 2,304 in 2025/26, 676 of 680 so far in 2026/27).
- **Tennis (second test):** ATP and WTA previews, 2024 to 2026: 907 previews, 854 matched to a match with odds.
- **Part 2 (Polymarket, from the TennisEdge draft):** when a tennis favourite loses the first set, does the price on Polymarket, a prediction market, fall further than it should?
- **Out:** live-score sites (they block scrapers or forbid it), real betting, apps, Reddit or X, cup and international football.

## Data (checked 16 November 2026)

| Source | What we get | How |
| --- | --- | --- |
| Sports Mole | Preview text, the "We say" pick, team news, publish time | Scrape via yearly sitemaps |
| football-data.co.uk | Results, earlier and closing odds | CSV download |
| tennis-data.co.uk | Results, set scores, closing odds | Excel download |
| Polymarket API | Tennis prices before and during play, set 1 markets | Public API |

Sports Mole previews every competition, not just our six leagues: about 45,000 football and tennis previews from 2021 to 2026. Matching keeps 13,157 of them (12,303 football, 854 tennis).

## Text analysis

| Feature | How |
| --- | --- |
| The pick ("We say: Fulham 1-0 Burnley") | Regex (`parse.py`) |
| Tone towards each side | Harvard IV adapted to sport (`sentiment.py`), on the sentences about each side |
| Team news (injured, suspended, without) | Absence words per 100 words of team news about each side |
| Uncertainty ("could", "tight") | Word list |
| Naive Bayes log odds | Words marked with the side they are about; trained on the training seasons only, cross-fitted by season |
| Recycled text (optional) | Jaccard similarity; not done |

Which sentences are about which side: a sentence that names one side is about it, and so are the sentences after it that name neither side. A sentence naming both sides is about neither, and each section starts afresh, so about a quarter of the words (most of the introduction and the verdict) count for neither side. Names include each team's nickname, manager and players, learned per season from the previews (`features.py`). In a sample of real previews, about 97% of football sentences and 95% of tennis sentences credited to a side were right.

Why adapt the dictionary: Loughran and McDonald (2011) showed that a general dictionary misreads the words of a field. Harvard IV counts "against", "points" and "clash" as negative and "home" as positive, and these words are in nearly every preview. It also lacks "win" and "victory". So, as Loughran and McDonald did for annual reports, we went through every Harvard word in at least 1% of the previews from before the test season, dropped those without tone in sport, and added the sports words Harvard lacks. The review, word by word with the reason, is in `data/processed/dictionary_review.csv`. One person did it; a second person coding the same words independently, and reporting how often the two agree, would make it stronger.

## Strategy

1. **Past data is the main test.** Football: train on 2021/22 to 2024/25, test once on 2025/26. Then run the same frozen model once on 2026/27 so far (676 games up to 14 November), a second check on games only weeks old. Tennis: train on 2024 and 2025, test once on 2026.
2. **Did the odds react?** Previews come out about 2 days before kick-off (median 50 hours). We compare football-data's early odds with the closing odds, the last odds before kick-off. The early odds are taken on the Friday afternoon (weekend games) or Tuesday afternoon (midweek games), which for most previews is after the preview came out. So the test asks whether the odds keep moving towards the pick after that snapshot. We also report the games where the preview came out after the snapshot (about 2 in 10) separately, because there the move includes the reaction to the preview itself. Football only: tennis-data has closing odds only.
3. **Upcoming games are a live check** on the weekends of 21 November, 28 November and 5 December (about 180 matches). This is for the demo, not the evidence.

## First results (all matched games)

| | Football (12,302 games) | Tennis (853 matches) |
| --- | --- | --- |
| Pick agrees with the bookmaker favourite | 62% | 87% |
| Pick is right | 46% | 72% |
| Backing the favourite is right | 53% | 73% |

Football pundits pick a draw in 31% of previews, but only 25% of games end in one, and the market almost never makes a draw the favourite. The three tests are in [results/](../results/README.md), made by `python -m pvb analyse`.

The text measures are not noise. On the training seasons, the tone towards each side correlates 0.25 with the odds and the Naive Bayes log odds 0.52. Yet adding them to the odds and the pick doesn't predict the 2025/26 results any better (test 1), nor 2026/27 so far, nor tennis 2026. So far, what the previews know, the odds already know.

## Roles

Five roles, one per person: scraper, odds and matching, text analysis, part 2 (Polymarket), analysis and writing. Who owns which files is in the [README](../README.md#team).

## Timeline

| When | What |
| --- | --- |
| 17 Nov | Agree topic and roles |
| 18–22 Nov | Full scrape finished (about 25 hours of downloading) |
| 21 Nov – 7 Dec | Live check, three weekends |
| 23–29 Nov | Matching and text features |
| 30 Nov – 6 Dec | Analysis, backtest, part 2 |
| 7–13 Dec | Proposal and slides |
| **13 Dec** | **Submit by email** |
| 18 Dec | Presentation |

## Risks

- The text may add nothing beyond the odds. We report that; it is a real finding.
- Wrong matches: the same fixture repeats every season, so we match on the publish date too.
- Sports Mole's terms forbid copying or republishing its content. We keep the texts private and publish only numbers. German law (§ 60d UrhG) allows text and data mining for non-commercial research.
