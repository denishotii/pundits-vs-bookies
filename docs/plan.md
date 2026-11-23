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
| The pick ("We say: Fulham 1-0 Burnley") | Regex |
| Tone towards each team | Sentiment dictionary on sentences naming that team |
| Team news (injured, suspended, doubtful) | Word counts in the Team News section |
| Uncertainty ("could", "tight") | Word list |
| Naive Bayes probabilities | Trained on earlier seasons |
| Recycled text (optional) | Jaccard similarity |

## Strategy

1. **Past data is the main test.** Football: train on 2021/22 to 2024/25, test once on 2025/26. Then run the same frozen model once on 2026/27 so far (676 games up to 14 November), a second check on games only weeks old. Tennis: train on 2024 and 2025, test once on 2026.
2. **Did the odds react?** Previews come out about 2 days before kick-off (median 50 hours). We compare football-data's early odds with the closing odds, the last odds before kick-off. The early odds are taken on the Friday afternoon (weekend games) or Tuesday afternoon (midweek games), which for most previews is after the preview came out. So the test asks whether the odds keep moving towards the pick after that snapshot, and we report Sunday and Monday games, where the snapshot usually comes first, separately. Football only: tennis-data has closing odds only.
3. **Upcoming games are a live check** on the weekends of 21 November, 28 November and 5 December (about 180 matches). This is for the demo, not the evidence.

## Pilot results

| | Football (36 previews) | Tennis (60 previews) |
| --- | --- | --- |
| Pick agrees with the bookmaker favourite | 53% | 90% |
| Pick is right | 36% | 82% |
| Backing the favourite is right | 50% | 82% |

In the pilot, football pundits predicted a draw in 42% of previews, while the market never made a draw the favourite.

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
