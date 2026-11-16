"""Command line: run each step of the pipeline from the project folder.

    python -m pvb urls --years 2021-2026      # 1. find preview URLs in the sitemaps
    python -m pvb scrape --limit 20           # 2. download raw HTML (resumable)
    python -m pvb parse                       # 3. raw HTML -> data/interim/previews.csv
    python -m pvb odds football               # 4. download odds files
    python -m pvb features                    # 5. text features -> data/processed/features.csv
"""

import argparse
import logging
import re
import sys

import pandas as pd

from . import config


def year_range(text: str) -> list[int]:
    """'2024' -> [2024], '2021-2026' -> [2021, ..., 2026]"""
    match = re.fullmatch(r"\s*(\d{4})\s*(?:[-–]\s*(\d{4}))?\s*", text)  # accepts - and an en dash
    if not match:
        raise argparse.ArgumentTypeError(f"expected a year like 2025 or a range like 2021-2026, got {text!r}")
    first, last = int(match.group(1)), int(match.group(2) or match.group(1))
    if last < first:
        raise argparse.ArgumentTypeError(f"the range {text!r} runs backwards")
    return list(range(first, last + 1))


def season_code(text: str) -> str:
    """football-data season codes: '2526' = 2025/26."""
    if not re.fullmatch(r"\d{2}(\d{2})", text) or int(text[2:]) != (int(text[:2]) + 1) % 100:
        raise argparse.ArgumentTypeError(f"expected a season like 2526 (= 2025/26), got {text!r}")
    return text


def cmd_urls(args) -> None:
    from .http import PoliteSession
    from .sportsmole import URLS_CSV, collect_preview_urls

    found = collect_preview_urls(PoliteSession(), args.years, args.sports)
    if found.empty:
        sys.exit("No previews found for those years; the URL list was left unchanged.")
    if URLS_CSV.exists():  # add to the list instead of replacing it
        found = pd.concat([pd.read_csv(URLS_CSV), found]).drop_duplicates("article_id", keep="last")
    found = found.sort_values("article_id")
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    found.to_csv(URLS_CSV, index=False)
    print(found.groupby(["sitemap_year", "sport"]).size().unstack(fill_value=0))
    print(f"\n{len(found)} preview URLs in {URLS_CSV}")


def cmd_scrape(args) -> None:
    from .http import PoliteSession
    from .sportsmole import URLS_CSV, download_previews

    if not URLS_CSV.exists():
        sys.exit("No URL list yet. Run `python -m pvb urls` first.")
    urls = pd.read_csv(URLS_CSV)
    if args.sport:
        urls = urls[urls["sport"] == args.sport]
    if args.years:
        urls = urls[urls["sitemap_year"].isin(args.years)]
    saved = download_previews(PoliteSession(), urls, limit=args.limit)
    print(f"saved {saved} pages to {config.SPORTSMOLE_RAW}")


def cmd_parse(args) -> None:
    from .parse import PREVIEWS_CSV, parse_all

    df = parse_all()
    if df.empty:
        sys.exit(f"No saved pages in {config.SPORTSMOLE_RAW}. Run `python -m pvb scrape` first.")
    has_pick = df["pick_result"].notna() | df["pick_winner"].notna()
    print(f"{len(df)} previews parsed, {has_pick.sum()} with a readable pick -> {PREVIEWS_CSV}")


def cmd_odds(args) -> None:
    from . import odds
    from .http import PoliteSession

    if args.sport == "football":
        if args.years:
            sys.exit("Football uses --seasons (e.g. --seasons 2425 2526), not --years.")
        odds.download_football(PoliteSession(), seasons=args.seasons or config.FOOTBALL_SEASONS, refresh=args.refresh)
    else:
        if args.seasons:
            sys.exit("Tennis uses --years (e.g. --years 2024-2026), not --seasons.")
        odds.download_tennis(PoliteSession(), years=args.years or config.TENNIS_YEARS, refresh=args.refresh)


def cmd_features(args) -> None:
    from .features import FEATURES_CSV, build

    df = build()
    print(f"{len(df)} rows -> {FEATURES_CSV}")


def cmd_todo(args) -> None:
    sys.exit(f"`{args.command}` is not built yet. See docs/plan.md for the plan.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pvb", description="Can pundits beat the bookies? Data pipeline.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    # -v also works after the command (`python -m pvb scrape -v`); SUPPRESS keeps it from undoing an earlier -v
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="show debug messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("urls", parents=[common], help="find preview URLs in Sports Mole's sitemaps")
    p.add_argument("--years", type=year_range, default=year_range("2021-2026"), help="e.g. 2025 or 2021-2026")
    p.add_argument("--sports", nargs="+", default=list(config.SPORTS), choices=config.SPORTS)
    p.set_defaults(func=cmd_urls)

    p = sub.add_parser("scrape", parents=[common], help="download preview pages (skips pages already saved)")
    p.add_argument("--sport", choices=config.SPORTS)
    p.add_argument("--years", type=year_range, help="only previews from these sitemap years, e.g. 2025")
    p.add_argument("--limit", type=int, help="stop after this many pages (good for testing)")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("parse", parents=[common], help="saved pages -> data/interim/previews.csv")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("odds", parents=[common], help="download football-data or tennis-data files")
    p.add_argument("sport", choices=config.SPORTS)
    p.add_argument("--seasons", nargs="+", type=season_code, help="football seasons, e.g. 2425 2526")
    p.add_argument("--years", type=year_range, help="tennis years, e.g. 2024-2026")
    p.add_argument("--refresh", action="store_true", help="download again even if saved")
    p.set_defaults(func=cmd_odds)

    p = sub.add_parser("features", parents=[common], help="previews -> data/processed/features.csv")
    p.set_defaults(func=cmd_features)

    for name, text in [("match", "match previews to odds (to do)"), ("analyse", "run the tests (to do)")]:
        sub.add_parser(name, parents=[common], help=text).set_defaults(func=cmd_todo)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
