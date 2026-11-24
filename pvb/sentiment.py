"""Positive and negative words for match previews: Harvard IV, adapted to sport.

The course's lesson from Loughran and McDonald (2011) applies to sport too: a
general dictionary misreads the words of a field. Harvard IV counts "against",
"points", "club" and "clash" as negative and "home", "back" and "games" as
positive, and these words are in nearly every preview. It also lacks "win",
"victory" and "unbeaten", which are in 80%, 73% and 40% of them.

So we did what Loughran and McDonald did for annual reports. We went through
every Harvard word that appears in at least 1% of the previews published before
the first test season (July 2025), and dropped the ones that carry no tone in a
preview (NOT_TONE, with the reason). Then we added the sports words Harvard
lacks, or gets the wrong way round (SPORT_POSITIVE, SPORT_NEGATIVE). Words that
Harvard lists as both positive and negative are dropped as well.
`python -m pvb features` writes this review to data/processed/dictionary_review.csv:
one row per word, with how often it appears and what we decided.

Harvard lists base forms in American spelling ("lose", "favor"). `forms` adds the
regular endings ("loses", "losing"), as Loughran and McDonald list inflections
instead of stemming, and `harvard` adds British spellings ("favour").

Harvard IV is the General Inquirer's word table (inqtabs.txt). Download it once
by hand from config.HARVARD_IV into data/raw/dictionaries/. Our scraper won't
fetch it, because the site has no robots.txt (it answers "access denied"), which
PoliteSession treats as "keep out".
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd

from . import config

HARVARD_FILE = config.RAW / "dictionaries" / "inqtabs.txt"
REVIEW_BEFORE = 2025  # the review looks only at seasons that started before 2025/26, the first test season
REVIEW_MIN_SHARE = 0.01  # and at the words in at least 1% of them

# Harvard words that carry no tone in a match preview, by reason. Each word also
# drops its endings (see `forms`): "open" drops "opening" and "opened".
NOT_TONE = {
    "describes the play, the venue or the fixture": {
        "advantage", "against", "attack", "back", "ball", "bitter", "box", "break",
        "clash", "club", "compete", "competition", "competitive", "cross", "deadlock",
        "defend", "defender", "defensive", "defensively", "division", "foe", "forward",
        "free", "friendlies", "friendly", "front", "game", "home", "kick", "meet",
        "meeting", "offensive", "open", "opponent", "opposing", "opposition", "point",
        "qualifying", "return", "round", "save", "serve", "service", "shot", "spot",
        "stick", "strike", "striking", "struck", "throw", "training", "travel",
    },
    # a result or a place in the table: good for one side, bad for the other
    "a result, with no direction": {
        "aggregate", "beat", "broke", "cancel", "crush", "defeated", "defeating",
        "dominate", "edge", "hit", "lead", "lie", "overturn", "race", "rival", "run",
        "runaway", "share", "shared", "shock", "split", "spoils", "stalemate", "straight",
        "thrash", "upset",
    },
    # every side wants these before a game, so they say nothing about either
    "what every side wants": {
        "alive", "avoid", "battle", "catch", "challenge", "challenging", "chase", "eager",
        "fight", "fought", "hope", "hopeful", "hopefuls", "hunt", "keen", "need",
        "opportunity", "push", "safe", "safety", "secure", "survival", "survive",
        "tempted", "trophy", "try", "vying", "wait", "wary",
    },
    # mostly "serious injury"; the absence words measure injuries
    "about injuries (absence words)": {"serious"},
    "tone depends on the phrase (hot form, danger man)": {
        "danger", "dangerous", "exploit", "hot", "ride",
    },
    "an everyday word, no tone here": {
        "ability", "able", "accrue", "actually", "admit", "allow", "appoint", "approach",
        "assist", "assistant", "availability", "aware", "belt", "bit", "board", "bound",
        "call", "capital", "charge", "clear", "clearly", "close", "closely", "complete",
        "completely", "concerning", "consider", "contend", "content", "courtesy", "create",
        "creative", "cut", "dead", "deal", "desire", "displace", "disposal", "double",
        "duties", "duty", "dying", "effectively", "ensure", "entertain", "entertaining",
        "established", "even", "exact", "exactly", "exception", "exciting", "expense",
        "experience", "experienced", "eye", "fair", "faith", "faithful", "familiar",
        "famous", "fancy", "fellow", "festive", "fill", "fine", "fire", "fitness", "fresh",
        "get", "give", "golden", "grab", "guarantee", "guide", "hand", "hang", "hard",
        "health", "heart", "highlight", "horns", "importance", "important", "interesting",
        "intriguing", "involve", "join", "just", "kind", "know", "let", "light", "like",
        "likely", "lone", "low", "lower", "luck", "main", "major", "make", "matter",
        "mean", "mind", "mix", "mixed", "modest", "normal", "notable", "odd", "odds",
        "offer", "order", "outset", "owing", "particular", "partner", "partnership",
        "pass", "pay", "pick", "piece", "plenty", "press", "pretty", "provide", "real",
        "realistic", "realistically", "really", "regard", "regarding", "regardless",
        "revenge", "revert", "right", "sense", "settle", "shake", "short", "shortly",
        "show", "significant", "slight", "snatch", "sound", "spare", "stand", "support",
        "taste", "thanks", "threat", "thrilling", "too", "true", "turn", "undoubtedly",
        "welcome", "well", "worth",
    },
}
NOT_TONE_WORDS = set().union(*NOT_TONE.values())

# Tone in sport that Harvard lacks or reads the wrong way round ("boast" is negative in
# Harvard, but "Arsenal boast the best defence" is good news for Arsenal).
SPORT_POSITIVE = {
    "boast", "clinical", "dominant", "formidable", "free-scoring", "in-form", "incredible",
    "momentum", "promoted", "promotion", "rampant", "resilient", "resurgent", "ruthless",
    "solid", "strong", "stronger", "strongest", "unbeaten", "undefeated", "victories",
    "victory", "win", "winner", "winners", "winning", "wins", "won",
}
SPORT_NEGATIVE = {
    "ban", "banned", "concede", "humbling", "inconsistent", "injury-hit", "leaky", "nursing",
    "out-of-form", "relegated", "relegation", "rock-bottom", "sacked", "setback", "sidelined",
    "unavailable", "vulnerable", "weak", "weaker", "winless",
}
BRITISH = {"favor": "favour", "honor": "honour", "humor": "humour", "rumor": "rumour", "vigor": "vigour",
           "valor": "valour", "splendor": "splendour", "behavior": "behaviour", "labor": "labour",
           "color": "colour", "ardor": "ardour"}

DOUBLES_LAST_LETTER = re.compile(r"[^aeiou][aeiou][bdgmnprt]$")  # hit -> hitting, drop -> dropped


def forms(word: str) -> set[str]:
    """The word with its regular endings.

    >>> sorted(forms("struggle"))
    ['struggle', 'struggled', 'strugglely', 'struggles', 'struggling']
    >>> {"hitting", "worries", "lying"} <= forms("hit") | forms("worry") | forms("lie")
    True

    Some forms aren't words ("strugglely"); they never match, so they do no harm.
    """
    if word.endswith("ie"):
        found = {word, word + "s", word + "d", word[:-2] + "ying"}
    elif word.endswith("e"):
        found = {word, word + "s", word + "d", word[:-1] + "ing", word + "ly"}
    else:
        found = {word, word + "s", word + "es", word + "ed", word + "ing", word + "ly"}
    if word.endswith("y"):
        found |= {word[:-1] + "ies", word[:-1] + "ied"}
    if DOUBLES_LAST_LETTER.search(word):
        found |= {word + word[-1] + "ed", word + word[-1] + "ing"}
    return found


def with_forms(words) -> set[str]:
    return set().union(*(forms(w) for w in words))


def british(words: set[str]) -> set[str]:
    """The words plus their British spellings: favorable -> favourable."""
    return words | {w.replace(us, uk) for w in words for us, uk in BRITISH.items() if us in w}


def harvard(path: Path = HARVARD_FILE) -> tuple[set[str], set[str]]:
    """Harvard IV's positive and negative words, lower case, without sense numbers (ABOUT#1 -> about)."""
    if not path.exists():
        raise FileNotFoundError(f"No {path}. Download {config.HARVARD_IV} and save it there (see pvb/sentiment.py).")
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    word = table["Entry"].str.replace(r"#.*$", "", regex=True).str.lower()
    return british(set(word[table["Positiv"] == "Positiv"])), british(set(word[table["Negativ"] == "Negativ"]))


@cache
def word_lists(path: Path = HARVARD_FILE) -> tuple[frozenset[str], frozenset[str]]:
    """(positive, negative): every form of every word, ready to look words up in."""
    positive, negative = (with_forms(words) for words in harvard(path))
    drop = (positive & negative) | with_forms(NOT_TONE_WORDS)
    sport_positive, sport_negative = with_forms(SPORT_POSITIVE), with_forms(SPORT_NEGATIVE)
    positive = (positive - drop - sport_negative) | sport_positive
    negative = (negative - drop - sport_positive) | sport_negative
    return frozenset(positive), frozenset(negative)


def review(texts, path: Path = HARVARD_FILE) -> pd.DataFrame:
    """The dictionary review: every listed word in at least 1% of `texts`, and what we decided.

    `texts` are the previews from before the test season. Only lower-case words
    count (see features.lower_case_words).
    """
    seen = Counter()
    for text in texts:
        seen.update({w for w in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", text) if w.islower()})
    harvard_positive, harvard_negative = (with_forms(words) for words in harvard(path))
    positive, negative = word_lists(path)
    reason_of = {form: reason for reason, words in NOT_TONE.items() for form in with_forms(words)}
    rows = []
    for word in (harvard_positive | harvard_negative | positive | negative):
        share = seen[word] / len(texts)
        if share < REVIEW_MIN_SHARE:
            continue
        in_harvard = ("both" if word in harvard_negative else "positive") if word in harvard_positive else (
            "negative" if word in harvard_negative else "")
        ours = "positive" if word in positive else "negative" if word in negative else ""
        if word in with_forms(SPORT_POSITIVE | SPORT_NEGATIVE):
            reason = "added: sports tone that Harvard lacks" if not in_harvard else "moved: Harvard has the wrong sign for sport"
        elif in_harvard == "both":
            reason = "in both Harvard lists"
        else:
            reason = reason_of.get(word, "")
        rows.append({"word": word, "share_of_previews": round(share, 4), "harvard": in_harvard, "ours": ours, "reason": reason})
    return pd.DataFrame(rows).sort_values(["share_of_previews", "word"], ascending=[False, True], ignore_index=True)
