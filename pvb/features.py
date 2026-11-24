"""Text features: turn each preview's text into numbers.

`python -m pvb features` writes data/processed/features.csv, one row per preview.
It holds no text, so it goes on GitHub. Side a is the home team in football and
the first-named player in tennis; side b is the other one.

- words_a, words_b: how many lower-case words are about each side (see "Which side" below).
- tone_a, tone_b, tone_diff: the tone of the sentences about each side, as in the
  course: (positive - negative) / (positive + negative) words, 0 if there are
  neither. The word lists are Harvard IV adapted to sport (sentiment.py). With few
  tone words the measure is noisy (one word gives +1 or -1), which only weakens it.
- absence_a, absence_b, absence_diff: absence words (injured, suspended, without,
  ...) per 100 words of team news about each side. Football only (empty for tennis,
  which has no team news).
- uncertainty_per_1000: hedging words ("could", "tight") per 1,000 words.
- nb_home_vs_draw, nb_away_vs_draw (football), nb_a_vs_b (tennis): Naive Bayes log
  odds, log P(home | words) - log P(draw | words) and so on. Only matched games
  have them, because the model learns from results (see `naive_bayes`).

Which side a sentence is about (`credit_sentences`): a sentence that names one
side is about that side. One that names neither is about the side of the sentence
before, because Sports Mole names a team once and then writes "they" or a
nickname ("Bournemouth ... The Cherries ..."). One that names both is about
neither, and so are the sentences after it until one side is named. Each section
starts afresh. So about a quarter of the words count for neither side: most of
the introduction (which names both teams) and the verdict.
A side is named by the words of its name, and in football also by "the hosts" or
"the visitors" and by the words learned for that team and season (`learn_aliases`):
mostly nicknames, the manager, the players and the ground. In a sample of real
previews, about 97% of the football sentences credited to a side, and 95% in
tennis, were about that side.

Only words written in lower case count as words (`lower_case_words`). Names are
capitalised, so names such as Crystal Palace, Inter or Wolves never count as tone,
and Naive Bayes can't learn which teams win; the odds already know that.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from . import config, sentiment
from .matching import FOOTBALL_CSV, TENNIS_CSV
from .parse import PREVIEWS_CSV

FEATURES_CSV = config.PROCESSED / "features.csv"

# "text_other" is left out: in tennis it is mostly lists of earlier scores.
SECTIONS = ["text_intro", "text_preview", "text_team_news", "text_head_to_head", "text_verdict"]
OUTCOMES = {"football": ["home", "draw", "away"], "tennis": ["a", "b"]}  # coded 0, 1, 2 as in analysis.py

ABSENCE_TERMS = [
    "injured", "injury", "injuries", "ruled out", "suspended", "suspension", "banned", "doubt",
    "doubtful", "sidelined", "unavailable", "absent", "absence", "miss", "misses", "missing", "without",
]
UNCERTAINTY_TERMS = ["could", "might", "perhaps", "tight", "unpredictable", "hard to call", "uncertain"]

ROLE_WORDS = {  # point at one football side without naming it
    "a": ("the hosts", "the home side", "the home team"),
    "b": ("the visitors", "the away side", "the away team", "the guests", "the visiting side"),
}
NOT_NAMES = {  # capitalised, but they name no side
    "The", "AFC", "Club", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}
ALIAS_MIN_PREVIEWS = 5  # learn aliases only for teams with this many previews in a season
ALIAS_MIN_SHARE = 0.25  # the word is in at least this share of the team's previews...
ALIAS_MIN_LIFT = 10  # ...and at least this many times as common there as in all previews
ALIAS_MAX_TEAMS = 2  # a word learned for more teams in a season ("Scudetto", "Garcia") names none
NB_MIN_PREVIEWS = 10  # Naive Bayes reads only words found in this many training previews
# Stop words carry no information, but sklearn's list also holds words that do here.
NB_STOP_WORDS = ENGLISH_STOP_WORDS - {
    "no", "not", "never", "nor", "nothing", "none", "without", "top", "bottom", "first", "last",
    "least", "less", "more", "most", "few", "full",
}

# A sentence ends at . ! or ? followed by a capital, a digit or a quote, but not after
# "No." (world No. 1), "St." (St. Pauli), "vs.", "Mr." or "Dr.".
SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])(?<!\bNo\.)(?<!\bSt\.)(?<!\bvs\.)(?<!\bMr\.)(?<!\bDr\.)\s+(?=[\"'‘“A-Z0-9])|\n+"
)
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")  # letters, with hyphens inside: "in-form"
NAME_PART = re.compile(r"[^\W\d_]{3,}")  # "Paris Saint-Germain" -> Paris, Saint, Germain


# Words -------------------------------------------------------------------------

@cache
def _terms_pattern(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", flags=re.I)


def count_terms(text: str, terms: list[str]) -> int:
    """How often any of `terms` appears as whole words (case-insensitive)."""
    if not text:
        return 0
    return len(_terms_pattern(tuple(terms)).findall(text))


def per_1000_words(count: int, words: int) -> float:
    return 1000 * count / words if words else 0.0


def sentences(text: str) -> list[str]:
    """Split at line breaks and at sentence ends (see SENTENCE_BREAK)."""
    return [s.strip() for s in SENTENCE_BREAK.split(text) if s.strip()]


def lower_case_words(text: str) -> list[str]:
    """The words written in lower case.

    >>> lower_case_words("Arsenal's in-form side beat Wolves 2-0.")
    ['s', 'in-form', 'side', 'beat']
    """
    return [w for w in WORD.findall(text) if w.islower()]


def name_words(text: str) -> set[str]:
    """Capitalised words of three letters or more: the words that can name a side."""
    return {w for w in NAME_PART.findall(text) if w[0].isupper()} - NOT_NAMES


def tone(words: list[str], positive, negative) -> float:
    """(positive - negative) / (positive + negative) words, or 0 if there are neither."""
    plus = sum(w in positive for w in words)
    minus = sum(w in negative for w in words)
    return (plus - minus) / (plus + minus) if plus + minus else 0.0


# Which side a sentence is about ------------------------------------------------------

def season_start(published: pd.Series) -> pd.Series:
    """The year the season started; seasons run from July to June."""
    dates = pd.to_datetime(published, utc=True, format="ISO8601")
    return dates.dt.year - (dates.dt.month < 7)


def full_text(previews: pd.DataFrame) -> pd.Series:
    return previews[SECTIONS].astype(object).fillna("").agg("\n".join, axis=1)


def learn_aliases(previews: pd.DataFrame) -> dict[tuple[str, int], set[str]]:
    """(team, season) -> other words for the team that season: nickname, manager, players, ground.

    A capitalised word becomes an alias of a team if it is in at least 25% of the
    team's previews that season, and at least 10 times as common there as in all
    previews. Per season, because players and managers move. Words of team names
    are left out, so a rival's name never becomes an alias, and so are words
    learned for more than two teams in a season: they name a league ("Scudetto")
    or are common names ("Garcia").
    """
    previews = previews.dropna(subset=["side_a", "side_b"])
    found = full_text(previews).map(name_words)
    everywhere = Counter(w for words in found for w in words)
    team_words = set().union(*map(name_words, pd.concat([previews["side_a"], previews["side_b"]]).unique()))
    seen, size = defaultdict(Counter), Counter()
    seasons = season_start(previews["published"])
    for column in ["side_a", "side_b"]:
        for team, season, words in zip(previews[column], seasons, found):
            seen[team, season].update(words)
            size[team, season] += 1
    aliases = {}
    for key, counts in seen.items():
        if size[key] < ALIAS_MIN_PREVIEWS:
            continue
        share = {w: k / size[key] for w, k in counts.items()}
        aliases[key] = {
            w for w, s in share.items()
            if s >= ALIAS_MIN_SHARE and s >= ALIAS_MIN_LIFT * everywhere[w] / len(previews)
        } - team_words
    teams_per_word = Counter((season, w) for (_, season), words in aliases.items() for w in words)
    return {
        (team, season): {w for w in words if teams_per_word[season, w] <= ALIAS_MAX_TEAMS}
        for (team, season), words in aliases.items()
    }


def side_words(name_a: str, name_b: str, aliases_a: set | frozenset = frozenset(),
               aliases_b: set | frozenset = frozenset()) -> tuple[set, set]:
    """The words that point at side a and at side b. A word both sides share points at neither.

    >>> a, b = side_words("Manchester United", "Manchester City")
    >>> sorted(a), sorted(b)
    (['United'], ['City'])
    """
    a, b = name_words(name_a), name_words(name_b)
    return (a - b) | (set(aliases_a) - set(aliases_b)), (b - a) | (set(aliases_b) - set(aliases_a))


def credit_sentences(text: str, words_a: set, words_b: set, roles_a=(), roles_b=()) -> list[tuple[str | None, str]]:
    """(side, sentence) for each sentence: side is "a", "b" or None (both sides, or not known yet)."""
    credited, side = [], None
    for sentence in sentences(text):
        found, lower = name_words(sentence), sentence.lower()
        about_a = bool(found & words_a) or any(r in lower for r in roles_a)
        about_b = bool(found & words_b) or any(r in lower for r in roles_b)
        if about_a != about_b:
            side = "a" if about_a else "b"
        elif about_a:  # names both
            side = None
        credited.append((side, sentence))
    return credited


def credit_preview(preview, aliases: dict, season: int) -> list[tuple[str, str | None, str]]:
    """(section, side, sentence) for every sentence of one preview (a row of previews.csv).

    The side carries on within a section, not into the next one.
    """
    if pd.isna(preview.side_a) or pd.isna(preview.side_b):
        return []
    football = preview.sport == "football"
    words_a, words_b = side_words(
        preview.side_a, preview.side_b,
        aliases.get((preview.side_a, season), set()) if football else set(),
        aliases.get((preview.side_b, season), set()) if football else set(),
    )
    roles = (ROLE_WORDS["a"], ROLE_WORDS["b"]) if football else ((), ())
    return [
        (section, side, sentence)
        for section in SECTIONS if isinstance(text := getattr(preview, section), str)
        for side, sentence in credit_sentences(text, words_a, words_b, *roles)
    ]


# Features of one preview ----------------------------------------------------------

def side_features(credited: list, positive, negative) -> dict:
    """Tone and absence words of the sentences about each side.

    Absence words are counted per 100 words of team news about the side. Raw counts
    mostly measure how much is written: big clubs get longer team news, so more
    absence words went with *better* odds, not worse.
    """
    words = {"a": [], "b": []}
    news_words = {"a": 0, "b": 0}
    absences = {"a": 0, "b": 0}
    for section, side, sentence in credited:
        if side is None:
            continue
        found = lower_case_words(sentence)
        words[side] += found
        if section == "text_team_news":
            news_words[side] += len(found)
            absences[side] += count_terms(sentence, ABSENCE_TERMS)
    tone_a, tone_b = tone(words["a"], positive, negative), tone(words["b"], positive, negative)
    absence_a, absence_b = (100 * absences[s] / news_words[s] if news_words[s] else 0.0 for s in "ab")
    return {
        "words_a": len(words["a"]), "words_b": len(words["b"]),
        "tone_a": tone_a, "tone_b": tone_b, "tone_diff": tone_a - tone_b,
        "absence_a": absence_a, "absence_b": absence_b, "absence_diff": absence_a - absence_b,
    }


def nb_document(credited: list) -> str:
    """The words Naive Bayes reads: lower case, no stop words, each marked with its side.

    "a:injured" is "injured" in a sentence about side a, so the model can learn
    that it is bad news for a and good news for b.

    >>> nb_document([("text_team_news", "a", "Saka is injured."), ("text_verdict", None, "A tight game.")])
    'a:injured tight game'
    """
    return " ".join(
        f"{side}:{w}" if side else w
        for _, side, sentence in credited
        for w in lower_case_words(sentence) if w not in NB_STOP_WORDS
    )


# Naive Bayes ----------------------------------------------------------------------

def seasons(games: pd.DataFrame, sport: str) -> pd.Series:
    """Football: football-data's season code ("2526"); tennis: the year ("2026")."""
    if sport == "football":
        return games["season"].astype(str)
    return pd.to_datetime(games["date"]).dt.year.astype(str)


def split_of(season: pd.Series, sport: str) -> pd.Series:
    """train, test or check, from config.SPLITS (NaN outside the study)."""
    sample_of = {s: sample for sample, ss in config.SPLITS[sport].items() for s in ss}
    return season.map(sample_of)


def naive_bayes(documents: pd.Series, results: pd.Series, samples: pd.Series, season: pd.Series) -> np.ndarray:
    """Log P(result | words) for each document: one row per document, one column per result.

    One model is fitted per training season, on the other training seasons
    (cross-fitting). Each training document gets the model that left its own
    season out. Without this, its probabilities would come from a model that has
    seen its result: they would look far better than they are, and the logit in
    analysis.py, which learns on the training seasons, would trust them too much.
    Test and check documents get the average of these models, so their numbers are
    on the same scale as the training documents' (a model fitted on all training
    seasons knows more words and gives more extreme numbers). An average of log
    probabilities no longer sums to 1 as probabilities; only differences between
    the columns (log odds) are used.
    """
    text, y, year = documents.to_numpy(), results.to_numpy(), season.to_numpy()
    train = (samples == "train").to_numpy()
    labelled = ~pd.isna(y)
    classes = np.unique(y[train & labelled]).astype(int)
    out = np.full((len(text), len(classes)), np.nan)
    others = np.zeros((len(text), len(classes)))
    folds = np.unique(year[train])
    for left_out in folds:
        fit = train & labelled & (year != left_out)
        vectoriser = CountVectorizer(token_pattern=r"\S+", lowercase=False, min_df=NB_MIN_PREVIEWS)
        model = MultinomialNB().fit(vectoriser.fit_transform(text[fit]), y[fit].astype(int))
        if list(model.classes_) != list(classes):
            raise ValueError(f"the training seasons without {left_out} lack some results: {model.classes_}")
        own, rest = train & (year == left_out), ~train
        out[own] = model.predict_log_proba(vectoriser.transform(text[own]))
        if rest.any():
            others[rest] += model.predict_log_proba(vectoriser.transform(text[rest])) / len(folds)
    out[~train] = others[~train]
    return out


def naive_bayes_features(games: pd.DataFrame, documents: dict, sport: str) -> pd.DataFrame:
    """Naive Bayes log odds for the matched games of `sport` (games: a matched table).

    Log odds against one result (the draw; player b), because a column per result
    would carry the model's normalising term in every column.
    """
    games = games[games["article_id"].isin(list(documents))]
    names = ["nb_home_vs_draw", "nb_away_vs_draw"] if sport == "football" else ["nb_a_vs_b"]
    if games.empty:
        return pd.DataFrame(columns=["article_id", *names], dtype=float)
    if sport == "football":
        results = games["result"].map({"H": 0, "D": 1, "A": 2})
    else:
        results = pd.Series(np.where(games["a_won"], 0, 1), index=games.index)
    season = seasons(games, sport)
    log_p = naive_bayes(games["article_id"].map(documents), results, split_of(season, sport), season)
    if sport == "football":
        columns = {"nb_home_vs_draw": log_p[:, 0] - log_p[:, 1], "nb_away_vs_draw": log_p[:, 2] - log_p[:, 1]}
    else:
        columns = {"nb_a_vs_b": log_p[:, 0] - log_p[:, 1]}
    return pd.DataFrame({"article_id": games["article_id"].to_numpy(), **columns})


# Build ----------------------------------------------------------------------------

def basic_features(previews: pd.DataFrame) -> pd.DataFrame:
    """The pick (for checks against picks.csv) and the share of hedging words."""
    out = previews[["article_id", "sport", "pick_result", "pick_home_goals", "pick_away_goals", "pick_winner", "pick_sets"]].copy()
    full = previews[[c for c in previews.columns if c.startswith("text_")]].astype(object).fillna("").agg("\n".join, axis=1)
    out["uncertainty_per_1000"] = [
        per_1000_words(count_terms(t, UNCERTAINTY_TERMS), n) for t, n in zip(full, previews["word_count"])
    ]
    return out


def read_matched() -> dict[str, pd.DataFrame]:
    for path in (FOOTBALL_CSV, TENNIS_CSV):
        if not path.exists():
            raise FileNotFoundError(f"No {path.name} yet. Run `python -m pvb match` first: Naive Bayes learns from the results.")
    return {
        "football": pd.read_csv(FOOTBALL_CSV, dtype={"season": str}),
        "tennis": pd.read_csv(TENNIS_CSV),
    }


def build() -> pd.DataFrame:
    """Write features.csv, and the dictionary review next to it (see sentiment.py)."""
    matched = read_matched()
    previews = pd.read_csv(PREVIEWS_CSV)
    positive, negative = sentiment.word_lists(sentiment.HARVARD_FILE)
    aliases = learn_aliases(previews[previews["sport"] == "football"])
    wanted = set(matched["football"]["article_id"]) | set(matched["tennis"]["article_id"])

    rows, documents = [], {}
    for preview, season in zip(previews.itertuples(index=False), season_start(previews["published"])):
        credited = credit_preview(preview, aliases, season)
        rows.append(side_features(credited, positive, negative) if credited else {})
        if preview.article_id in wanted:
            documents[preview.article_id] = nb_document(credited)

    features = basic_features(previews).join(pd.DataFrame(rows, index=previews.index))
    features.loc[features["sport"] != "football", ["absence_a", "absence_b", "absence_diff"]] = np.nan
    for sport in config.SPORTS:
        features = features.merge(naive_bayes_features(matched[sport], documents, sport),
                                  on="article_id", how="left", validate="one_to_one")
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.round(6).to_csv(FEATURES_CSV, index=False)

    before = season_start(previews["published"]) < sentiment.REVIEW_BEFORE
    sentiment.review(full_text(previews[before]).tolist(), sentiment.HARVARD_FILE).to_csv(
        FEATURES_CSV.parent / "dictionary_review.csv", index=False)
    return features
