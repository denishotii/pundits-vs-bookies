"""Tests for the text features (pvb/features.py) and the word lists (pvb/sentiment.py).

The examples are short texts written for the tests, so the answers can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from pvb import features, sentiment
from pvb.features import (
    credit_preview, credit_sentences, learn_aliases, naive_bayes, naive_bayes_features, name_words,
    nb_document, season_start, sentences, side_features, side_words, split_of, tone,
)

# Words and sentences -------------------------------------------------------------------


def test_sentences_end_at_full_stops_and_line_breaks():
    text = "Arsenal won 2-0. They looked sharp!\nSaka is out. A 1-1 draw? Perhaps."
    assert sentences(text) == ["Arsenal won 2-0.", "They looked sharp!", "Saka is out.", "A 1-1 draw?", "Perhaps."]


def test_sentences_do_not_end_at_abbreviations():
    text = "World No. 1 Sinner beat St. Pauli fan Alcaraz. Then came Ruud vs. Rune."
    assert sentences(text) == ["World No. 1 Sinner beat St. Pauli fan Alcaraz.", "Then came Ruud vs. Rune."]


@pytest.mark.parametrize("name, words", [
    ("Paris Saint-Germain", {"Paris", "Saint", "Germain"}),
    ("AFC Bournemouth", {"Bournemouth"}),
    ("Al-Ittihad", {"Ittihad"}),  # "Al" is too short to name anyone
    ("Atlético Madrid", {"Atlético", "Madrid"}),
])
def test_name_words(name, words):
    assert name_words(name) == words


def test_words_both_sides_share_point_at_neither():
    a, b = side_words("Manchester United", "Manchester City", {"Devils", "Red"}, {"Guardiola", "Red"})
    assert (a, b) == ({"United", "Devils"}, {"City", "Guardiola"})


# Which side a sentence is about -----------------------------------------------------------


def test_a_sentence_is_about_the_side_it_names_or_the_side_before():
    text = (
        "Arsenal have won five in a row. The Gunners look sharp.\n"
        "Chelsea, meanwhile, have lost twice. They are struggling.\n"
        "Arsenal beat Chelsea last time. It was tight."
    )
    credited = credit_sentences(text, {"Arsenal", "Gunners"}, {"Chelsea"})
    assert [side for side, _ in credited] == ["a", "a", "b", "b", None, None]


def test_the_hosts_and_the_visitors_name_a_side():
    text = "The hosts are unbeaten. The visitors have lost three in a row."
    credited = credit_sentences(text, set(), set(), ("the hosts",), ("the visitors",))
    assert [side for side, _ in credited] == ["a", "b"]


def preview_table(rows: list[dict]) -> pd.DataFrame:
    columns = ["article_id", "sport", "published", "side_a", "side_b",
               "text_intro", "text_preview", "text_team_news", "text_head_to_head", "text_verdict"]
    return pd.DataFrame(rows).reindex(columns=columns)


def test_aliases_are_frequent_words_seen_mostly_with_one_team():
    rows = []
    for i in range(6):  # Arsenal: "Gunners" in 5 of 6 previews, "Arteta" in 1, a rival's name in all
        text = ("The Gunners are in form. " if i < 5 else "") + ("Arteta is happy. " if i == 0 else "") + "Chelsea lead the Premier League."
        rows.append({"side_a": "Arsenal", "side_b": f"Team{i}", "published": "2024-10-01", "text_preview": text})
    for i in range(60):  # other games, so that "Gunners" is rare overall
        rows.append({"side_a": "Leeds", "side_b": "Hull", "published": "2024-10-01", "text_preview": "The Premier League awaits."})
    rows.append({"side_a": "Chelsea", "side_b": "Hull", "published": "2024-10-01", "text_preview": "Chelsea."})
    aliases = learn_aliases(preview_table(rows))
    assert aliases["Arsenal", 2024] == {"Gunners"}  # not Arteta (1 in 6), Chelsea (a team) or Premier (everywhere)
    assert ("Arsenal", 2025) not in aliases  # learned per season
    assert ("Team0", 2024) not in aliases  # too few previews to learn from


def test_a_word_learned_for_many_teams_names_none():
    rows = []
    for team in ["Inter", "Milan", "Napoli"]:  # "Scudetto" is in every preview of all three
        for i in range(5):
            rows.append({"side_a": team, "side_b": f"{team}{i}", "published": "2024-10-01",
                         "text_preview": f"The Scudetto race goes on. Nickname{team} are ready."})
    for _ in range(150):
        rows.append({"side_a": "Leeds", "side_b": "Hull", "published": "2024-10-01", "text_preview": "Nothing."})
    aliases = learn_aliases(preview_table(rows))
    assert aliases["Inter", 2024] == {"NicknameInter"}


def one_preview(sport: str):
    table = preview_table([{
        "sport": sport, "side_a": "Arsenal", "side_b": "Chelsea",
        "text_intro": "Arsenal host Chelsea.",
        "text_preview": "The Gunners are flying. They have won six.",
        "text_team_news": "Saka is injured.",
        "text_verdict": "Chelsea will struggle.",
    }])
    return next(table.itertuples(index=False))


def test_credit_preview_uses_aliases_in_football_and_restarts_each_section():
    aliases = {("Arsenal", 2024): {"Gunners"}}
    football = credit_preview(one_preview("football"), aliases, 2024)
    assert [(section, side) for section, side, _ in football] == [
        ("text_intro", None), ("text_preview", "a"), ("text_preview", "a"),
        ("text_team_news", None),  # Saka is named nowhere, and the side doesn't carry over
        ("text_verdict", "b"),
    ]
    tennis = credit_preview(one_preview("tennis"), aliases, 2024)
    assert [side for _, side, _ in tennis][1:3] == [None, None]  # no learned aliases in tennis
    assert credit_preview(one_preview("football"), aliases, 2025)[1][1] is None  # other season


# Features of one preview -----------------------------------------------------------------


def test_tone_and_absence_per_side():
    credited = [
        ("text_preview", "a", "Arsenal won and looked impressive."),
        ("text_preview", "b", "Chelsea lost again."),
        ("text_team_news", "a", "Saka is injured and Rice is out."),
        ("text_team_news", "b", "Chelsea have no injuries."),
        ("text_verdict", None, "Arsenal lost nothing."),  # about neither side: ignored
    ]
    found = side_features(credited, positive={"won", "impressive"}, negative={"lost", "injured"})
    # a: won, impressive, injured -> (2 - 1) / 3; b: lost -> -1
    assert found["tone_a"] == pytest.approx(1 / 3)
    assert found["tone_b"] == -1
    assert found["tone_diff"] == pytest.approx(4 / 3)
    # absence words per 100 team-news words: a has 1 in 5 (is injured and is out), b 1 in 3
    assert found["absence_a"] == pytest.approx(20)
    assert found["absence_b"] == pytest.approx(100 / 3)
    assert (found["words_a"], found["words_b"]) == (9, 5)


def test_tone_is_zero_without_tone_words():
    assert tone(["the", "ball"], {"won"}, {"lost"}) == 0


def test_naive_bayes_reads_no_names_and_marks_the_side():
    credited = [("text_preview", "a", "Arsenal are rampant."), ("text_verdict", None, "Expect a tight game.")]
    assert nb_document(credited) == "a:rampant tight game"  # "Expect" is capitalised, so it is dropped too


# Word lists ---------------------------------------------------------------------------------

HARVARD = """Entry\tSource\tPositiv\tNegativ\tOther
HOME\tH4\tPositiv\t\t
AGAINST\tH4\t\tNegativ\t
IMPRESSIVE\tH4\tPositiv\t\t
LOSE\tH4\t\tNegativ\t
BOAST\tH4\t\tNegativ\t
FAIR#1\tH4\tPositiv\t\t
FAIR#2\tH4\t\tNegativ\t
FAVORABLE\tH4\tPositiv\t\t
HONOR#_10\tH4\tPositiv\t\t
"""


@pytest.fixture
def word_lists(tmp_path):
    path = tmp_path / "inqtabs.txt"
    path.write_text(HARVARD)
    return sentiment.word_lists(path)


def test_word_lists_adapt_harvard_to_sport(word_lists):
    positive, negative = word_lists
    assert {"impressive", "win", "won", "unbeaten", "boast", "boasting"} <= positive
    assert {"lose", "loses", "losing", "winless", "conceded", "nursing"} <= negative
    for word in ["home", "against", "fair"]:  # no tone in a preview, or both in Harvard
        assert word not in positive | negative
    assert not positive & negative
    assert {"favourable", "honour"} <= positive  # British spellings of Harvard's American ones


def test_sport_lists_and_not_tone_never_clash():
    sport = sentiment.with_forms(sentiment.SPORT_POSITIVE) | sentiment.with_forms(sentiment.SPORT_NEGATIVE)
    assert not sentiment.with_forms(sentiment.NOT_TONE) & sport
    assert not sentiment.SPORT_POSITIVE & sentiment.SPORT_NEGATIVE


def test_missing_dictionary_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="inquirer"):
        sentiment.harvard(tmp_path / "missing.txt")


# Naive Bayes --------------------------------------------------------------------------------


def nb_data(seed=0, per_season=200):
    """Two training seasons and a test season. "a:injured" means b wins, "b:injured" means a wins."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, sample in [("s1", "train"), ("s2", "train"), ("s3", "test")]:
        for _ in range(per_season):
            result = int(rng.integers(2))
            injured = "a:injured" if result == 1 else "b:injured"
            if rng.random() < 0.2:  # the signal is not perfect
                injured = "b:injured" if result == 1 else "a:injured"
            rows.append({"document": f"{injured} game tight form", "result": result, "season": season, "sample": sample})
    return pd.DataFrame(rows)


def run_nb(data: pd.DataFrame) -> np.ndarray:
    return naive_bayes(data["document"], data["result"], data["sample"], data["season"])


def test_naive_bayes_learns_from_the_words():
    data = nb_data()
    log_p = run_nb(data)
    test = (data["sample"] == "test").to_numpy()
    assert np.allclose(np.exp(log_p[~test]).sum(axis=1), 1)  # test rows hold an average of log probabilities
    a_hurt = data["document"].str.startswith("a:injured").to_numpy()
    assert (log_p[test & a_hurt, 1] > log_p[test & a_hurt, 0]).all()  # b more likely when a has injuries
    assert (log_p[test & ~a_hurt, 0] > log_p[test & ~a_hurt, 1]).all()


def test_naive_bayes_never_sees_the_result_it_predicts():
    data = nb_data()
    before = run_nb(data)
    # Flip season s1's results: s1's own probabilities come from a model fitted on s2 only...
    flipped = data.assign(result=np.where(data["season"] == "s1", 1 - data["result"], data["result"]))
    after = run_nb(flipped)
    s1 = (data["season"] == "s1").to_numpy()
    assert np.allclose(before[s1], after[s1])
    assert not np.allclose(before[~s1], after[~s1])  # ...while s2 and the test season did learn from s1
    # The test season's results are never used at all
    shuffled = data.assign(result=np.where(data["sample"] == "test", data["result"].sample(frac=1, random_state=1).to_numpy(), data["result"]))
    assert np.allclose(before, run_nb(shuffled))


# Seasons --------------------------------------------------------------------------------------


def test_seasons_run_from_july_to_june():
    published = pd.Series(["2025-08-01T12:00:00+01:00", "2026-03-01T10:00:00+00:00", "2025-06-30T10:00:00+00:00"])
    assert season_start(published).tolist() == [2025, 2025, 2024]


def test_split_of_follows_config_splits():
    found = split_of(pd.Series(["2425", "2526", "2627", "1920"]), "football")
    assert found.iloc[:3].tolist() == ["train", "test", "check"]
    assert pd.isna(found.iloc[3])



def test_naive_bayes_columns_follow_the_results():
    """Home-win words must raise nb_home_vs_draw; a_won must feed nb_a_vs_b, not the other way round."""
    data = nb_data(per_season=300)
    seasons_of = {"s1": "2324", "s2": "2425", "s3": "2526"}
    football = pd.DataFrame({
        "article_id": np.arange(len(data)), "season": data["season"].map(seasons_of),
        "result": np.where(data["result"] == 0, "H", "A"),  # a:injured -> away win
    })
    football.loc[::7, "result"] = "D"  # some draws, so all three results are there
    documents = dict(zip(football["article_id"], data["document"]))
    found = naive_bayes_features(football, documents, "football")
    a_hurt = data["document"].str.startswith("a:injured").to_numpy()
    assert found.loc[a_hurt, "nb_away_vs_draw"].mean() > found.loc[~a_hurt, "nb_away_vs_draw"].mean()
    assert found.loc[~a_hurt, "nb_home_vs_draw"].mean() > found.loc[a_hurt, "nb_home_vs_draw"].mean()

    tennis = pd.DataFrame({
        "article_id": np.arange(len(data)), "a_won": data["result"] == 0,
        "date": data["season"].map({"s1": "2024-05-01", "s2": "2025-05-01", "s3": "2026-05-01"}),
    })
    found = naive_bayes_features(tennis, documents, "tennis")
    assert (found.loc[~a_hurt, "nb_a_vs_b"] > 0).mean() > 0.9
    assert (found.loc[a_hurt, "nb_a_vs_b"] < 0).mean() > 0.9


# The whole build, on a small made-up corpus -----------------------------------------------------


def test_build_keeps_each_preview_on_its_own_row(tmp_path, monkeypatch):
    """Each row's features must come from its own text, after all the joins and merges."""
    rng = np.random.default_rng(3)
    previews, football = [], []
    for i in range(240):
        season, published = [("2324", "2023-10-01"), ("2425", "2024-10-01"), ("2526", "2025-10-01")][i % 3]
        good = rng.random() < 0.5
        previews.append({
            "article_id": 1000 + i, "sport": "football", "published": f"{published}T12:00:00+00:00",
            "side_a": "Arsenal", "side_b": "Chelsea", "word_count": 20,
            "pick_result": "H", "pick_home_goals": 1, "pick_away_goals": 0, "pick_winner": None, "pick_sets": None,
            "text_intro": "Arsenal host Chelsea.",
            "text_preview": "Arsenal looked impressive." if good else "Arsenal are winless.",
            "text_team_news": "Chelsea are without three players.", "text_head_to_head": None,
            "text_verdict": "We expect goals.",
        })
        football.append({"article_id": 1000 + i, "season": season, "result": "H" if good else ["A", "D"][i % 2]})
    previews.append({"article_id": 5000, "sport": "tennis", "published": "2025-05-01T12:00:00+00:00",
                     "side_a": "Jannik Sinner", "side_b": "Carlos Alcaraz", "word_count": 5,
                     "text_preview": "Sinner looked impressive."})
    paths = {"PREVIEWS_CSV": tmp_path / "previews.csv", "FOOTBALL_CSV": tmp_path / "football.csv",
             "TENNIS_CSV": tmp_path / "tennis.csv", "FEATURES_CSV": tmp_path / "out" / "features.csv"}
    pd.DataFrame(previews).to_csv(paths["PREVIEWS_CSV"], index=False)
    pd.DataFrame(football).to_csv(paths["FOOTBALL_CSV"], index=False)
    pd.DataFrame(columns=["article_id", "a_won", "date"]).to_csv(paths["TENNIS_CSV"], index=False)
    harvard = tmp_path / "inqtabs.txt"
    harvard.write_text(HARVARD)
    for name, path in paths.items():
        monkeypatch.setattr(features, name, path)
    monkeypatch.setattr(sentiment, "HARVARD_FILE", harvard)
    monkeypatch.setattr(features, "NB_MIN_PREVIEWS", 1)

    built = features.build().set_index("article_id")
    good = pd.Series([p["text_preview"] == "Arsenal looked impressive." for p in previews[:240]],
                     index=[p["article_id"] for p in previews[:240]])
    assert (built.loc[good.index, "tone_a"] == np.where(good, 1.0, -1.0)).all()
    assert (built.loc[good.index, "absence_b"] == 100 / 4).all()  # "are without three players": 1 in 4
    assert built.loc[good.index, "nb_home_vs_draw"].notna().all()
    assert built.loc[5000, "tone_a"] == 1.0 and pd.isna(built.loc[5000, "absence_a"])
    assert pd.isna(built.loc[5000, "nb_a_vs_b"])  # not matched to a game
    assert (paths["FEATURES_CSV"].parent / "dictionary_review.csv").exists()
