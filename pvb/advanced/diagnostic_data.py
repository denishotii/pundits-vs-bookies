"""Development-only cohorts and conservative sentence attribution."""

from collections import Counter
from hashlib import sha256
import re
import unicodedata

import numpy as np
import pandas as pd

from ..analysis import early_odds_time
from ..features import ROLE_WORDS, sentences
from ..matching import ALIASES, same_team
from .dataset import ODDS_COLUMNS, _read_source, load_football
from .market import market_probabilities
from .preprocessing import PRIMARY_TEXT_COLUMNS
from .validation import FOOTBALL_SPLITS

DEVELOPMENT = tuple(FOOTBALL_SPLITS["train"])
TIMING_BUFFER_HOURS = 3
REVIEW_SALT = "pvb-phase-06a-attribution-v1"
TIMING_GROUPS = ("pre_snapshot", "pre_snapshot_buffer_3h", "post_snapshot_retrospective")


def load_development(market, matches_path, previews_path, *, include_outcomes=True):
    data = load_football(
        market, matches_path=matches_path, previews_path=previews_path,
        seasons=DEVELOPMENT, include_outcomes=include_outcomes,
    )
    frames = [data.text, data.numerical] + ([data.target] if include_outcomes else [])
    if any(not frame.index.equals(data.metadata.index) for frame in frames):
        raise ValueError("Numerical inputs, text and target must align with fixture metadata")
    if not data.metadata["season"].isin(DEVELOPMENT).all():
        raise ValueError("Diagnostics require development-only inputs")
    data.metadata = data.metadata.set_index("article_id")
    if not data.metadata.index.is_unique:
        raise ValueError("Development article IDs must be unique")
    data.text, data.numerical = (frame.set_axis(data.metadata.index) for frame in (data.text, data.numerical))
    if include_outcomes:
        data.target = data.target.set_axis(data.metadata.index)
    return data


def movement_timing(published, kickoff, snapshot):
    """Classify strict publication timing, treating midnight UTC as date-only."""
    date_only = published.eq(published.dt.normalize())
    available = published.where(~date_only, published + pd.Timedelta(days=1))

    def before(cutoff):
        return available.lt(cutoff) | (date_only & available.eq(cutoff))

    usable = snapshot.lt(kickoff) & before(kickoff)
    masks = pd.DataFrame({
        "snapshot_not_before_kickoff": snapshot.ge(kickoff),
        "pre_snapshot": usable & before(snapshot),
        "pre_snapshot_buffer_3h": usable & before(snapshot - pd.Timedelta(hours=TIMING_BUFFER_HOURS)),
        "post_snapshot_retrospective": usable & published.gt(snapshot),
    })
    masks["timing_uncertain"] = ~masks[[
        "snapshot_not_before_kickoff", "pre_snapshot", "post_snapshot_retrospective",
    ]].any(axis=1)
    return masks, available


def movement_inputs(closing, matches_path):
    """Match valid dual snapshots via IDs without using outcomes."""
    raw = _read_source(matches_path, ("article_id", *ODDS_COLUMNS["early"])).set_index("article_id")
    early = raw.loc[closing.metadata.index].apply(pd.to_numeric, errors="coerce").astype(float)
    valid = (np.isfinite(early) & early.gt(1)).all(axis=1)
    metadata = closing.metadata.loc[valid].copy()
    snapshot = early_odds_time(metadata["kickoff"])
    timing, available = movement_timing(metadata["published"], metadata["kickoff"], snapshot)
    coverage = []
    for season in (*DEVELOPMENT, "pooled"):
        eligible = closing.metadata if season == "pooled" else closing.metadata[closing.metadata["season"].eq(season)]
        matched = timing.loc[timing.index.intersection(eligible.index)]
        coverage.append({
            "season": season, "closing_eligible": len(eligible), "valid_dual_snapshots": len(matched),
            "invalid_or_missing_early_odds": len(eligible) - len(matched), **matched.sum().astype(int).to_dict(),
            **{f"{group}_fraction_of_dual": matched[group].mean() for group in TIMING_GROUPS},
        })
    partitions = {}
    for group in TIMING_GROUPS:
        selected = metadata.loc[timing[group]].copy()
        origin = available if group == "post_snapshot_retrospective" else snapshot
        selected["prediction_at"] = origin.loc[selected.index]
        early_probs = market_probabilities(early.loc[selected.index], "early")
        close_probs = market_probabilities(closing.numerical.loc[selected.index], "closing")
        partitions[group] = (selected, early_probs, close_probs)
    return partitions, pd.DataFrame(coverage)


def _name_form(value):
    # Preserve capitalization for proper-name evidence, fold accents and punctuation.
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", value))


def side_orientation(side_a, side_b, home, away):
    if any(pd.isna(value) or not _name_form(value) for value in (side_a, side_b, home, away)):
        return "missing"
    ordered = same_team(home, side_a) and same_team(away, side_b)
    reversed_order = same_team(home, side_b) and same_team(away, side_a)
    if ordered and reversed_order:
        return "ambiguous"
    if ordered:
        return "ordered"
    return "reversed" if reversed_order else "unmatched"


def _names(fixture_name, article_name):
    candidates = [fixture_name, ALIASES.get(fixture_name, fixture_name), article_name]
    return {_name_form(value) for value in candidates if isinstance(value, str) and _name_form(value)}


def _expression(names, flags=0):
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(names, key=lambda name: (-len(name), name)))) + r")\b",
                      flags)


UNRESOLVED_REFERENCE = re.compile(r"\b(?:they|their|them|the (?:hosts?|visitors?|side|team|club|outfit))\b", re.I)


class SentenceAttribution:
    """Use explicit fixture names and football roles without learned or carried-forward aliases."""

    def __init__(self, home, away, source_home=None, source_away=None):
        left, right = _names(home, source_home), _names(away, source_away)
        left_words = set(" ".join(left).split())
        right_words = set(" ".join(right).split())
        shared_names = {name.casefold() for name in left} & {name.casefold() for name in right}
        shared_tokens = {
            word for word in left_words & right_words if len(word) >= 3
        }
        self.shared = _expression(shared_tokens | shared_names, re.I)
        self.names = [
            _expression({name for name in names if name.casefold() not in shared_names})
            for names in (left, right)
        ]
        self.roles = [_expression(set(ROLE_WORDS[side]), re.I) for side in ("a", "b")]

    def name_mentions(self, normalized):
        mentions = [(match.start(), match.end(), side) for side, pattern in enumerate(self.names) if pattern
                    for match in pattern.finditer(normalized)]
        # A short name contained in the other team's longer name is not a second mention.
        return [
            (start, end, side) for start, end, side in mentions
            if not any(i <= start and end <= j and (i < start or end < j) for i, j, _ in mentions)
        ]

    def classify(self, sentence):
        normalized = _name_form(sentence)
        mentions = self.name_mentions(normalized)
        named_sides = {side for _, _, side in mentions}
        roles = {side for side, pattern in enumerate(self.roles) if pattern.search(normalized)}
        common = bool(self.shared and any(
            not any(i <= match.start() and match.end() <= j for i, j, _ in mentions)
            for match in self.shared.finditer(normalized)
        ))
        sides = named_sides | roles
        label = {frozenset(): "neither", frozenset({0}): "home", frozenset({1}): "away",
                 frozenset({0, 1}): "both"}[frozenset(sides)]
        unresolved = not sides and bool(UNRESOLVED_REFERENCE.search(normalized))
        conflict = bool(named_sides and roles and named_sides.isdisjoint(roles))
        return label, common, unresolved, conflict


def review_sample(metadata):
    """Select 50 IDs by salted hash, stratified over the four development seasons."""
    selected = set()
    for season, count in zip(DEVELOPMENT, (13, 13, 12, 12), strict=True):
        ids = metadata.index[metadata["season"].eq(season)]
        ordered = sorted(ids, key=lambda article_id: sha256(f"{REVIEW_SALT}:{article_id}".encode()).digest())
        selected.update(ordered[:count])
    return selected


def attribution_audit(closing, matches_path, previews_path):
    columns = ("article_id", "side_a", "side_b")
    source = _read_source(previews_path, columns).set_index("article_id").loc[closing.metadata.index]
    matched = _read_source(matches_path, columns).set_index("article_id").loc[closing.metadata.index]
    sides = source.add_prefix("source_").join(matched.add_prefix("matched_"))
    fixtures = closing.metadata.join(sides).join(closing.text.loc[:, list(PRIMARY_TEXT_COLUMNS)])
    sample = review_sample(fixtures)
    counters = {season: Counter() for season in DEVELOPMENT}
    rows = []
    for row in fixtures.itertuples():
        counts = counters[row.season]
        counts["articles"] += 1
        home, away = row.home_team, row.away_team
        orientation = side_orientation(row.source_side_a, row.source_side_b, home, away)
        counts[f"source_alignment_{orientation}"] += 1
        counts[f"matched_alignment_{side_orientation(row.matched_side_a, row.matched_side_b, home, away)}"] += 1
        agrees = (
            not any(pd.isna(value) for value in (row.source_side_a, row.source_side_b, row.matched_side_a, row.matched_side_b))
            and row.source_side_a == row.matched_side_a and row.source_side_b == row.matched_side_b
        )
        counts["source_matches_exported_sides"] += agrees
        oriented = (None, None)
        if orientation in ("ordered", "reversed"):
            oriented = (row.source_side_a, row.source_side_b)
            if orientation == "reversed":
                oriented = oriented[::-1]
        attribution = SentenceAttribution(home, away, *oriented)
        labels = Counter()
        for section in PRIMARY_TEXT_COLUMNS:
            for sentence in sentences(getattr(row, section)):
                label, ambiguous, unresolved, conflict = attribution.classify(sentence)
                labels[label] += 1
                counts[f"sentences_{label}"] += 1
                counts["sentences_ambiguous_shared_name"] += ambiguous
                counts["sentences_unresolved_generic_reference"] += unresolved
                counts["sentences_conflicting_role_and_name"] += conflict
                if row.Index in sample:
                    rows.append({
                        "article_id": row.Index, "season": row.season, "home_team": home, "away_team": away,
                        "section": section, "sentence": sentence, "proposed_attribution": label,
                        "ambiguous_name": ambiguous, "unresolved_reference": unresolved,
                        "conflicting_reference": conflict, "human_label": "",
                    })
        counts["articles_without_primary_sentences"] += not labels
        counts["articles_missing_home_only"] += labels["home"] == 0
        counts["articles_missing_away_only"] += labels["away"] == 0
        counts["articles_missing_either_exclusive"] += labels["home"] == 0 or labels["away"] == 0
        counts["articles_missing_both_exclusive"] += labels["home"] == 0 and labels["away"] == 0
        counts["articles_missing_either_including_both"] += (
            labels["home"] + labels["both"] == 0 or labels["away"] + labels["both"] == 0
        )
        counts["sampled_articles"] += row.Index in sample
    pooled = sum(counters.values(), Counter())
    keys = sorted({key for counts in counters.values() for key in counts})
    summary = pd.DataFrame([
        {"season": season, **{key: counts[key] for key in keys}}
        for season, counts in [*counters.items(), ("pooled", pooled)]
    ])
    summary["sentences"] = summary[[f"sentences_{label}" for label in ("home", "away", "both", "neither")]].sum(axis=1)
    return summary, pd.DataFrame(rows)
