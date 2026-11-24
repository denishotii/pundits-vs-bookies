"""Conservative side documents and paired sparse or frozen text features."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import gzip
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse

from ..features import ROLE_WORDS, SENTENCE_BREAK, sentences
from ..matching import ALIASES
from .dataset import _read_source
from .diagnostic_data import SentenceAttribution, _expression, _name_form, _names, side_orientation
from .embeddings import _fingerprint, embed_articles
from .preprocessing import PRIMARY_TEXT_COLUMNS, fit_tfidf, prepare_documents

SIDES = ("home", "away")
LABELS = (*SIDES, "both", "neither")
PLACEHOLDER = "TEAM"
ATTRIBUTION_CONFIG = {
    "schema": 1,
    "names": "Fixture names, curated spellings, and verified fixture-oriented article sides",
    "mask": PLACEHOLDER,
    "policy": "Case-sensitive names, longest span, no standalone shared tokens, no coreference or both-team text",
    "roles": ROLE_WORDS,
    "aliases": ALIASES,
    "sentence_break": SENTENCE_BREAK.pattern,
    "sections": PRIMARY_TEXT_COLUMNS,
}


def aligned_articles(metadata, matches_path, previews_path):
    """Read the two side names per source strictly as fixture metadata."""
    columns = ("article_id", "side_a", "side_b")
    sides = []
    for path, prefix in ((previews_path, "source_"), (matches_path, "matched_")):
        frame = _read_source(path, columns).set_index("article_id").loc[metadata.index].add_prefix(prefix)
        sides.append(frame)
    return metadata.join(sides)


class TeamAttribution(SentenceAttribution):
    def __init__(self, home, away, side_a=None, side_b=None):
        self.orientation = side_orientation(side_a, side_b, home, away)
        oriented = {"ordered": (side_a, side_b), "reversed": (side_b, side_a)}
        source_home, source_away = oriented.get(self.orientation, (None, None))
        super().__init__(home, away, source_home, source_away)
        left, right = _names(home, source_home), _names(away, source_away)
        left_folded, right_folded = ({name.casefold() for name in names} for names in (left, right))
        shared = (left_folded & right_folded) | {
            token for token in set(" ".join(left_folded).split()) & set(" ".join(right_folded).split())
            if len(token) >= 3 or token in left_folded or token in right_folded
        }
        self.shared = _expression(shared, re.I)
        self.names = [_expression({name for name in names if name.casefold() not in shared})
                      for names in (left, right)]

    def proposal(self, sentence):
        label, shared, unresolved, conflict = self.classify(sentence)
        if label in SIDES and (shared or conflict):
            label = "neither"
        return label, shared, unresolved, conflict

    def mask(self, sentence):
        sentence = unicodedata.normalize("NFC", sentence)
        normalized = _name_form(sentence)
        mentions = sorted(self.name_mentions(normalized))
        if not mentions:
            return sentence
        # Track source spans through accent folding so punctuation and possessives survive.
        chars, origins = [], []
        for i, char in enumerate(sentence):
            for folded in unicodedata.normalize("NFKD", char):
                if not unicodedata.combining(folded):
                    chars.append(folded)
                    origins.append(i)
        starts, ends, offset = {}, {}, 0
        for token in re.finditer(r"[^\W_]+", "".join(chars)):
            starts[offset] = origins[token.start()]
            offset += len(token.group())
            ends[offset] = origins[token.end() - 1] + 1
            offset += 1
        parts, last = [], 0
        for start, end, _ in mentions:
            first, after = starts[start], ends[end]
            if first >= last:
                parts.extend((sentence[last:first], PLACEHOLDER))
                last = after
        return "".join((*parts, sentence[last:]))


@dataclass
class SideDocuments:
    home: pd.DataFrame
    away: pd.DataFrame
    coverage: pd.DataFrame
    fingerprint: str
    cache_hit: bool

    def reindex(self, index):
        return SideDocuments(self.home.loc[index], self.away.loc[index], self.coverage.loc[index],
                             self.fingerprint, self.cache_hit)


def _attribute_row(row):
    rule = TeamAttribution(row.home_team, row.away_team, row.source_side_a, row.source_side_b)
    counts = Counter({f"sentences_{label}": 0 for label in LABELS})
    counts.update({key: 0 for key in ("ambiguous", "unresolved", "conflicting")})
    documents = {side: [] for side in SIDES}
    for section in PRIMARY_TEXT_COLUMNS:
        current = {side: [] for side in SIDES}
        for sentence in sentences(getattr(row, section)):
            label, ambiguous, unresolved, conflict = rule.proposal(sentence)
            counts[f"sentences_{label}"] += 1
            counts["ambiguous"] += ambiguous
            counts["unresolved"] += unresolved
            counts["conflicting"] += conflict
            if label in SIDES:
                current[label].append(rule.mask(sentence))
        for side in SIDES:
            documents[side].append(" ".join(current[side]))
    coverage = {
        **counts,
        "source_orientation": rule.orientation,
        "matched_orientation": side_orientation(row.matched_side_a, row.matched_side_b, row.home_team, row.away_team),
        "source_matches_export": bool(pd.notna(row.source_side_a) and pd.notna(row.source_side_b)
                                      and row.source_side_a == row.matched_side_a
                                      and row.source_side_b == row.matched_side_b)
        if pd.notna(row.matched_side_a) and pd.notna(row.matched_side_b) else False,
    }
    return documents["home"], documents["away"], coverage


def side_documents(articles, text, cache_path: Path):
    columns = ["home_team", "away_team", "source_side_a", "source_side_b", "matched_side_a", "matched_side_b"]
    if not articles.index.is_unique or not articles.index.equals(text.index):
        raise ValueError("Attribution requires identical, unique fixture and text IDs")
    inputs = articles[columns].join(text[list(PRIMARY_TEXT_COLUMNS)]).fillna("")
    rows = list(inputs.itertuples())
    key = _fingerprint((ATTRIBUTION_CONFIG, rows))
    cached = None
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            stored_key, stored_rows = json.load(stream)
        if stored_key == key:
            cached = stored_rows
    cache_hit = cached is not None
    if cached is None:
        cached = [_attribute_row(row) for row in rows]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
                json.dump((key, cached), stream, ensure_ascii=False)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    home, away, coverage = zip(*cached, strict=True)
    return SideDocuments(*(pd.DataFrame(values, index=articles.index, columns=PRIMARY_TEXT_COLUMNS)
                           for values in (home, away)), pd.DataFrame(coverage, index=articles.index), key, cache_hit)


def comparison_features(home, away, missing):
    """Difference, mean and two missing flags in fixture order, retaining sparse input."""
    if home.shape != away.shape or missing.shape != (home.shape[0], 2):
        raise ValueError("Side features and missing flags must align")
    blocks = (home - away, (home + away) * 0.5, missing)
    if sparse.issparse(home):
        return sparse.hstack(blocks, format="csr")
    return np.column_stack(blocks)


def side_tfidf(documents, train, valid):
    home, away = documents
    if not home.index.equals(away.index):
        raise ValueError("Home and away document IDs must align")
    vectorizer, training = fit_tfidf(pd.concat([home.iloc[train], away.iloc[train]]))
    valid_documents = pd.concat([home.iloc[valid], away.iloc[valid]])
    validation = vectorizer.transform(valid_documents)

    def combine(matrix, positions):
        n_rows = len(positions)
        missing = np.column_stack([side.iloc[positions].eq("").to_numpy() for side in documents])
        return comparison_features(matrix[:n_rows], matrix[n_rows:], missing)

    return vectorizer, combine(training, train), combine(validation, valid)


def team_embeddings(documents, encoder, cache_path):
    signature = _fingerprint(ATTRIBUTION_CONFIG)
    sections = pd.concat([documents.home, documents.away], ignore_index=True)
    ids = [f"{signature}:{side}:{article_id}" for side in SIDES for article_id in documents.home.index]
    embedded = embed_articles(sections.set_axis(ids), encoder, cache_path)
    n_rows = len(documents.home)
    vectors = embedded.vectors.to_numpy()
    missing = np.column_stack([prepare_documents(frame).eq("").to_numpy()
                               for frame in (documents.home, documents.away)])
    return comparison_features(vectors[:n_rows], vectors[n_rows:], missing), embedded.statistics


def prepare_review(source_path, output_path, articles, *, per_label=48):
    """Select a deterministic subset of the Phase 06A review and preserve human labels."""
    review = pd.read_csv(source_path, dtype="string", keep_default_na=False)
    prior = (pd.read_csv(output_path, dtype="string", keep_default_na=False).set_index("review_id")["human_label"]
             if output_path.exists() else pd.Series(dtype="string"))
    proposals = []
    for row in review.itertuples():
        fixture = articles.loc[row.article_id]
        rule = TeamAttribution(fixture.home_team, fixture.away_team, fixture.source_side_a, fixture.source_side_b)
        proposals.append(rule.proposal(row.sentence))
    review[["phase06b_proposal", "phase06b_ambiguous", "phase06b_unresolved", "phase06b_conflicting"]] = pd.DataFrame(
        proposals, index=review.index,
    )
    identity = ["article_id", "section", "sentence"]
    occurrence = review.groupby(identity, sort=False).cumcount()
    review["review_id"] = [
        sha256(json.dumps((*values, i), ensure_ascii=False).encode()).hexdigest()
        for i, values in zip(occurrence, review[identity].itertuples(index=False, name=None), strict=True)
    ]
    previous_labels = review["review_id"].map(prior).astype("string").replace("", pd.NA)
    review["human_label"] = previous_labels.fillna(review["human_label"])
    ranked = review.sort_values("review_id")
    # Interleave seasons and reserve representation for ambiguous and unresolved cues.
    flagged = ranked[["phase06b_ambiguous", "phase06b_conflicting", "phase06b_unresolved"]].any(axis=1)
    ranked = ranked.assign(flagged=flagged)
    ranked["season_rank"] = ranked.groupby(["phase06b_proposal", "season", "flagged"], sort=False).cumcount()
    ranked = ranked.sort_values(["season_rank", "flagged", "review_id"], ascending=[True, False, True])
    selected = ranked.groupby("phase06b_proposal", sort=False).head(per_label)
    needed = len(LABELS) * per_label - len(selected)
    selected = pd.concat([selected, ranked.drop(selected.index).head(needed)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.drop(columns=["flagged", "season_rank"]).to_csv(output_path, index=False)
    annotated = selected["human_label"].str.strip().str.lower()
    valid = annotated.isin(LABELS)
    return {
        "sample_rows": len(selected), "proposals": selected["phase06b_proposal"].value_counts().to_dict(),
        "recognized_human_labels": int(valid.sum()),
        "unrecognized_nonempty_labels": int((annotated.ne("") & ~valid).sum()),
        "proposal_agreement_on_labeled_subset": (
            float(selected.loc[valid, "phase06b_proposal"].eq(annotated[valid]).mean()) if valid.any() else None
        ),
    }
