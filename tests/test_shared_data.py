"""The tables in data/processed/ go on GitHub, so they must never hold Sports Mole text.

These tests check the committed files themselves, so GitHub runs them on every push.
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
TABLES = sorted(PROCESSED.glob("*.csv"))
TEXT_COLUMNS = {"title", "pick_text"}
MAX_WORDS = 10  # team, player and author names are short; a sentence of article text is not


@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.name)
def test_no_article_text(path):
    table = pd.read_csv(path, dtype=str)
    text_columns = [c for c in table.columns if c.startswith("text_") or c in TEXT_COLUMNS]
    assert not text_columns, f"{path.name} has text columns: {text_columns}"

    words = table.stack().str.split().str.len()
    too_long = words[words > MAX_WORDS]
    assert too_long.empty, f"{path.name} has text at (row, column): {list(too_long.index[:5])}"
