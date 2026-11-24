
import pandas as pd
import numpy as np

football = pd.read_csv(
    "data/processed/matched_football.csv",
    dtype={"season": str}
)
features = pd.read_csv("data/processed/features.csv")
previews = pd.read_csv("data/interim/previews.csv")

print("\n=== MATCHED DATA ===")
print(football.shape)
print(football.columns.tolist())

print("\n=== DUPLICATES ===")
print(football["article_id"].duplicated().sum())

print("\n=== SEASONS ===")
print(football["season"].value_counts().sort_index())

print("\n=== REAL TARGET DISTRIBUTION ===")
print(football["result"].value_counts(normalize=True))

print("\n=== MISSING VALUES ===")
print(football.isna().sum())

print("\n=== TIMESTAMP CHECK ===")
published = pd.to_datetime(football["published"], utc=True)
kickoff = pd.to_datetime(football["kickoff"], utc=True)

hours = (kickoff - published).dt.total_seconds() / 3600

print(hours.describe())
print("Published at or after kickoff:", (hours <= 0).sum())
print("Published less than 1h before:", ((hours > 0) & (hours < 1)).sum())

print("\n=== JOIN COVERAGE ===")
print("Missing previews:",
      (~football["article_id"].isin(previews["article_id"])).sum())
print("Missing features:",
      (~football["article_id"].isin(features["article_id"])).sum())

print("\n=== ODDS QUALITY ===")
odds_cols = [
    c for c in football.columns if "odds" in c
]

for col in odds_cols:
    values = football[col].dropna()
    print(col, {
        "missing": football[col].isna().sum(),
        "invalid": (values <= 1).sum(),
        "min": values.min(),
        "max": values.max()
    })

print("\n=== TARGET BY SEASON ===")
print(pd.crosstab(
    football["season"],
    football["result"],
    normalize="index"
).round(3))
