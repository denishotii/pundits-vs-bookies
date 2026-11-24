
import pandas as pd

df = pd.read_csv("data/interim/previews.csv")

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== SPORTS ===")
print(df["sport"].value_counts())

print("\n=== DUPLICATES ===")
print(df["article_id"].duplicated().sum())

print("\n=== MISSING VALUES (%) ===")
print((df.isna().mean() * 100).round(2))

print("\n=== PUBLICATION DATES ===")
dates = pd.to_datetime(df["published"], utc=True, errors="coerce")
print("Min:", dates.min())
print("Max:", dates.max())
print("Invalid dates:", dates.isna().sum())

print("\n=== FOOTBALL TARGET DISTRIBUTION ===")
print(df[df["sport"] == "football"]["pick_result"].value_counts())

print("\n=== TEXT LENGTH ===")
print(df["word_count"].describe())
