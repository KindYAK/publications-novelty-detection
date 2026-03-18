"""Audit the collected dataset: distribution, biases, coverage."""

import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/papers_ml_nlp.csv")

print("=" * 70)
print("DATASET AUDIT")
print("=" * 70)

# 1. Citation distribution
print("\n--- Citation Distribution ---")
print(f"Total papers: {len(df)}")
print(f"Min citations: {df['citation_count'].min()}")
print(f"Max citations: {df['citation_count'].max()}")
print(f"Mean: {df['citation_count'].mean():.1f}")
print(f"Median: {df['citation_count'].median():.1f}")
print(f"Std: {df['citation_count'].std():.1f}")

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\nPercentiles:")
for p in percentiles:
    val = df['citation_count'].quantile(p / 100)
    print(f"  {p:>3}th: {val:>10.0f}")

# Citation buckets
bins = [0, 10, 20, 50, 100, 500, 1000, 5000, 200000]
labels = ["0-10", "10-20", "20-50", "50-100", "100-500", "500-1k", "1k-5k", "5k+"]
df["cite_bucket"] = pd.cut(df["citation_count"], bins=bins, labels=labels, right=False)
print("\nCitation buckets:")
for label in labels:
    count = (df["cite_bucket"] == label).sum()
    pct = count / len(df) * 100
    bar = "#" * int(pct)
    print(f"  {label:>8}: {count:>5} ({pct:5.1f}%) {bar}")

# 2. Year distribution
print("\n--- Year Distribution ---")
year_counts = df["year"].value_counts().sort_index()
for year, count in year_counts.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct)
    print(f"  {int(year)}: {count:>5} ({pct:5.1f}%) {bar}")

# 3. arXiv coverage
print("\n--- arXiv Coverage ---")
has_arxiv = df["arxiv_id"].notna().sum()
print(f"With arXiv ID: {has_arxiv} ({has_arxiv/len(df)*100:.1f}%)")
print(f"Without arXiv: {len(df) - has_arxiv} ({(len(df)-has_arxiv)/len(df)*100:.1f}%)")

# 4. Abstract coverage
print("\n--- Abstract Coverage ---")
has_abs = df["abstract"].notna().sum()
avg_len = df.loc[df["abstract"].notna(), "abstract"].str.len().mean()
print(f"With abstract: {has_abs} ({has_abs/len(df)*100:.1f}%)")
print(f"Avg abstract length: {avg_len:.0f} chars")

# 5. Venue distribution (top 15)
print("\n--- Top 15 Venues ---")
venues = df["venue"].value_counts().head(15)
for venue, count in venues.items():
    if venue:
        print(f"  {count:>4}x  {venue[:60]}")

# 6. Field of study
print("\n--- Fields of Study ---")
# fields_of_study is stored as string repr of list
fos_counts = {}
for fos_str in df["fields_of_study"].dropna():
    try:
        fields = eval(fos_str) if isinstance(fos_str, str) else fos_str
        for f in fields:
            fos_counts[f] = fos_counts.get(f, 0) + 1
    except Exception:
        pass
for f, c in sorted(fos_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {c:>5}x  {f}")

# 7. PROBLEM: no low-citation papers
print("\n--- BIAS WARNING ---")
low = (df["citation_count"] < 10).sum()
zero = (df["citation_count"] == 0).sum()
print(f"Papers with < 10 citations: {low}")
print(f"Papers with 0 citations: {zero}")
if low < len(df) * 0.1:
    print("WARNING: Dataset is heavily biased toward cited papers!")
    print("For BPI validation we NEED low-citation papers as negative examples.")
    print("The current min_citation_count=10 filter removes them.")
