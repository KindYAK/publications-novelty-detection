"""V2 collection: get ALL papers from a narrow scope, including low-citation ones.

Strategy: use Semantic Scholar bulk search WITHOUT min_citation_count filter,
for a specific venue+year or topic+year combination. This gives us the full
distribution needed for BPI validation.

We collect in 3 tiers:
  - Tier 1 (background):   no citation filter, broad query → lots of low-citation papers
  - Tier 2 (mid-impact):   min_citation_count=50 → solid papers
  - Tier 3 (high-impact):  min_citation_count=500 → potential breakthroughs
"""

import json
import logging
import sys
import time

sys.path.insert(0, "src")

import pandas as pd
from novelty_search.data.semantic_scholar import SemanticScholarClient, Paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

client = SemanticScholarClient()
all_papers: dict[str, Paper] = {}


def collect_query(query: str, year_range: str, min_cites: int | None, max_results: int, label: str):
    """Run one query and add to global paper dict."""
    logging.info("[%s] query='%s' year=%s min_cites=%s max=%d", label, query, year_range, min_cites, max_results)
    try:
        papers = client.search_bulk(
            query=query,
            year_range=year_range,
            fields_of_study=["Computer Science"],
            min_citation_count=min_cites,
            max_results=max_results,
        )
        new = sum(1 for p in papers if p.paper_id not in all_papers)
        for p in papers:
            all_papers[p.paper_id] = p
        logging.info("[%s] got %d papers, %d new → total %d", label, len(papers), new, len(all_papers))
    except Exception as e:
        logging.warning("[%s] FAILED: %s", label, e)
    time.sleep(2)


# ═══════════════════════════════════════════════════════════════
# TIER 1: Background papers (NO citation filter)
# These are the "ordinary" papers. Many will have 0-10 citations.
# We need these as negative examples for BPI validation.
# ═══════════════════════════════════════════════════════════════
tier1_queries = [
    # NLP topics — broad, to capture ordinary papers too
    ("machine translation", "2016-2018"),
    ("text classification", "2016-2018"),
    ("named entity recognition", "2016-2018"),
    ("question answering", "2016-2018"),
    ("sentiment analysis", "2016-2018"),
    ("language model", "2016-2018"),
    ("word embeddings", "2016-2018"),
    ("relation extraction", "2016-2018"),
    ("text generation", "2016-2018"),
    ("semantic parsing", "2016-2018"),
    ("reading comprehension", "2016-2018"),
    ("dialogue systems", "2016-2018"),
    # Broader ML
    ("neural network training", "2016-2018"),
    ("recurrent neural network", "2016-2018"),
    ("attention mechanism", "2016-2018"),
    ("transfer learning", "2016-2018"),
]

for query, yr in tier1_queries:
    collect_query(query, yr, min_cites=None, max_results=200, label="TIER1-bg")

# ═══════════════════════════════════════════════════════════════
# TIER 2: Mid-impact papers (≥50 citations)
# Solid, well-known papers. Some are good, some are surveys.
# ═══════════════════════════════════════════════════════════════
tier2_queries = [
    ("natural language processing deep learning", "2016-2018"),
    ("neural network representation learning NLP", "2016-2018"),
    ("sequence to sequence model", "2016-2018"),
    ("convolutional neural network text", "2016-2018"),
    ("generative adversarial network", "2016-2018"),
    ("reinforcement learning", "2016-2018"),
    ("graph neural network", "2016-2018"),
    ("variational autoencoder", "2016-2018"),
]

for query, yr in tier2_queries:
    collect_query(query, yr, min_cites=50, max_results=200, label="TIER2-mid")

# ═══════════════════════════════════════════════════════════════
# TIER 3: High-impact papers (≥500 citations)
# These include known breakthroughs + famous surveys.
# ═══════════════════════════════════════════════════════════════
tier3_queries = [
    ("deep learning", "2016-2018"),
    ("neural network", "2016-2018"),
    ("attention transformer", "2016-2018"),
    ("generative model", "2016-2018"),
    ("natural language processing", "2016-2018"),
]

for query, yr in tier3_queries:
    collect_query(query, yr, min_cites=500, max_results=200, label="TIER3-high")

# ═══════════════════════════════════════════════════════════════
# Build final dataset
# ═══════════════════════════════════════════════════════════════
papers_list = list(all_papers.values())
records = [p.to_dict() for p in papers_list]
df = pd.DataFrame(records)

# Filter: must have abstract
df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)].copy()
df = df.sort_values("citation_count", ascending=False).reset_index(drop=True)

# Save
from pathlib import Path
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "papers_v2_full.csv", index=False)
with open(out / "papers_v2_full.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════════
# Full audit
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("V2 DATASET — FULL DISTRIBUTION")
print(f"{'=' * 70}")
print(f"Total papers (with abstracts): {len(df)}")
print(f"With arXiv IDs:       {df['arxiv_id'].notna().sum()} ({df['arxiv_id'].notna().mean()*100:.1f}%)")
print(f"Year range:           {df['year'].min()} - {df['year'].max()}")

# Citation distribution
bins = [0, 1, 5, 10, 20, 50, 100, 500, 1000, 5000, 200000]
labels = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500-999", "1k-4999", "5000+"]
df["cite_bucket"] = pd.cut(df["citation_count"], bins=bins, labels=labels, right=False)
print("\nCitation distribution:")
for label in labels:
    count = (df["cite_bucket"] == label).sum()
    pct = count / len(df) * 100
    bar = "#" * max(1, int(pct / 2))
    print(f"  {label:>8}: {count:>5} ({pct:5.1f}%) {bar}")

# Year
print("\nYear distribution:")
for year in sorted(df["year"].unique()):
    count = (df["year"] == year).sum()
    print(f"  {int(year)}: {count:>5} ({count/len(df)*100:.1f}%)")

# Top papers
print("\nTop 10 most cited:")
for _, row in df.head(10).iterrows():
    arx = row['arxiv_id'] if pd.notna(row['arxiv_id']) else "no-arx"
    print(f"  [{row['citation_count']:>6}] ({row['year']}) {row['title'][:70]}")

# Low-citation check
low = (df["citation_count"] < 10).sum()
zero = (df["citation_count"] == 0).sum()
print(f"\nPapers with 0 citations:  {zero}")
print(f"Papers with < 10 citations: {low}")
print(f"Papers with >= 500 citations: {(df['citation_count'] >= 500).sum()}")

if low > 0:
    print("\n[OK] Dataset has low-citation papers -- good for BPI validation!")
else:
    print("\n[WARN] Still missing low-citation papers")

print(f"\nSaved to: {out / 'papers_v2_full.csv'}")
