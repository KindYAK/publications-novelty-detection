"""Collect ML/NLP/LLM papers with meaningful citation counts.

Strategy: multiple targeted queries, deduplicated, filtered to papers with abstracts.
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

# Targeted queries that capture ML/NLP/LLM research
QUERIES = [
    # Core NLP
    ("transformer attention mechanism", "2017-2021"),
    ("BERT language model pretraining", "2018-2022"),
    ("word embeddings representation learning", "2013-2019"),
    ("neural machine translation sequence to sequence", "2014-2020"),
    ("text classification sentiment analysis deep learning", "2015-2020"),
    # Core ML
    ("deep learning convolutional neural network image", "2014-2020"),
    ("generative adversarial network GAN", "2014-2020"),
    ("reinforcement learning policy gradient", "2015-2020"),
    ("graph neural network", "2017-2021"),
    # LLM era
    ("large language model GPT generation", "2019-2023"),
    ("prompt engineering in-context learning", "2020-2023"),
    ("instruction tuning RLHF alignment", "2021-2023"),
]

all_papers: dict[str, Paper] = {}

for query, year_range in QUERIES:
    logging.info("Query: '%s' (%s)", query, year_range)
    try:
        papers = client.search_bulk(
            query=query,
            year_range=year_range,
            fields_of_study=["Computer Science"],
            min_citation_count=10,  # filter noise
            max_results=200,
        )
        new = 0
        for p in papers:
            if p.paper_id not in all_papers:
                all_papers[p.paper_id] = p
                new += 1
        logging.info("  Got %d, %d new (total: %d)", len(papers), new, len(all_papers))
    except Exception as e:
        logging.warning("  Failed: %s", e)
    time.sleep(2)  # be nice to the API

# Build DataFrame
records = [p.to_dict() for p in all_papers.values()]
df = pd.DataFrame(records)

# Filter: must have abstract
df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)].copy()
df = df.sort_values("citation_count", ascending=False).reset_index(drop=True)

# Save
from pathlib import Path
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

df.to_csv(out / "papers_ml_nlp.csv", index=False)
with open(out / "papers_ml_nlp.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# Stats
print(f"\n{'=' * 60}")
print(f"ML/NLP/LLM Dataset")
print(f"{'=' * 60}")
print(f"Total papers (with abstracts): {len(df)}")
print(f"With arXiv IDs:       {df['arxiv_id'].notna().sum()}")
print(f"Year range:           {df['year'].min()} - {df['year'].max()}")
print(f"Citation stats:")
print(f"  Mean:               {df['citation_count'].mean():.0f}")
print(f"  Median:             {df['citation_count'].median():.0f}")
print(f"  Max:                {df['citation_count'].max()}")
print(f"  Top 1%:             {df['citation_count'].quantile(0.99):.0f}")
print(f"  Top 5%:             {df['citation_count'].quantile(0.95):.0f}")
print(f"\nTop 15 most cited:")
for _, row in df.head(15).iterrows():
    arx = f"arx:{row['arxiv_id']}" if pd.notna(row['arxiv_id']) else "no-arx"
    print(f"  [{row['citation_count']:>6}] ({row['year']}) [{arx:>20}] {row['title'][:70]}")

print(f"\nSaved to: {out / 'papers_ml_nlp.csv'}")
