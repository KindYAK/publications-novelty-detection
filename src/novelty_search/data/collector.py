"""High-level data collection: build a dataset of papers with citations + text.

Usage:
    uv run python -m novelty_search.data.collector
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .arxiv_text import download_latex_source, extract_text_from_latex
from .semantic_scholar import Paper, SemanticScholarClient

logger = logging.getLogger(__name__)


# Predefined search queries that approximate ML/NLP/LLM research
SEARCH_CONFIGS = {
    "nlp_core": {
        "query": "natural language processing",
        "year_range": "2015-2020",
        "max_results": 200,
    },
    "transformers": {
        "query": "transformer attention mechanism neural",
        "year_range": "2017-2021",
        "max_results": 200,
    },
    "llm": {
        "query": "large language model pretraining",
        "year_range": "2018-2023",
        "max_results": 200,
    },
    "representation_learning": {
        "query": "word embeddings sentence representation learning",
        "year_range": "2013-2019",
        "max_results": 200,
    },
    "ml_general": {
        "query": "deep learning neural network",
        "year_range": "2015-2020",
        "max_results": 200,
    },
}


def collect_papers(
    config_name: str = "nlp_core",
    output_dir: str = "data/processed",
    api_key: str | None = None,
    download_source: bool = False,
) -> pd.DataFrame:
    """Collect papers from Semantic Scholar and optionally arXiv full text.

    Args:
        config_name: One of SEARCH_CONFIGS keys, or "all" to run all.
        output_dir: Where to save the dataset.
        api_key: Semantic Scholar API key (optional, increases rate limit).
        download_source: Whether to download LaTeX source from arXiv.

    Returns:
        DataFrame with paper metadata, citations, and text.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = SemanticScholarClient(api_key=api_key)

    if config_name == "all":
        configs = SEARCH_CONFIGS
    else:
        configs = {config_name: SEARCH_CONFIGS[config_name]}

    all_papers: dict[str, Paper] = {}  # dedupe by paper_id

    for name, cfg in configs.items():
        logger.info("Searching: %s (%s)", name, cfg["query"])
        papers = client.search_bulk(
            query=cfg["query"],
            year_range=cfg.get("year_range"),
            fields_of_study=["Computer Science"],
            min_citation_count=cfg.get("min_citation_count", 0),
            max_results=cfg["max_results"],
        )
        for p in papers:
            all_papers[p.paper_id] = p
        logger.info("  Found %d papers, total unique: %d", len(papers), len(all_papers))

    papers_list = list(all_papers.values())
    logger.info("Total unique papers collected: %d", len(papers_list))

    # Build DataFrame
    records = []
    for p in papers_list:
        rec = p.to_dict()
        rec["has_abstract"] = p.abstract is not None and len(p.abstract) > 50
        rec["has_arxiv"] = p.arxiv_id is not None
        records.append(rec)

    df = pd.DataFrame(records)

    # Optionally download full text from arXiv
    if download_source:
        source_dir = out / "arxiv_source"
        source_dir.mkdir(exist_ok=True)
        full_texts = {}

        arxiv_papers = df[df["has_arxiv"]].head(50)  # limit to 50 for now
        for _, row in tqdm(arxiv_papers.iterrows(), total=len(arxiv_papers), desc="Downloading LaTeX"):
            src = download_latex_source(row["arxiv_id"], source_dir)
            if src:
                text = extract_text_from_latex(src)
                if text:
                    full_texts[row["paper_id"]] = text

        df["full_text"] = df["paper_id"].map(full_texts)
        logger.info("Downloaded full text for %d papers", len(full_texts))

    # Save
    csv_path = out / f"papers_{config_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d papers to %s", len(df), csv_path)

    # Also save as JSON for richer data (lists etc)
    json_path = out / f"papers_{config_name}.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary stats
    print(f"\n{'=' * 60}")
    print(f"Dataset: {config_name}")
    print(f"{'=' * 60}")
    print(f"Total papers:         {len(df)}")
    print(f"With abstracts:       {df['has_abstract'].sum()}")
    print(f"With arXiv IDs:       {df['has_arxiv'].sum()}")
    print(f"Year range:           {df['year'].min()} - {df['year'].max()}")
    print("Citation stats:")
    print(f"  Mean:               {df['citation_count'].mean():.0f}")
    print(f"  Median:             {df['citation_count'].median():.0f}")
    print(f"  Max:                {df['citation_count'].max()}")
    print(f"  Top 1% threshold:   {df['citation_count'].quantile(0.99):.0f}")
    top5 = df.nlargest(5, "citation_count")[["title", "citation_count", "year"]]
    print("\nTop 5 most cited:")
    for _, row in top5.iterrows():
        print(f"  [{row['citation_count']:>6}] ({row['year']}) {row['title'][:80]}")
    print(f"\nSaved to: {csv_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    collect_papers(config_name="nlp_core")
