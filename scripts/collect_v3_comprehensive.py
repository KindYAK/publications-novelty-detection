"""V3: Comprehensive collection — ALL arXiv papers in target categories, 2016-2026.

Strategy:
  1. Use arXiv API to list EVERY paper in cs.CL, cs.LG, cs.CV, cs.AI
     - Split large categories/years into quarters to stay under arXiv 30K limit
     - submittedDate format: YYYYMMDDHHMM (no T separator)
     - Conservative rate limiting: 10s cooldown between quarters
  2. Use Semantic Scholar BATCH endpoint (POST /paper/batch, 500 IDs/req) for citations
  3. Save incrementally (resume-safe: per-category-year-quarter caches)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import arxiv
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.AI"]
YEAR_START = 2016
YEAR_END = 2026

OUTPUT_DIR = Path("data/processed/v3")

# Quarters for splitting large result sets
QUARTERS = [
    ("0101", "0401"),  # Q1: Jan 1 - Mar 31
    ("0401", "0701"),  # Q2: Apr 1 - Jun 30
    ("0701", "1001"),  # Q3: Jul 1 - Sep 30
    ("1001", "1231"),  # Q4: Oct 1 - Dec 31
]

# arXiv rate limiting
ARXIV_PAGE_SIZE = 1000
ARXIV_DELAY_SECONDS = 5.0
ARXIV_RETRIES = 5
ARXIV_QUARTER_COOLDOWN = 10  # seconds between quarter fetches

# Semantic Scholar
SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS = "title,abstract,year,venue,publicationDate,citationCount,influentialCitationCount,externalIds,s2FieldsOfStudy,openAccessPdf"
SS_BATCH_SIZE = 450
SS_REQUEST_INTERVAL = 1.1


# ═══════════════════════════════════════════════════════════════
# STEP 1: Collect arXiv paper IDs
# ═══════════════════════════════════════════════════════════════

def collect_arxiv_chunk(category: str, date_from: str, date_to: str, max_results: int = 30000) -> list[dict]:
    """Fetch papers for a category within a date range with robust retry.

    date_from/date_to format: YYYYMMDD (we append HHMM for arXiv API).
    Returns list of paper dicts, or empty list on persistent failure.
    """
    query = f"cat:{category} AND submittedDate:[{date_from}0000 TO {date_to}2359]"

    papers = []
    for attempt in range(ARXIV_RETRIES):
        client = arxiv.Client(
            page_size=ARXIV_PAGE_SIZE,
            delay_seconds=ARXIV_DELAY_SECONDS,
            num_retries=3,
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        try:
            for result in client.results(search):
                arxiv_id = result.entry_id.split("/abs/")[-1]
                if "v" in arxiv_id:
                    arxiv_id = arxiv_id.rsplit("v", 1)[0]

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": result.title,
                    "published": result.published.isoformat() if result.published else None,
                    "categories": [c for c in result.categories],
                    "abstract": result.summary,
                })
            # Success — got all results
            return papers

        except arxiv.HTTPError as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                logger.warning("arXiv 429 rate limit (attempt %d/%d, got %d so far). Waiting %ds...",
                               attempt + 1, ARXIV_RETRIES, len(papers), wait)
                time.sleep(wait)
            elif "500" in str(e):
                wait = 15 * (attempt + 1)
                logger.warning("arXiv 500 server error (attempt %d/%d, got %d so far). Waiting %ds...",
                               attempt + 1, ARXIV_RETRIES, len(papers), wait)
                time.sleep(wait)
            else:
                logger.error("arXiv unexpected error (attempt %d/%d): %s", attempt + 1, ARXIV_RETRIES, e)
                time.sleep(10)

        except Exception as e:
            logger.error("arXiv fetch failed (attempt %d/%d) after %d results: %s",
                         attempt + 1, ARXIV_RETRIES, len(papers), e)
            time.sleep(10 * (attempt + 1))

    logger.error("arXiv fetch GAVE UP after %d attempts. Returning %d partial results.", ARXIV_RETRIES, len(papers))
    return papers


def collect_category_year(category: str, year: int) -> list[dict]:
    """Collect all papers for a category/year, splitting into quarters if needed."""
    cat_safe = category.replace(".", "_")

    # Check if full-year cache exists (from previous successful run)
    full_cache = OUTPUT_DIR / f"arxiv_ids_{cat_safe}_{year}.json"
    if full_cache.exists():
        with open(full_cache, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) > 0:
            logger.info("Cached: %d papers for %s/%d", len(data), category, year)
            return data
        else:
            logger.info("Empty cache for %s/%d, re-fetching by quarters", category, year)

    # Fetch by quarters
    all_papers = []
    any_failed = False
    for q_idx, (q_start, q_end) in enumerate(QUARTERS, 1):
        quarter_cache = OUTPUT_DIR / f"arxiv_ids_{cat_safe}_{year}_Q{q_idx}.json"

        if quarter_cache.exists():
            with open(quarter_cache, encoding="utf-8") as f:
                quarter_papers = json.load(f)
            logger.info("Cached Q%d: %d papers for %s/%d", q_idx, len(quarter_papers), category, year)
        else:
            date_from = f"{year}{q_start}"
            date_to = f"{year}{q_end}"
            logger.info("Fetching %s/%d Q%d (%s - %s)", category, year, q_idx, date_from, date_to)

            quarter_papers = collect_arxiv_chunk(category, date_from, date_to)
            logger.info("  Got %d papers for %s/%d Q%d", len(quarter_papers), category, year, q_idx)

            if len(quarter_papers) == 0:
                # Don't cache empty results — might be rate limited
                # Exception: future quarters of current year (genuinely empty)
                import datetime
                now = datetime.datetime.now()
                quarter_start_month = int(q_start[:2])
                if year > now.year or (year == now.year and quarter_start_month > now.month):
                    logger.info("  (Future quarter — saving empty cache)")
                    with open(quarter_cache, "w", encoding="utf-8") as f:
                        json.dump([], f)
                else:
                    logger.warning("  NOT caching empty Q%d (likely rate limited)", q_idx)
                    any_failed = True
            else:
                with open(quarter_cache, "w", encoding="utf-8") as f:
                    json.dump(quarter_papers, f, ensure_ascii=False)

            # Cooldown between quarter fetches
            logger.info("  Cooling down %ds before next quarter...", ARXIV_QUARTER_COOLDOWN)
            time.sleep(ARXIV_QUARTER_COOLDOWN)

        all_papers.extend(quarter_papers)

    # Deduplicate within year
    seen = set()
    deduped = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            deduped.append(p)

    # Only save full-year cache if all quarters succeeded
    if not any_failed:
        with open(full_cache, "w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False)
        logger.info("Total for %s/%d: %d papers (deduped from %d)", category, year, len(deduped), len(all_papers))
    else:
        logger.warning("INCOMPLETE %s/%d: %d papers (some quarters failed — no full-year cache)",
                       category, year, len(deduped))

    return deduped


def collect_all_arxiv_ids() -> pd.DataFrame:
    """Collect arXiv IDs for all categories and years."""
    parquet_cache = OUTPUT_DIR / "arxiv_ids_all.parquet"

    if parquet_cache.exists():
        logger.info("Loading cached complete arXiv IDs from %s", parquet_cache)
        return pd.read_parquet(parquet_cache)

    all_papers = []
    for category in CATEGORIES:
        logger.info("=== Category: %s ===", category)
        for year in range(YEAR_START, YEAR_END + 1):
            papers = collect_category_year(category, year)
            for p in papers:
                p["source_category"] = category
                p["source_year"] = year
            all_papers.extend(papers)

    df = pd.DataFrame(all_papers)

    # Deduplicate across categories (papers often cross-listed)
    before = len(df)
    df = df.drop_duplicates(subset="arxiv_id", keep="first").reset_index(drop=True)
    logger.info("Cross-category dedup: %d -> %d unique papers", before, len(df))

    df.to_parquet(parquet_cache, index=False)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: Enrich with Semantic Scholar citations
# ═══════════════════════════════════════════════════════════════

def enrich_batch_ss(arxiv_ids: list[str]) -> list[dict | None]:
    """POST /paper/batch with ArXiv IDs."""
    payload = {"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]}
    params = {"fields": SS_FIELDS}

    for attempt in range(5):
        try:
            resp = requests.post(SS_BATCH_URL, json=payload, params=params, timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning("SS rate limited, waiting %d sec", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("SS batch failed (attempt %d): %s", attempt, e)
            time.sleep(5 * (attempt + 1))

    return [None] * len(arxiv_ids)


def enrich_with_citations(arxiv_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich all papers with SS citation data. Resume-safe."""
    enriched_cache = OUTPUT_DIR / "enriched.parquet"
    if enriched_cache.exists():
        logger.info("Loading cached enriched data")
        return pd.read_parquet(enriched_cache)

    all_ids = arxiv_df["arxiv_id"].tolist()
    total = len(all_ids)

    # Load partial progress
    partial_cache = OUTPUT_DIR / "ss_partial.json"
    ss_data: dict[str, dict] = {}
    if partial_cache.exists():
        with open(partial_cache, encoding="utf-8") as f:
            ss_data = json.load(f)
        logger.info("Resuming SS enrichment: %d/%d done", len(ss_data), total)

    remaining = [aid for aid in all_ids if aid not in ss_data]
    logger.info("SS enrichment: %d remaining of %d total", len(remaining), total)

    batches = [remaining[i:i + SS_BATCH_SIZE] for i in range(0, len(remaining), SS_BATCH_SIZE)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="SS enrichment")):
        results = enrich_batch_ss(batch)
        if results:
            for arxiv_id, result in zip(batch, results):
                ss_data[arxiv_id] = result if result else {"not_found": True}

        if (batch_idx + 1) % 25 == 0:
            with open(partial_cache, "w", encoding="utf-8") as f:
                json.dump(ss_data, f, ensure_ascii=False)
            logger.info("SS progress: %d/%d", len(ss_data), total)

        time.sleep(SS_REQUEST_INTERVAL)

    # Final save
    with open(partial_cache, "w", encoding="utf-8") as f:
        json.dump(ss_data, f, ensure_ascii=False)

    # Build enriched DataFrame
    records = []
    for _, row in arxiv_df.iterrows():
        aid = row["arxiv_id"]
        rec = {
            "arxiv_id": aid,
            "title": row["title"],
            "abstract": row["abstract"],
            "published": row["published"],
            "categories": row["categories"],
            "source_category": row["source_category"],
        }

        ss = ss_data.get(aid, {})
        if ss and not ss.get("not_found"):
            ext = ss.get("externalIds") or {}
            fos = ss.get("s2FieldsOfStudy") or []
            oa = ss.get("openAccessPdf") or {}
            rec.update({
                "ss_paper_id": ss.get("paperId"),
                "year": ss.get("year"),
                "venue": ss.get("venue") or "",
                "publication_date": ss.get("publicationDate"),
                "citation_count": ss.get("citationCount") or 0,
                "influential_citation_count": ss.get("influentialCitationCount") or 0,
                "doi": ext.get("DOI"),
                "fields_of_study": [f["category"] for f in fos],
                "open_access_url": oa.get("url") or None,
                "ss_found": True,
            })
        else:
            rec.update({
                "ss_paper_id": None, "year": None, "venue": "",
                "publication_date": None, "citation_count": 0,
                "influential_citation_count": 0, "doi": None,
                "fields_of_study": [], "open_access_url": None,
                "ss_found": False,
            })
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(enriched_cache, index=False)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: Audit
# ═══════════════════════════════════════════════════════════════

def audit(df: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print("V3 COMPREHENSIVE DATASET")
    print(f"{'=' * 70}")
    print(f"Total unique papers: {len(df)}")
    print(f"SS matched:          {df['ss_found'].sum()} ({df['ss_found'].mean()*100:.1f}%)")
    print(f"With abstracts:      {df['abstract'].notna().sum()}")

    print(f"\nPer source category:")
    for cat in CATEGORIES:
        n = (df["source_category"] == cat).sum()
        print(f"  {cat}: {n:>7}")

    print(f"\nPer year:")
    for y in range(YEAR_START, YEAR_END + 1):
        n = (df["year"] == y).sum()
        pct = n / max(len(df), 1) * 100
        bar = "#" * max(1, int(pct / 2))
        print(f"  {y}: {n:>7} ({pct:4.1f}%) {bar}")

    m = df[df["ss_found"]].copy()
    if len(m):
        bins = [0, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, 50000, 999999]
        labels = ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100-499", "500-1k", "1k-5k", "5k-50k", "50k+"]
        m["cb"] = pd.cut(m["citation_count"], bins=bins, labels=labels, right=False)
        print(f"\nCitations (n={len(m)}):")
        for lb in labels:
            c = (m["cb"] == lb).sum()
            pct = c / len(m) * 100
            bar = "#" * max(1, int(pct / 2))
            print(f"  {lb:>8}: {c:>7} ({pct:5.1f}%) {bar}")

        print(f"\n  Mean:   {m['citation_count'].mean():.1f}")
        print(f"  Median: {m['citation_count'].median():.1f}")
        print(f"  Max:    {m['citation_count'].max()}")

    print(f"\nTop 20:")
    for _, r in df.nlargest(20, "citation_count").iterrows():
        print(f"  [{r['citation_count']:>6}] ({r.get('year','?')}) [{r['source_category']}] {str(r['title'])[:60]}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 1: Collecting arXiv paper IDs")
    arxiv_df = collect_all_arxiv_ids()
    logger.info("Total unique arXiv papers: %d", len(arxiv_df))

    logger.info("STEP 2: Enriching with Semantic Scholar")
    enriched_df = enrich_with_citations(arxiv_df)

    logger.info("STEP 3: Saving")
    enriched_df.to_parquet(OUTPUT_DIR / "dataset_v3.parquet", index=False)
    csv_df = enriched_df.copy()
    csv_df["categories"] = csv_df["categories"].apply(lambda x: ";".join(x) if isinstance(x, list) else str(x))
    csv_df["fields_of_study"] = csv_df["fields_of_study"].apply(lambda x: ";".join(x) if isinstance(x, list) else str(x))
    csv_df.to_csv(OUTPUT_DIR / "dataset_v3.csv", index=False)

    audit(enriched_df)


if __name__ == "__main__":
    main()
