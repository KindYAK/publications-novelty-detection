"""Collect ALL remaining ACL Anthology papers (not just embeddings).

Fills the gap: conference-only NLP papers not on arXiv.

Sources:
  1. HuggingFace ACL-OCL/acl-anthology-corpus (pre-2022, ~45K new papers, has full text)
  2. acl-anthology Python package (2023+, ~8.6K new papers, needs PDF download)
  3. Semantic Scholar batch API (citation enrichment)

Output:
  data/processed/v3/acl_all.parquet — all new ACL papers
  data/processed/v3/acl_all_fulltext/ — full text files
"""

from __future__ import annotations

import json
import logging
import re
import time
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/v3")
OUT_PARQUET = DATA_DIR / "acl_all.parquet"
FT_DIR = DATA_DIR / "acl_all_fulltext"
PROGRESS_FILE = DATA_DIR / "acl_all_progress.json"

SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS = "title,citationCount,influentialCitationCount,externalIds,openAccessPdf"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def load_existing_titles() -> set[str]:
    """Load all titles from v3 + embeddings supplement."""
    titles = set()
    for f in ["dataset_v3.parquet", "embeddings_acl.parquet"]:
        p = DATA_DIR / f
        if p.exists():
            df = pd.read_parquet(p, columns=["title"])
            titles |= set(df["title"].apply(norm_title).values)
    return titles


def save_progress(data: dict):
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    tmp.replace(PROGRESS_FILE)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════
# STEP 1: HuggingFace ACL Anthology (pre-2022)
# ═══════════════════════════════════════════════════════════════

def collect_hf_all(existing_titles: set[str]) -> list[dict]:
    """Get ALL papers from HF ACL Anthology that we don't have."""
    from huggingface_hub import hf_hub_download

    cache_file = DATA_DIR / "acl_all_step1_hf.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            papers = json.load(f)
        logger.info("Loaded %d papers from cache (step1)", len(papers))
        return papers

    logger.info("Loading HF ACL Anthology parquet...")
    path = hf_hub_download(
        repo_id="ACL-OCL/acl-anthology-corpus",
        filename="acl-publication-info.74k.v2.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    logger.info("Total HF papers: %d", len(df))

    # Filter out papers we already have
    df["_norm"] = df["title"].apply(norm_title)
    df_new = df[~df["_norm"].isin(existing_titles)].copy()
    logger.info("New papers (not in v3 or embeddings): %d", len(df_new))

    papers = []
    for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc="HF papers"):
        papers.append({
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "full_text": row.get("full_text", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "venue": row.get("booktitle") or row.get("journal", ""),
            "acl_id": row.get("acl_id", ""),
            "doi": row.get("doi", ""),
            "citation_count": int(row["numcitedby"]) if pd.notna(row.get("numcitedby")) else 0,
            "corpus_paper_id": str(row.get("corpus_paper_id", "")),
            "source": "hf_acl_all",
        })

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, default=str)
    logger.info("Saved %d papers to cache", len(papers))

    return papers


# ═══════════════════════════════════════════════════════════════
# STEP 2: Recent ACL papers (2023+) via acl-anthology package
# ═══════════════════════════════════════════════════════════════

def collect_recent_all(existing_titles: set[str]) -> list[dict]:
    """Get ALL 2023+ papers from ACL Anthology package."""
    cache_file = DATA_DIR / "acl_all_step2_recent.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            papers = json.load(f)
        logger.info("Loaded %d papers from cache (step2)", len(papers))
        return papers

    import warnings
    warnings.filterwarnings("ignore")
    from acl_anthology import Anthology

    logger.info("Loading ACL Anthology from repo...")
    anthology = Anthology.from_repo()

    target = re.compile(r"(20[2-9]\d)\.(acl|emnlp|naacl|eacl|coling|findings|tacl)", re.IGNORECASE)

    papers = []
    collection_ids = [cid for cid in anthology.collections if target.match(str(cid)) and int(str(cid)[:4]) >= 2023]
    logger.info("Scanning %d recent collections...", len(collection_ids))

    for cid in tqdm(collection_ids, desc="Recent ACL"):
        collection = anthology.collections[cid]
        for volume in collection.volumes():
            for paper in volume.papers():
                title = str(paper.title) if paper.title else ""
                if not title:
                    continue

                nt = norm_title(title)
                if nt in existing_titles:
                    continue

                abstract = str(paper.abstract) if hasattr(paper, "abstract") and paper.abstract else ""
                doi_val = str(paper.doi) if hasattr(paper, "doi") and paper.doi else ""

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "year": int(str(cid)[:4]),
                    "venue": str(cid),
                    "acl_id": str(paper.full_id),
                    "doi": doi_val,
                    "citation_count": 0,
                    "source": "acl_recent_all",
                })

                existing_titles.add(nt)  # prevent self-dupes

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, default=str)
    logger.info("Found %d new recent papers", len(papers))

    return papers


# ═══════════════════════════════════════════════════════════════
# STEP 3: Semantic Scholar enrichment
# ═══════════════════════════════════════════════════════════════

def enrich_citations(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich with SS citations. For HF papers, keep their numcitedby as fallback."""
    logger.info("Enriching %d papers with SS...", len(df))

    # Build SS IDs — prefer CorpusId (most reliable), then DOI, then ACL
    ss_ids = []
    for _, row in df.iterrows():
        corpus = str(row.get("corpus_paper_id", "") or "").strip()
        doi = str(row.get("doi", "") or "").strip()
        acl = str(row.get("acl_id", "") or "").strip()

        if corpus and corpus not in ("0", "", "nan", "None"):
            ss_ids.append(f"CorpusId:{corpus}")
        elif doi and doi.startswith("10."):
            ss_ids.append(f"DOI:{doi}")
        elif acl:
            ss_ids.append(f"ACL:{acl}")
        else:
            ss_ids.append(None)

    # Batch lookup
    results = {}
    valid = [(i, sid) for i, sid in enumerate(ss_ids) if sid]
    batch_size = 450

    for batch_start in tqdm(range(0, len(valid), batch_size), desc="SS enrich"):
        batch = valid[batch_start:batch_start + batch_size]
        batch_ids = [sid for _, sid in batch]

        for attempt in range(3):
            try:
                resp = requests.post(
                    SS_BATCH_URL,
                    params={"fields": SS_FIELDS},
                    json={"ids": batch_ids},
                    timeout=60,
                )
                if resp.status_code == 200:
                    for (idx, _), result in zip(batch, resp.json()):
                        results[idx] = result
                    break
                elif resp.status_code == 429:
                    wait = 30 * (attempt + 1) + random.randint(5, 15)
                    logger.warning("SS 429, waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    logger.warning("SS HTTP %d", resp.status_code)
                    time.sleep(10)
            except Exception as e:
                logger.warning("SS error: %s", e)
                time.sleep(15)

        time.sleep(1.5)

    # Apply
    citation_counts = []
    influential_counts = []
    statuses = []
    ss_paper_ids = []
    arxiv_ids = []
    oa_urls = []

    for i in range(len(df)):
        row = df.iloc[i]
        result = results.get(i)
        orig_cc = row.get("citation_count", 0)
        if pd.isna(orig_cc):
            orig_cc = 0
        orig_cc = int(orig_cc)

        if result is not None:
            cc = result.get("citationCount", 0) or 0
            citation_counts.append(cc)
            influential_counts.append(result.get("influentialCitationCount", 0) or 0)
            statuses.append("verified")
            ss_paper_ids.append(result.get("paperId", "") or "")
            ext = result.get("externalIds", {}) or {}
            arxiv_ids.append(ext.get("ArXiv", "") or "")
            oa = result.get("openAccessPdf", {}) or {}
            oa_urls.append(oa.get("url", "") or "")
        elif orig_cc > 0:
            # HF had citation data, use it
            citation_counts.append(orig_cc)
            influential_counts.append(0)
            statuses.append("hf_verified")
            ss_paper_ids.append("")
            arxiv_ids.append("")
            oa_urls.append("")
        else:
            # No data
            citation_counts.append(0)
            influential_counts.append(0)
            year = row.get("year")
            try:
                y = int(year)
            except:
                y = 0
            if y >= 2024:
                statuses.append("too_recent")
            elif y >= 2023:
                statuses.append("recent_unverified")
            else:
                statuses.append("ss_not_found")
            ss_paper_ids.append("")
            arxiv_ids.append("")
            oa_urls.append("")

    df = df.copy()
    df["citation_count"] = citation_counts
    df["influential_citation_count"] = influential_counts
    df["citation_status"] = statuses
    df["ss_paper_id"] = ss_paper_ids
    df["arxiv_id"] = arxiv_ids
    df["open_access_url"] = oa_urls

    for status in df["citation_status"].unique():
        cnt = (df["citation_status"] == status).sum()
        logger.info("  %s: %d", status, cnt)

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4: Full text
# ═══════════════════════════════════════════════════════════════

def save_hf_fulltext(papers: list[dict]):
    """Save full text from HF corpus papers."""
    FT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for p in papers:
        ft = p.get("full_text", "")
        acl_id = p.get("acl_id", "")
        if ft and acl_id and len(str(ft)) > 300:
            safe_id = acl_id.replace("/", "_").replace(".", "_")
            out_file = FT_DIR / f"{safe_id}.txt"
            if not out_file.exists():
                out_file.write_text(str(ft), encoding="utf-8")
                saved += 1
    logger.info("Saved %d HF full texts", saved)
    return saved


def download_acl_pdfs(papers: list[dict]):
    """Download PDFs for papers without full text."""
    import fitz

    FT_DIR.mkdir(parents=True, exist_ok=True)

    need = []
    for p in papers:
        acl_id = p.get("acl_id", "")
        if not acl_id:
            continue
        safe_id = acl_id.replace("/", "_").replace(".", "_")
        if not (FT_DIR / f"{safe_id}.txt").exists():
            need.append(p)

    logger.info("Papers needing PDF download: %d", len(need))
    if not need:
        return

    saved = 0
    failed = 0
    for p in tqdm(need, desc="ACL PDFs"):
        acl_id = p["acl_id"]
        safe_id = acl_id.replace("/", "_").replace(".", "_")
        out_file = FT_DIR / f"{safe_id}.txt"

        url = f"https://aclanthology.org/{acl_id}.pdf"
        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "novelty-search/0.1 (academic research; polite bot)"
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                doc = fitz.open(stream=resp.content, filetype="pdf")
                parts = [page.get_text() for page in doc]
                doc.close()
                text = "\n".join(parts)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = text.strip()
                if len(text) > 300:
                    out_file.write_text(text, encoding="utf-8")
                    saved += 1
                else:
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        time.sleep(0.3)

    logger.info("PDF download: %d saved, %d failed", saved, failed)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FT_DIR.mkdir(parents=True, exist_ok=True)

    existing_titles = load_existing_titles()
    logger.info("Existing titles to dedup against: %d", len(existing_titles))

    # Step 1: HF ACL Anthology (pre-2022)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: HuggingFace ACL Anthology (all NLP papers)")
    logger.info("=" * 60)
    hf_papers = collect_hf_all(existing_titles)
    # Add HF titles to existing for dedup in step 2
    for p in hf_papers:
        existing_titles.add(norm_title(p.get("title", "")))
    logger.info("HF papers: %d", len(hf_papers))

    # Save HF full texts immediately
    logger.info("Saving HF full texts...")
    save_hf_fulltext(hf_papers)

    # Step 2: Recent ACL (2023+)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Recent ACL Anthology (2023+, all topics)")
    logger.info("=" * 60)
    recent_papers = collect_recent_all(existing_titles)
    logger.info("Recent papers: %d", len(recent_papers))

    # Combine
    all_papers = hf_papers + recent_papers
    logger.info("Total new papers: %d", len(all_papers))

    # Self-deduplicate
    df = pd.DataFrame(all_papers)
    df["_norm"] = df["title"].apply(norm_title)
    before = len(df)
    df = df.drop_duplicates(subset=["_norm"], keep="first")
    df = df.drop(columns=["_norm"])
    logger.info("After self-dedup: %d (removed %d)", len(df), before - len(df))

    # Step 3: SS enrichment
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Semantic Scholar enrichment")
    logger.info("=" * 60)
    df = enrich_citations(df)

    # Save parquet
    keep_cols = [
        "title", "abstract", "year", "venue", "citation_count",
        "influential_citation_count", "citation_status",
        "ss_paper_id", "arxiv_id", "doi", "acl_id",
        "open_access_url", "source",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[keep_cols].copy()

    df.to_parquet(OUT_PARQUET, index=False)
    logger.info("Saved %d papers to %s", len(df), OUT_PARQUET)

    # Stats
    print("\n" + "=" * 60)
    print("ACL All-NLP Collection Stats")
    print("=" * 60)
    print(f"Total new papers: {len(df):,}")
    print(f"\nBy source:")
    for src, cnt in df["source"].value_counts().items():
        print(f"  {src}: {cnt:,}")
    print(f"\nCitation status:")
    for status, cnt in df["citation_status"].value_counts().items():
        sub = df[df["citation_status"] == status]
        cc = pd.to_numeric(sub["citation_count"], errors="coerce").fillna(0)
        print(f"  {status}: {cnt:,} (mean cites: {cc.mean():.1f})")
    print(f"\nBy year (grouped):")
    years = pd.to_numeric(df["year"], errors="coerce")
    for label, lo, hi in [("<2000", 0, 2000), ("2000-09", 2000, 2010), ("2010-15", 2010, 2016),
                           ("2016-19", 2016, 2020), ("2020-22", 2020, 2023), ("2023+", 2023, 2100)]:
        cnt = ((years >= lo) & (years < hi)).sum()
        print(f"  {label}: {cnt:,}")
    print(f"\nTop 15 venues:")
    for v, cnt in df["venue"].value_counts().head(15).items():
        print(f"  {cnt:>5,} | {str(v)[:80]}")

    # Step 4: Download PDFs for recent papers
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Full text (PDF downloads for 2023+ papers)")
    logger.info("=" * 60)
    download_acl_pdfs(recent_papers)

    ft_count = len(list(FT_DIR.glob("*.txt")))
    print(f"\nFull text files: {ft_count:,}")
    print("DONE!")


if __name__ == "__main__":
    main()
