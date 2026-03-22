"""Fill gaps in embeddings_acl dataset: fulltext + citations + citation_status.

Adds `citation_status` column:
  - "verified": SS returned this paper (citationCount may be 0 or more)
  - "ss_not_found": SS could not find this paper (citation_count is unreliable)
"""

from __future__ import annotations

import json
import re
import time
import random
import logging
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
PARQUET = DATA_DIR / "embeddings_acl.parquet"
FT_DIR = DATA_DIR / "embeddings_acl_fulltext"
MAIN_FT_DIR = DATA_DIR / "fulltext"

SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS = "title,abstract,year,venue,citationCount,influentialCitationCount,externalIds,openAccessPdf"


def load_fulltext_index() -> set[str]:
    """Get set of all paper IDs that have fulltext."""
    stems = set()
    for d in [FT_DIR, MAIN_FT_DIR]:
        if d.exists():
            for f in d.glob("*.txt"):
                stems.add(f.stem)
    return stems


def paper_has_fulltext(row, ft_index: set[str]) -> bool:
    for field in ["acl_id", "arxiv_id"]:
        val = str(row.get(field, "") or "")
        if val:
            safe = val.replace("/", "_").replace(".", "_")
            if safe in ft_index:
                return True
    return False


# ═══════════════════════════════════════════════════════════════
# STEP 1: Re-enrich ALL papers with Semantic Scholar
# ═══════════════════════════════════════════════════════════════

def build_ss_ids(df: pd.DataFrame) -> list[str | None]:
    """Build SS lookup IDs for each paper, trying multiple strategies."""
    ids = []
    for _, row in df.iterrows():
        # Priority: DOI > ArXiv > ACL > CorpusId > title search
        doi = str(row.get("doi", "") or "").strip()
        arxiv = str(row.get("arxiv_id", "") or "").strip()
        acl = str(row.get("acl_id", "") or "").strip()
        ssid = str(row.get("ss_paper_id", "") or "").strip()
        corpus = str(row.get("corpus_paper_id", "") or "").strip()

        if doi:
            ids.append(f"DOI:{doi}")
        elif arxiv:
            ids.append(f"ArXiv:{arxiv}")
        elif acl:
            ids.append(f"ACL:{acl}")
        elif ssid:
            ids.append(ssid)
        elif corpus and corpus != "0" and corpus != "":
            ids.append(f"CorpusId:{corpus}")
        else:
            ids.append(None)
    return ids


def batch_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Re-enrich all papers with SS, tracking which ones SS found."""
    ss_ids = build_ss_ids(df)

    # Track results
    results = {}  # index -> SS result or None
    batch_size = 450

    valid = [(i, sid) for i, sid in enumerate(ss_ids) if sid]
    logger.info("Enriching %d papers (out of %d total)", len(valid), len(df))

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
                    for (idx, sid), result in zip(batch, resp.json()):
                        results[idx] = result  # None if not found
                    break
                elif resp.status_code == 429:
                    wait = 30 * (attempt + 1) + random.randint(5, 15)
                    logger.warning("SS 429, waiting %ds (attempt %d/3)", wait, attempt + 1)
                    time.sleep(wait)
                else:
                    logger.warning("SS batch HTTP %d, retrying...", resp.status_code)
                    time.sleep(10)
            except Exception as e:
                logger.warning("SS batch error: %s", e)
                time.sleep(15)

        time.sleep(1.5)

    # Apply results
    citation_counts = []
    influential_counts = []
    citation_statuses = []
    ss_paper_ids = []
    arxiv_ids_new = []
    open_access_urls = []

    for i in range(len(df)):
        row = df.iloc[i]
        result = results.get(i)

        if result is not None:
            # SS found this paper
            citation_statuses.append("verified")
            citation_counts.append(result.get("citationCount", 0) or 0)
            influential_counts.append(result.get("influentialCitationCount", 0) or 0)
            ss_paper_ids.append(result.get("paperId", "") or "")

            ext = result.get("externalIds", {}) or {}
            arxiv_ids_new.append(ext.get("ArXiv", "") or str(row.get("arxiv_id", "") or ""))

            oa = result.get("openAccessPdf", {}) or {}
            open_access_urls.append(oa.get("url", "") or "")
        elif i in [idx for idx, sid in enumerate(ss_ids) if sid is None]:
            # We had no ID to look up
            citation_statuses.append("no_id")
            citation_counts.append(0)
            influential_counts.append(0)
            ss_paper_ids.append(str(row.get("ss_paper_id", "") or ""))
            arxiv_ids_new.append(str(row.get("arxiv_id", "") or ""))
            open_access_urls.append("")
        else:
            # We had an ID but SS didn't find it
            citation_statuses.append("ss_not_found")
            citation_counts.append(0)
            influential_counts.append(0)
            ss_paper_ids.append(str(row.get("ss_paper_id", "") or ""))
            arxiv_ids_new.append(str(row.get("arxiv_id", "") or ""))
            open_access_urls.append("")

    df = df.copy()
    df["citation_count"] = citation_counts
    df["influential_citation_count"] = influential_counts
    df["citation_status"] = citation_statuses
    df["ss_paper_id"] = ss_paper_ids
    df["arxiv_id"] = arxiv_ids_new
    df["open_access_url"] = open_access_urls

    # Stats
    status_counts = df["citation_status"].value_counts()
    logger.info("Citation status: %s", dict(status_counts))
    verified = df[df["citation_status"] == "verified"]
    logger.info("Verified papers: %d, with citations>0: %d, verified-zero: %d",
                len(verified),
                (verified["citation_count"] > 0).sum(),
                (verified["citation_count"] == 0).sum())

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: Fill fulltext gaps
# ═══════════════════════════════════════════════════════════════

def fill_fulltext_gaps(df: pd.DataFrame):
    """Try to get fulltext for papers that are missing it."""
    import fitz

    ft_index = load_fulltext_index()
    FT_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for i, row in df.iterrows():
        if not paper_has_fulltext(row, ft_index):
            missing.append((i, row))

    logger.info("Papers missing fulltext: %d", len(missing))

    saved = 0
    failed = 0

    for i, row in tqdm(missing, desc="Fill fulltext"):
        acl_id = str(row.get("acl_id", "") or "")
        arxiv_id = str(row.get("arxiv_id", "") or "")
        oa_url = str(row.get("open_access_url", "") or "")

        # Determine file ID
        if acl_id:
            safe_id = acl_id.replace("/", "_").replace(".", "_")
        elif arxiv_id:
            safe_id = arxiv_id.replace("/", "_").replace(".", "_")
        else:
            safe_id = str(row.get("ss_paper_id", "") or "").replace("/", "_")

        if not safe_id:
            failed += 1
            continue

        out_file = FT_DIR / f"{safe_id}.txt"
        if out_file.exists():
            saved += 1
            continue

        # Try multiple PDF sources
        urls_to_try = []

        # 1. ACL Anthology PDF
        if acl_id:
            urls_to_try.append(f"https://aclanthology.org/{acl_id}.pdf")

        # 2. Open access PDF from SS
        if oa_url:
            urls_to_try.append(oa_url)

        # 3. arXiv PDF
        if arxiv_id:
            urls_to_try.append(f"https://arxiv.org/pdf/{arxiv_id}")
            urls_to_try.append(f"https://arxiv.org/pdf/{arxiv_id}v1")

        text = None
        for url in urls_to_try:
            try:
                resp = requests.get(url, timeout=30, headers={
                    "User-Agent": "novelty-search/0.1 (academic research)"
                })
                if resp.status_code == 200 and len(resp.content) > 1000:
                    ct = resp.headers.get("Content-Type", "")
                    if "pdf" in ct.lower() or resp.content[:5] == b"%PDF-":
                        doc = fitz.open(stream=resp.content, filetype="pdf")
                        parts = [page.get_text() for page in doc]
                        doc.close()
                        text = "\n".join(parts)
                        text = re.sub(r"\n{3,}", "\n\n", text)
                        text = re.sub(r"[ \t]+", " ", text)
                        text = text.strip()
                        if len(text) > 300:
                            break
                        text = None
            except Exception:
                pass
            time.sleep(0.2)

        if text and len(text) > 300:
            out_file.write_text(text, encoding="utf-8")
            saved += 1
        else:
            failed += 1

        time.sleep(0.3)

    logger.info("Fulltext gap-fill: %d saved, %d failed", saved, failed)
    return saved, failed


# ═══════════════════════════════════════════════════════════════
# STEP 3: Try title-based SS lookup for remaining unknowns
# ═══════════════════════════════════════════════════════════════

def title_search_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """For papers where SS batch lookup failed, try title search."""
    SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    not_found = df[df["citation_status"].isin(["ss_not_found", "no_id"])].copy()
    logger.info("Trying title search for %d papers...", len(not_found))

    updates = {}

    for idx, row in tqdm(not_found.iterrows(), total=len(not_found), desc="Title search"):
        title = str(row.get("title", ""))
        if not title or len(title) < 10:
            continue

        # Clean title for search
        clean = re.sub(r"[{}\\]", "", title).strip()
        if len(clean) < 10:
            continue

        for attempt in range(2):
            try:
                resp = requests.get(
                    SS_SEARCH_URL,
                    params={
                        "query": clean[:200],
                        "fields": "title,citationCount,influentialCitationCount,externalIds",
                        "limit": 3,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    # Find exact or near-exact title match
                    norm = re.sub(r"[^a-z0-9]", "", clean.lower())
                    for p in data:
                        p_norm = re.sub(r"[^a-z0-9]", "", (p.get("title", "") or "").lower())
                        if norm == p_norm or (len(norm) > 20 and norm in p_norm):
                            ext = p.get("externalIds", {}) or {}
                            updates[idx] = {
                                "citation_count": p.get("citationCount", 0) or 0,
                                "influential_citation_count": p.get("influentialCitationCount", 0) or 0,
                                "citation_status": "verified",
                                "ss_paper_id": p.get("paperId", ""),
                                "arxiv_id": ext.get("ArXiv", "") or str(row.get("arxiv_id", "") or ""),
                            }
                            break
                    break
                elif resp.status_code == 429:
                    time.sleep(30 + random.randint(5, 15))
                else:
                    break
            except Exception:
                time.sleep(10)

        time.sleep(1.5 + random.random())

    logger.info("Title search found %d additional matches", len(updates))

    for idx, upd in updates.items():
        for col, val in upd.items():
            df.at[idx, col] = val

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    df = pd.read_parquet(PARQUET)
    logger.info("Loaded %d papers from %s", len(df), PARQUET)

    # Step 1: Re-enrich all papers with SS
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Full SS re-enrichment")
    logger.info("=" * 60)
    df = batch_enrich(df)

    # Step 2: Title search for remaining unknowns
    not_found_count = (df["citation_status"].isin(["ss_not_found", "no_id"])).sum()
    if not_found_count > 0 and not_found_count < 500:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Title-based search for %d unmatched papers", not_found_count)
        logger.info("=" * 60)
        df = title_search_remaining(df)
    elif not_found_count >= 500:
        logger.info("Skipping title search — too many unmatched (%d), would hit rate limits", not_found_count)

    # Save updated parquet
    df.to_parquet(PARQUET, index=False)
    logger.info("Saved updated parquet to %s", PARQUET)

    # Step 3: Fill fulltext gaps
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fill fulltext gaps")
    logger.info("=" * 60)
    saved, failed = fill_fulltext_gaps(df)

    # Final stats
    ft_index = load_fulltext_index()
    df["has_ft"] = df.apply(lambda r: paper_has_fulltext(r, ft_index), axis=1)

    print("\n" + "=" * 60)
    print("Gap-filling Results")
    print("=" * 60)
    print(f"Total papers: {len(df):,}")
    print(f"\nCitation status:")
    for status, cnt in df["citation_status"].value_counts().items():
        sub = df[df["citation_status"] == status]
        cc = pd.to_numeric(sub["citation_count"], errors="coerce").fillna(0)
        print(f"  {status}: {cnt:,} papers (mean cites: {cc.mean():.1f}, zero: {(cc==0).sum():,})")
    print(f"\nFull text: {df['has_ft'].sum():,} / {len(df):,} ({df['has_ft'].mean()*100:.1f}%)")
    print(f"Full text files in dir: {len(list(FT_DIR.glob('*.txt'))):,}")
    print(f"\nDONE!")


if __name__ == "__main__":
    main()
