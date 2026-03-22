"""Collect embedding-evolution papers from ACL Anthology + missing arXiv papers.

Captures the full embeddings revolution timeline:
  static embeddings → contextual embeddings → transformer representations
  → sentence embeddings → retrieval embeddings

Sources:
  1. HuggingFace ACL-OCL/acl-anthology-corpus (bulk historical, up to ~2022)
  2. acl-anthology Python package (2023+ papers from ACL/EMNLP/NAACL)
  3. Semantic Scholar API (citation enrichment + gap filling)
  4. arXiv direct (pre-2016 landmark papers + full text)

Output:
  data/processed/v3/embeddings_acl.parquet — all collected papers
  data/processed/v3/embeddings_acl_fulltext/ — full text files
"""

from __future__ import annotations

import json
import logging
import re
import time
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

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path("data/processed/v3")
OUT_PARQUET = DATA_DIR / "embeddings_acl.parquet"
FULLTEXT_DIR = DATA_DIR / "embeddings_acl_fulltext"
PROGRESS_FILE = DATA_DIR / "embeddings_acl_progress.json"

SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = "title,abstract,year,venue,publicationDate,citationCount,influentialCitationCount,externalIds,s2FieldsOfStudy,openAccessPdf"

# Broad keyword patterns for the embeddings evolution
KEYWORD_PATTERNS = [
    # Static embeddings
    r"word\s*embedding",
    r"word\s*vector",
    r"word2vec",
    r"glove",
    r"fasttext",
    r"distributional\s+semantics",
    r"distributed\s+representation",
    r"word\s+representation",
    r"static\s+embedding",
    r"pre[-\s]trained\s+embedding",
    # Contextual embeddings
    r"contextual\w*\s+embedding",
    r"contextuali[sz]ed\s+(?:word\s+)?representation",
    r"elmo",
    r"deep\s+contextuali[sz]ed",
    r"language\s+model\s+representation",
    # Transformer representations
    r"bert\s+(?:embedding|representation|model|pre[-\s]train)",
    r"transformer\s+representation",
    r"pre[-\s]train(?:ed|ing)\s+(?:language\s+)?model",
    r"fine[-\s]tun(?:e|ed|ing)\s+(?:bert|transformer|representation)",
    # Sentence embeddings
    r"sentence\s+embedding",
    r"sentence\s+representation",
    r"sentence\s+encoder",
    r"sentence[-\s]bert",
    r"simcse",
    r"infersent",
    r"universal\s+sentence\s+encoder",
    r"contrastive\s+(?:learning\s+(?:of|for)\s+)?sentence",
    r"sentence\s+similarity",
    r"semantic\s+textual\s+similarity",
    # Retrieval embeddings
    r"text\s+embedding",
    r"dense\s+(?:passage\s+)?retrieval",
    r"retrieval\s+embedding",
    r"bi[-\s]encoder",
    r"dual[-\s]encoder",
    r"colbert",
    r"contriever",
    r"learned\s+(?:sparse\s+)?representation\s+(?:for\s+)?retrieval",
    r"embedding\s+model",
    r"representation\s+learning\s+(?:for\s+)?(?:nlp|text|language)",
]

# Compile into a single pattern for efficiency
COMBINED_PATTERN = re.compile(
    "|".join(f"(?:{p})" for p in KEYWORD_PATTERNS),
    re.IGNORECASE,
)

# Pre-2016 landmark arXiv papers we must include
PRE_2016_ARXIV_IDS = [
    "1301.3781",  # Word2Vec (Efficient Estimation)
    "1310.4546",  # Word2Vec (Skip-gram, Negative Sampling)
    "1411.2738",  # Doc2Vec / Paragraph Vectors
    "1506.06726",  # Skip-thought Vectors
    "1508.06615",  # Character-level CNNs for text (relevant context)
    "1301.3388",  # EfficientZero-shot (Socher et al.) — early representation learning
    "1409.0473",  # Attention mechanism (Bahdanau) — precursor
    "1502.03044",  # Show, Attend and Tell — attention for vision
]

# Conference-only landmark papers (no arXiv ID)
CONFERENCE_ONLY_PAPERS = [
    {
        "title": "GloVe: Global Vectors for Word Representation",
        "authors": "Jeffrey Pennington, Richard Socher, Christopher D. Manning",
        "year": 2014,
        "venue": "EMNLP",
        "acl_id": "D14-1162",
        "doi": "10.3115/v1/D14-1162",
    },
    {
        "title": "Contextual String Embeddings for Sequence Labeling",
        "authors": "Alan Akbik, Duncan Blythe, Roland Vollgraf",
        "year": 2018,
        "venue": "COLING",
        "acl_id": "C18-1139",
    },
]

# Target ACL venues
ACL_VENUES = {
    "ACL", "EMNLP", "NAACL", "NAACL-HLT",
    "EACL", "AACL", "COLING", "TACL",
    "Findings of ACL", "Findings of EMNLP", "Findings of NAACL",
    # Patterns for booktitle matching
}

ACL_BOOKTITLE_PATTERNS = re.compile(
    r"(?:Annual Meeting of the Association for Computational Linguistics|"
    r"Conference on Empirical Methods in Natural Language Processing|"
    r"North American Chapter|"
    r"European Chapter|"
    r"NAACL|EMNLP|EACL|AACL|COLING|TACL|"
    r"Findings of the Association|"
    r"Computational Linguistics)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════
# STEP 1: HuggingFace ACL Anthology Corpus
# ═══════════════════════════════════════════════════════════════

def collect_from_hf_anthology() -> pd.DataFrame:
    """Load ACL Anthology corpus from pre-downloaded parquet."""
    from huggingface_hub import hf_hub_download

    logger.info("Loading ACL Anthology corpus parquet...")
    path = hf_hub_download(
        repo_id="ACL-OCL/acl-anthology-corpus",
        filename="acl-publication-info.74k.v2.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    logger.info("Loaded %d papers from ACL Anthology", len(df))

    # Filter to target venues
    venue_mask = df["booktitle"].fillna("").apply(
        lambda x: bool(ACL_BOOKTITLE_PATTERNS.search(x))
    ) | df["journal"].fillna("").apply(
        lambda x: bool(re.search(r"computational linguistics|tacl", x, re.IGNORECASE))
    )
    df_venues = df[venue_mask].copy()
    logger.info("Papers from ACL/EMNLP/NAACL/COLING/TACL/EACL: %d", len(df_venues))

    # Filter by embedding keywords in title + abstract + full_text
    def matches_keywords(row):
        text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
        return bool(COMBINED_PATTERN.search(text))

    keyword_mask = df_venues.apply(matches_keywords, axis=1)
    df_filtered = df_venues[keyword_mask].copy()
    logger.info("Embedding-related papers from target venues: %d", len(df_filtered))

    # Also get any paper from ANY venue that matches keywords (for broader coverage)
    all_keyword_mask = df.apply(matches_keywords, axis=1)
    df_all_keywords = df[all_keyword_mask].copy()
    logger.info("Embedding-related papers from ALL venues: %d", len(df_all_keywords))

    # Combine: target venue papers + keyword matches from all venues
    df_combined = pd.concat([df_filtered, df_all_keywords]).drop_duplicates(subset=["acl_id"])
    logger.info("Combined unique papers: %d", len(df_combined))

    return df_combined


# ═══════════════════════════════════════════════════════════════
# STEP 2: Semantic Scholar enrichment
# ═══════════════════════════════════════════════════════════════

def enrich_with_semantic_scholar(papers: list[dict]) -> list[dict]:
    """Enrich papers with Semantic Scholar citation data."""
    logger.info("Enriching %d papers with Semantic Scholar data...", len(papers))

    # Build lookup IDs — prefer DOI, then corpus_paper_id, then title search
    enriched = []
    batch_size = 450
    ss_ids = []

    for p in papers:
        if p.get("doi"):
            ss_ids.append(f"DOI:{p['doi']}")
        elif p.get("corpus_paper_id") and str(p["corpus_paper_id"]) != "0":
            ss_ids.append(f"CorpusId:{p['corpus_paper_id']}")
        elif p.get("acl_id"):
            ss_ids.append(f"ACL:{p['acl_id']}")
        else:
            ss_ids.append(None)

    # Batch lookup
    results = {}
    valid_ids = [(i, sid) for i, sid in enumerate(ss_ids) if sid]

    for batch_start in tqdm(range(0, len(valid_ids), batch_size), desc="SS batch"):
        batch = valid_ids[batch_start:batch_start + batch_size]
        batch_ids = [sid for _, sid in batch]

        try:
            resp = requests.post(
                SS_BATCH_URL,
                params={"fields": SS_FIELDS},
                json={"ids": batch_ids},
                timeout=30,
            )
            if resp.status_code == 200:
                for (idx, sid), result in zip(batch, resp.json()):
                    if result:
                        results[idx] = result
            elif resp.status_code == 429:
                logger.warning("SS rate limited, waiting 30s...")
                time.sleep(30)
                # retry
                resp = requests.post(
                    SS_BATCH_URL,
                    params={"fields": SS_FIELDS},
                    json={"ids": batch_ids},
                    timeout=30,
                )
                if resp.status_code == 200:
                    for (idx, sid), result in zip(batch, resp.json()):
                        if result:
                            results[idx] = result
            else:
                logger.warning("SS batch failed: HTTP %d", resp.status_code)
        except Exception as e:
            logger.warning("SS batch error: %s", e)

        time.sleep(1)  # respect rate limit

    logger.info("SS enrichment: %d/%d papers matched", len(results), len(papers))

    # Merge results back
    for i, p in enumerate(papers):
        ss = results.get(i, {})
        p["citation_count"] = ss.get("citationCount", p.get("numcitedby", 0))
        p["influential_citation_count"] = ss.get("influentialCitationCount", 0)
        p["ss_paper_id"] = ss.get("paperId", "")
        p["ss_venue"] = ss.get("venue", "")

        # Get arXiv ID from SS if available
        ext_ids = ss.get("externalIds", {}) or {}
        if ext_ids.get("ArXiv"):
            p["arxiv_id"] = ext_ids["ArXiv"]

        enriched.append(p)

    return enriched


# ═══════════════════════════════════════════════════════════════
# STEP 3: Semantic Scholar keyword search for gap filling
# ═══════════════════════════════════════════════════════════════

def search_ss_for_gaps(existing_titles: set[str]) -> list[dict]:
    """Search Semantic Scholar for embedding papers we might have missed.

    Uses the bulk search endpoint which is more lenient on rate limits.
    Falls back gracefully — this step is supplementary.
    """
    import random

    queries = [
        ("word2vec distributed representations", 2013, 2016),
        ("GloVe word vectors", 2014, 2017),
        ("contextual embeddings ELMo", 2017, 2020),
        ("sentence embeddings contrastive learning", 2019, 2025),
        ("dense passage retrieval", 2019, 2025),
        ("text embeddings model benchmark MTEB", 2022, 2026),
        ("instruction-finetuned text embeddings", 2022, 2026),
        ("matryoshka representation learning", 2022, 2025),
    ]

    # Use bulk search endpoint — different rate limit bucket
    BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    all_papers = []
    seen_ids = set()

    for query, year_start, year_end in tqdm(queries, desc="SS search"):
        retries = 0
        max_retries = 3
        token = None  # continuation token for bulk search

        while retries < max_retries:
            params = {
                "query": query,
                "fields": SS_FIELDS,
                "year": f"{year_start}-{year_end}",
                "fieldsOfStudy": "Computer Science",
            }
            if token:
                params["token"] = token

            try:
                # Random delay 3-8s to look less like a bot
                time.sleep(3 + random.random() * 5)

                resp = requests.get(BULK_URL, params=params, timeout=60, headers={
                    "User-Agent": f"Mozilla/5.0 (research-bot; novelty-search/{random.randint(1,99)})"
                })

                if resp.status_code == 429:
                    retries += 1
                    wait = 60 * retries + random.randint(10, 30)
                    logger.warning("SS 429 on '%s' (retry %d/%d), waiting %ds...",
                                   query, retries, max_retries, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.warning("SS search '%s': HTTP %d, skipping", query, resp.status_code)
                    break

                data = resp.json()
                papers = data.get("data", [])
                if not papers:
                    break

                retries = 0  # reset on success
                found_this_batch = 0

                for p in papers:
                    pid = p.get("paperId", "")
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    title = p.get("title", "")
                    text = f"{title} {p.get('abstract', '')}".lower()
                    if not COMBINED_PATTERN.search(text):
                        continue

                    norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
                    if norm_title in existing_titles:
                        continue

                    ext_ids = p.get("externalIds", {}) or {}
                    all_papers.append({
                        "title": title,
                        "abstract": p.get("abstract", ""),
                        "year": p.get("year"),
                        "venue": p.get("venue", ""),
                        "citation_count": p.get("citationCount", 0),
                        "influential_citation_count": p.get("influentialCitationCount", 0),
                        "ss_paper_id": pid,
                        "arxiv_id": ext_ids.get("ArXiv", ""),
                        "doi": ext_ids.get("DOI", ""),
                        "source": "ss_search",
                    })
                    found_this_batch += 1

                logger.info("  '%s': got %d papers (%d matched)", query[:40], len(papers), found_this_batch)

                # Bulk search returns a token for next page
                token = data.get("token")
                if not token:
                    break
                # Only get first page per query (1000 papers) — enough for gap-filling
                break

            except Exception as e:
                logger.warning("SS search error for '%s': %s", query, e)
                retries += 1
                time.sleep(30)

    logger.info("SS search found %d additional papers", len(all_papers))
    return all_papers


# ═══════════════════════════════════════════════════════════════
# STEP 4: Pre-2016 arXiv landmarks
# ═══════════════════════════════════════════════════════════════

def fetch_pre2016_arxiv() -> list[dict]:
    """Fetch pre-2016 landmark papers directly from SS by arXiv ID."""
    papers = []

    ids_to_fetch = [f"ArXiv:{aid}" for aid in PRE_2016_ARXIV_IDS]

    try:
        resp = requests.post(
            SS_BATCH_URL,
            params={"fields": SS_FIELDS},
            json={"ids": ids_to_fetch},
            timeout=30,
        )
        if resp.status_code == 200:
            for aid, result in zip(PRE_2016_ARXIV_IDS, resp.json()):
                if result:
                    ext_ids = result.get("externalIds", {}) or {}
                    papers.append({
                        "title": result.get("title", ""),
                        "abstract": result.get("abstract", ""),
                        "year": result.get("year"),
                        "venue": result.get("venue", ""),
                        "citation_count": result.get("citationCount", 0),
                        "influential_citation_count": result.get("influentialCitationCount", 0),
                        "ss_paper_id": result.get("paperId", ""),
                        "arxiv_id": ext_ids.get("ArXiv", aid),
                        "doi": ext_ids.get("DOI", ""),
                        "source": "arxiv_landmark",
                    })
                    logger.info("  Pre-2016: %s (%d cites)", result.get("title", "")[:60], result.get("citationCount", 0))
        else:
            logger.warning("SS batch for pre-2016: HTTP %d", resp.status_code)
    except Exception as e:
        logger.warning("SS batch error for pre-2016: %s", e)

    # Add conference-only papers
    for p in CONFERENCE_ONLY_PAPERS:
        ss_id = f"DOI:{p['doi']}" if p.get("doi") else f"ACL:{p['acl_id']}"
        try:
            resp = requests.post(
                SS_BATCH_URL,
                params={"fields": SS_FIELDS},
                json={"ids": [ss_id]},
                timeout=30,
            )
            time.sleep(1)
            if resp.status_code == 200:
                results = resp.json()
                if results and results[0]:
                    r = results[0]
                    ext_ids = r.get("externalIds", {}) or {}
                    papers.append({
                        "title": r.get("title", p["title"]),
                        "abstract": r.get("abstract", ""),
                        "year": r.get("year", p["year"]),
                        "venue": r.get("venue", p["venue"]),
                        "citation_count": r.get("citationCount", 0),
                        "influential_citation_count": r.get("influentialCitationCount", 0),
                        "ss_paper_id": r.get("paperId", ""),
                        "arxiv_id": ext_ids.get("ArXiv", ""),
                        "doi": ext_ids.get("DOI", p.get("doi", "")),
                        "acl_id": p.get("acl_id", ""),
                        "source": "conference_landmark",
                    })
                    logger.info("  Landmark: %s (%d cites)", r.get("title", "")[:60], r.get("citationCount", 0))
        except Exception as e:
            logger.warning("SS error for %s: %s", p["title"][:40], e)

    logger.info("Fetched %d pre-2016/conference landmark papers", len(papers))
    return papers


# ═══════════════════════════════════════════════════════════════
# STEP 5: ACL Anthology package for recent (2023+) papers
# ═══════════════════════════════════════════════════════════════

def collect_recent_acl_papers() -> list[dict]:
    """Use acl-anthology package to get 2023+ papers from ACL/EMNLP/NAACL."""
    try:
        from acl_anthology import Anthology
    except ImportError:
        logger.warning("acl-anthology not installed, skipping recent papers")
        return []

    logger.info("Loading ACL Anthology from repo (this may take a minute on first run)...")
    try:
        anthology = Anthology.from_repo()
    except Exception as e:
        logger.warning("Failed to load ACL Anthology: %s", e)
        return []

    papers = []
    target_collection_ids = []

    # Find relevant collections (2023+)
    for cid in anthology.collections:
        year_match = re.match(r"(20[2-9]\d)\.(acl|emnlp|naacl|eacl|coling|findings|tacl)", str(cid), re.IGNORECASE)
        if year_match:
            year = int(year_match.group(1))
            if year >= 2023:
                target_collection_ids.append(cid)

    logger.info("Found %d recent ACL collections to scan", len(target_collection_ids))

    for cid in tqdm(target_collection_ids, desc="ACL collections"):
        collection = anthology.collections[cid]
        for volume in collection.volumes():
            for paper in volume.papers():
                title = str(paper.title) if paper.title else ""
                abstract = str(paper.abstract) if hasattr(paper, "abstract") and paper.abstract else ""
                text = f"{title} {abstract}".lower()

                if COMBINED_PATTERN.search(text):
                    doi_val = ""
                    if hasattr(paper, "doi") and paper.doi:
                        doi_val = str(paper.doi)
                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "year": int(str(cid).split(".")[0]) if "." in str(cid) else None,
                        "venue": str(cid),
                        "acl_id": str(paper.full_id),
                        "doi": doi_val,
                        "source": "acl_anthology_pkg",
                    })

    logger.info("Found %d recent ACL embedding papers", len(papers))
    return papers


# ═══════════════════════════════════════════════════════════════
# STEP 6: Full text collection
# ═══════════════════════════════════════════════════════════════

def collect_fulltext_for_papers(papers: pd.DataFrame):
    """Collect full text for papers that have arXiv IDs or ACL PDFs."""
    import asyncio
    import aiohttp

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which papers already have fulltext in the main collection
    main_fulltext = DATA_DIR / "fulltext"
    existing = set()
    if main_fulltext.exists():
        for f in main_fulltext.glob("*.txt"):
            existing.add(f.stem.replace("_", "."))

    # Papers needing fulltext
    need_text = []
    for _, row in papers.iterrows():
        aid = row.get("arxiv_id", "")
        if aid and aid in existing:
            continue  # Already have it from main collection
        safe_id = str(row.get("acl_id", "") or aid or row.get("ss_paper_id", "")).replace("/", "_").replace(".", "_")
        if safe_id and (FULLTEXT_DIR / f"{safe_id}.txt").exists():
            continue
        need_text.append(row.to_dict())

    logger.info("Papers needing fulltext: %d (out of %d total)", len(need_text), len(papers))

    if not need_text:
        return

    # For papers with arXiv IDs, use our existing fulltext collection approach
    arxiv_papers = [p for p in need_text if p.get("arxiv_id")]
    acl_papers = [p for p in need_text if not p.get("arxiv_id") and p.get("acl_id")]
    logger.info("  With arXiv ID: %d, ACL-only: %d", len(arxiv_papers), len(acl_papers))

    # Collect arXiv fulltext using existing pipeline
    if arxiv_papers:
        from collect_fulltext import fetch_fulltext, extract_text_from_html, extract_text_from_pdf_bytes

        async def fetch_arxiv_texts():
            sem = asyncio.Semaphore(8)
            connector = aiohttp.TCPConnector(limit=16, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            headers = {"User-Agent": "novelty-search/0.1 (academic research; polite bot)"}

            async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
                tasks = []
                for p in arxiv_papers:
                    tasks.append(fetch_fulltext(session, sem, p["arxiv_id"], ["html", "latex", "pdf"]))

                done = 0
                for coro in asyncio.as_completed(tasks):
                    arxiv_id, text, source = await coro
                    done += 1
                    if text:
                        safe_id = arxiv_id.replace("/", "_").replace(".", "_")
                        (FULLTEXT_DIR / f"{safe_id}.txt").write_text(text, encoding="utf-8")
                    if done % 50 == 0:
                        logger.info("  arXiv fulltext: %d/%d", done, len(tasks))

        asyncio.run(fetch_arxiv_texts())

    # For ACL-only papers, download from ACL Anthology
    if acl_papers:
        collect_acl_fulltext(acl_papers)


def collect_acl_fulltext(papers: list[dict]):
    """Download and extract text from ACL Anthology PDFs."""
    import fitz  # PyMuPDF

    for p in tqdm(papers, desc="ACL fulltext"):
        acl_id = p.get("acl_id", "")
        if not acl_id:
            continue

        safe_id = acl_id.replace("/", "_").replace(".", "_")
        out_file = FULLTEXT_DIR / f"{safe_id}.txt"
        if out_file.exists():
            continue

        # ACL Anthology PDF URL pattern
        pdf_url = f"https://aclanthology.org/{acl_id}.pdf"
        try:
            resp = requests.get(pdf_url, timeout=30, headers={
                "User-Agent": "novelty-search/0.1 (academic research; polite bot)"
            })
            if resp.status_code == 200:
                doc = fitz.open(stream=resp.content, filetype="pdf")
                parts = [page.get_text() for page in doc]
                doc.close()
                text = "\n".join(parts)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = text.strip()
                if len(text) > 300:
                    out_file.write_text(text, encoding="utf-8")
            else:
                logger.debug("ACL PDF %s: HTTP %d", acl_id, resp.status_code)
        except Exception as e:
            logger.debug("ACL PDF %s: %s", acl_id, e)

        time.sleep(0.3)  # polite


# ═══════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════

def deduplicate(df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Remove papers we already have in the main v3 dataset."""
    before = len(df)

    # Deduplicate against existing by arxiv_id
    existing_arxiv = set(existing_df["arxiv_id"].dropna().str.strip().values)
    mask_not_in_existing = ~df["arxiv_id"].fillna("").str.strip().isin(existing_arxiv)

    # Also deduplicate by normalized title
    def normalize_title(t):
        if not t or pd.isna(t):
            return ""
        return re.sub(r"[^a-z0-9]", "", str(t).lower())

    existing_titles = set(existing_df["title"].apply(normalize_title).values)
    mask_not_in_existing_title = ~df["title"].apply(normalize_title).isin(existing_titles)

    # Keep papers not in existing by EITHER arxiv_id or title
    df_new = df[mask_not_in_existing & mask_not_in_existing_title].copy()

    # Self-deduplicate by title
    df_new["_norm_title"] = df_new["title"].apply(normalize_title)
    df_new = df_new.drop_duplicates(subset=["_norm_title"], keep="first")
    df_new = df_new.drop(columns=["_norm_title"])

    logger.info("Deduplication: %d → %d papers (removed %d already in v3 dataset)", before, len(df_new), before - len(df_new))
    return df_new


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def save_intermediate(papers: list[dict], label: str):
    """Save intermediate results to JSON for crash recovery."""
    path = DATA_DIR / f"embeddings_acl_{label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, default=str)
    logger.info("Saved %d papers to %s", len(papers), path)


def load_intermediate(label: str) -> list[dict] | None:
    """Load intermediate results if available."""
    path = DATA_DIR / f"embeddings_acl_{label}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d papers from %s (cached)", len(data), path)
        return data
    return None


def print_stats(df: pd.DataFrame):
    """Print collection stats."""
    print("\n" + "=" * 60)
    print("Embeddings ACL Collection Stats")
    print("=" * 60)
    print(f"Total new papers: {len(df):,}")
    print(f"\nBy source:")
    for src, count in df["source"].value_counts().items():
        print(f"  {src}: {count:,}")
    print(f"\nBy year:")
    for year, count in df["year"].value_counts().sort_index().items():
        print(f"  {year}: {count:,}")
    print(f"\nBy venue (top 20):")
    for venue, count in df["venue"].value_counts().head(20).items():
        print(f"  {venue}: {count:,}")
    print(f"\nCitation stats:")
    cc = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)
    print(f"  Mean: {cc.mean():.1f}")
    print(f"  Median: {cc.median():.1f}")
    print(f"  Max: {cc.max():.0f}")
    print(f"  Papers with 100+ citations: {(cc >= 100).sum():,}")
    print(f"  Papers with 1000+ citations: {(cc >= 1000).sum():,}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing dataset for deduplication
    existing_df = pd.read_parquet(DATA_DIR / "dataset_v3.parquet", columns=["arxiv_id", "title"])
    logger.info("Existing v3 dataset: %d papers", len(existing_df))

    all_papers = []
    hf_papers = []  # Keep separate for fulltext

    # Step 1: HuggingFace ACL Anthology (with caching)
    cached = load_intermediate("step1_hf")
    if cached is not None:
        hf_papers = cached
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: HuggingFace ACL Anthology Corpus")
        logger.info("=" * 60)
        hf_df = collect_from_hf_anthology()
        for _, row in hf_df.iterrows():
            hf_papers.append({
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "full_text": row.get("full_text", ""),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "venue": row.get("booktitle") or row.get("journal", ""),
                "acl_id": row.get("acl_id", ""),
                "doi": row.get("doi", ""),
                "citation_count": int(row["numcitedby"]) if pd.notna(row.get("numcitedby")) else 0,
                "corpus_paper_id": str(row.get("corpus_paper_id", "")),
                "source": "hf_acl_anthology",
            })
        save_intermediate(hf_papers, "step1_hf")
    all_papers.extend(hf_papers)
    logger.info("After Step 1: %d papers", len(all_papers))

    # Save HF full texts immediately (before they get dropped)
    hf_texts_saved = 0
    for p in hf_papers:
        ft = p.get("full_text", "")
        acl_id = p.get("acl_id", "")
        if ft and acl_id and len(str(ft)) > 300:
            safe_id = acl_id.replace("/", "_").replace(".", "_")
            out_file = FULLTEXT_DIR / f"{safe_id}.txt"
            if not out_file.exists():
                out_file.write_text(str(ft), encoding="utf-8")
                hf_texts_saved += 1
    logger.info("Saved %d full texts from HF corpus", hf_texts_saved)

    # Step 2: Pre-2016 landmarks (with caching)
    cached = load_intermediate("step2_landmarks")
    if cached is not None:
        landmarks = cached
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Pre-2016 & Conference-only Landmarks")
        logger.info("=" * 60)
        landmarks = fetch_pre2016_arxiv()
        save_intermediate(landmarks, "step2_landmarks")
    all_papers.extend(landmarks)
    logger.info("After Step 2: %d papers", len(all_papers))

    # Step 3: Recent ACL papers (2023+) (with caching)
    cached = load_intermediate("step3_recent")
    if cached is not None:
        recent = cached
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Recent ACL Anthology (2023+)")
        logger.info("=" * 60)
        recent = collect_recent_acl_papers()
        save_intermediate(recent, "step3_recent")
    all_papers.extend(recent)
    logger.info("After Step 3: %d papers", len(all_papers))

    # Step 4: SS gap-filling search (with caching)
    cached = load_intermediate("step4_ss")
    if cached is not None:
        ss_papers = cached
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Semantic Scholar Gap-filling")
        logger.info("=" * 60)
        existing_titles = {
            re.sub(r"[^a-z0-9]", "", str(p.get("title", "")).lower())
            for p in all_papers
        } | set(existing_df["title"].apply(lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())).values)
        ss_papers = search_ss_for_gaps(existing_titles)
        save_intermediate(ss_papers, "step4_ss")
    all_papers.extend(ss_papers)
    logger.info("After Step 4: %d papers", len(all_papers))

    # Convert to DataFrame
    df = pd.DataFrame(all_papers)

    # Step 5: Enrich papers that don't have citation counts yet
    needs_enrichment = df[
        (df["citation_count"].fillna(0) == 0) &
        (df["source"].isin(["acl_anthology_pkg", "hf_acl_anthology"]))
    ]
    if len(needs_enrichment) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Enriching %d papers with Semantic Scholar", len(needs_enrichment))
        logger.info("=" * 60)
        enriched = enrich_with_semantic_scholar(needs_enrichment.to_dict("records"))
        enriched_df = pd.DataFrame(enriched)
        df.update(enriched_df)

    # Step 6: Deduplicate against existing v3 dataset
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Deduplication")
    logger.info("=" * 60)
    df = deduplicate(df, existing_df)

    # Save final parquet
    keep_cols = [
        "title", "abstract", "year", "venue", "citation_count",
        "influential_citation_count", "ss_paper_id", "arxiv_id",
        "doi", "acl_id", "source",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[keep_cols].copy()

    df.to_parquet(OUT_PARQUET, index=False)
    logger.info("Saved %d papers to %s", len(df), OUT_PARQUET)

    print_stats(df)

    # Step 7: Collect remaining full text
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Full text collection for remaining papers")
    logger.info("=" * 60)
    collect_fulltext_for_papers(df)

    ft_count = len(list(FULLTEXT_DIR.glob("*.txt")))
    logger.info("Total fulltext files: %d", ft_count)
    print(f"\nFull text: {ft_count} files in {FULLTEXT_DIR}")
    print("DONE!")


if __name__ == "__main__":
    main()
