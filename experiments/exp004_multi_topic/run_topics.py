#!/usr/bin/env python3
"""Multi-topic novelty detection replication study — using LOCAL data.

Replicates exp001-003 findings across 10 diverse CS/ML topics + 1 overall.
Uses 442K arXiv + 71K ACL papers with fulltext.

Pipeline per topic:
  Local filter → fulltext chunks → LLM idea extraction → embed ideas
  → max-pool paper embeddings → kNN novelty + D_M + BPI → validate vs citations
"""
import asyncio
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# === Paths ===
PROJECT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data" / "processed" / "v3"
sys.path.insert(0, str(PROJECT / "src"))

# Load .env
_env = PROJECT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# === Models ===
LLM_MODEL = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
EXTRACTION_CONCURRENCY = 200
EMBEDDING_BATCH = 500

# === Chunking (same as exp001) ===
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = 3000 * CHARS_PER_TOKEN  # 12000
MIN_CHUNK_CHARS = 80 * CHARS_PER_TOKEN  # 320

SYSTEM_PROMPT = "You extract scientific ideas from paper excerpts. Output JSON only."
USER_PROMPT = """Extract novel ideas from this scientific paper excerpt.
- Return JSON: {{"ideas": ["idea1", "idea2", ...]}}
- 0-5 ideas, each 5-20 words, specific
- Focus on novel methods, techniques, models, findings
- Skip background, related work, references

Title: {title}

{chunk}"""

# ============================================================================
# LOAD DATASETS (once)
# ============================================================================
print("Loading datasets...")
_t0 = time.time()
ARXIV_DF = pd.read_parquet(DATA_DIR / "dataset_v3.parquet")
ACL_DF = pd.read_parquet(DATA_DIR / "acl_all.parquet")
print(f"  arXiv: {len(ARXIV_DF):,}, ACL: {len(ACL_DF):,} ({time.time()-_t0:.1f}s)")

FULLTEXT_DIR = DATA_DIR / "fulltext"
ACL_FULLTEXT_DIR = DATA_DIR / "acl_all_fulltext"


# ============================================================================
# TOPIC DEFINITIONS
# ============================================================================
TOPICS = {
    "transformers": {
        "keywords": [
            "transformer", "self-attention", "multi-head attention",
            "pre-trained language model", "bert", "gpt",
            "encoder-decoder attention", "positional encoding",
        ],
        "max_papers": 3000,
        "golden_high": [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional",
            "Language Models are Few-Shot Learners",
        ],
        "golden_medium": [
            "Reformer: The Efficient Transformer",
            "Exploring the Limits of Transfer Learning",
        ],
    },
    "reinforcement_learning": {
        "keywords": [
            "reinforcement learning", "policy gradient", "q-learning",
            "actor-critic", "reward shaping", "markov decision",
            "deep reinforcement", "policy optimization",
        ],
        "max_papers": 3000,
        "golden_high": [
            "Playing Atari with Deep Reinforcement Learning",
            "Human-level control through deep reinforcement",
            "Proximal Policy Optimization Algorithms",
        ],
        "golden_medium": [
            "Soft Actor-Critic",
            "Continuous control with deep reinforcement learning",
        ],
    },
    "gans": {
        "keywords": [
            "generative adversarial", "gan ", "adversarial training",
            "generator discriminator", "image generation adversarial",
            "conditional adversarial", "wasserstein distance",
        ],
        "max_papers": 3000,
        "golden_high": [
            "Generative Adversarial",
            "Unsupervised Representation Learning with Deep Convolutional Generative",
            "Image-to-Image Translation with Conditional Adversarial",
        ],
        "golden_medium": [
            "Style-Based Generator Architecture",
            "Unpaired Image-to-Image Translation",
        ],
    },
    "graph_neural_networks": {
        "keywords": [
            "graph neural network", "graph convolutional",
            "node classification", "graph attention",
            "message passing neural", "graph representation learning",
            "link prediction graph",
        ],
        "max_papers": 3000,
        "golden_high": [
            "Semi-Supervised Classification with Graph Convolutional",
            "Inductive Representation Learning on Large Graphs",
            "Graph Attention Networks",
        ],
        "golden_medium": [
            "How Powerful are Graph Neural Networks",
            "Neural Message Passing for Quantum Chemistry",
        ],
    },
    "object_detection": {
        "keywords": [
            "object detection", "bounding box", "anchor-based",
            "region proposal", "single-shot detector",
            "feature pyramid", "non-maximum suppression",
        ],
        "max_papers": 3000,
        "golden_high": [
            "You Only Look Once",
            "Faster R-CNN",
            "Feature Pyramid Networks for Object Detection",
        ],
        "golden_medium": [
            "SSD: Single Shot MultiBox Detector",
            "End-to-End Object Detection with Transformers",
        ],
    },
    "neural_machine_translation": {
        "keywords": [
            "neural machine translation", "sequence to sequence",
            "encoder decoder translation", "beam search",
            "attention mechanism translation", "subword",
            "back-translation", "multilingual translation",
        ],
        "max_papers": 2500,
        "golden_high": [
            "Sequence to Sequence Learning with Neural Networks",
            "Neural Machine Translation by Jointly Learning to Align",
            "Google's Neural Machine Translation System",
        ],
        "golden_medium": [
            "Effective Approaches to Attention-based Neural Machine Translation",
            "Convolutional Sequence to Sequence Learning",
        ],
    },
    "knowledge_distillation": {
        "keywords": [
            "knowledge distillation", "model compression",
            "teacher student", "network pruning",
            "model distillation", "quantization neural",
            "compact model", "lightweight network",
        ],
        "max_papers": 2500,
        "golden_high": [
            "Distilling the Knowledge in a Neural Network",
            "DistilBERT",
        ],
        "golden_medium": [
            "TinyBERT",
            "Born Again Neural Networks",
        ],
    },
    "diffusion_models": {
        "keywords": [
            "diffusion model", "denoising diffusion",
            "score matching", "noise schedule",
            "diffusion probabilistic", "score-based generative",
            "latent diffusion", "classifier-free guidance",
        ],
        "max_papers": 2500,
        "golden_high": [
            "Denoising Diffusion Probabilistic Models",
            "High-Resolution Image Synthesis with Latent Diffusion",
            "Diffusion Models Beat GANs",
        ],
        "golden_medium": [
            "Score-Based Generative Modeling through Stochastic Differential",
            "Denoising Diffusion Implicit Models",
        ],
    },
    "federated_learning": {
        "keywords": [
            "federated learning", "federated optimization",
            "privacy preserving", "differential privacy",
            "communication efficient", "federated averaging",
            "data heterogeneity federated",
        ],
        "max_papers": 2500,
        "golden_high": [
            "Communication-Efficient Learning of Deep Networks from Decentralized",
        ],
        "golden_medium": [
            "Federated Optimization in Heterogeneous Networks",
            "Advances and Open Problems in Federated Learning",
        ],
    },
    "few_shot_meta_learning": {
        "keywords": [
            "few-shot learning", "meta-learning",
            "learning to learn", "prototypical network",
            "few-shot classification", "episodic training",
            "model-agnostic meta",
        ],
        "max_papers": 2500,
        "golden_high": [
            "Model-Agnostic Meta-Learning",
            "Prototypical Networks for Few-shot Learning",
            "Matching Networks for One Shot Learning",
        ],
        "golden_medium": [
            "Learning to Compare: Relation Network",
        ],
    },
}


# ============================================================================
# PHASE 1: FILTER & MAP FULLTEXT
# ============================================================================
def find_fulltext_arxiv(arxiv_id):
    """Map arxiv_id to fulltext path."""
    if not isinstance(arxiv_id, str) or not arxiv_id:
        return None
    fname = arxiv_id.replace("/", "_").replace(".", "_") + ".txt"
    p = FULLTEXT_DIR / fname
    # Also try with dots preserved
    if not p.exists():
        p = FULLTEXT_DIR / (arxiv_id.replace("/", "_") + ".txt")
    if not p.exists():
        # Try just the raw id
        p = FULLTEXT_DIR / (arxiv_id + ".txt")
    return str(p) if p.exists() else None


def find_fulltext_acl(acl_id):
    """Map acl_id to fulltext path."""
    if not isinstance(acl_id, str) or not acl_id:
        return None
    fname = acl_id.replace(".", "_").replace("/", "_") + ".txt"
    p = ACL_FULLTEXT_DIR / fname
    return str(p) if p.exists() else None


def filter_topic(keywords, max_papers):
    """Filter papers from arXiv+ACL by keywords in title+abstract."""
    kw_lower = [k.lower() for k in keywords]

    def matches(row):
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text = title + " " + abstract
        return any(kw in text for kw in kw_lower)

    # Filter arXiv
    arxiv_mask = ARXIV_DF.apply(matches, axis=1)
    arxiv_sub = ARXIV_DF[arxiv_mask].copy()
    arxiv_sub["paper_id"] = arxiv_sub["arxiv_id"]
    arxiv_sub["source"] = "arxiv"
    arxiv_sub["fulltext_path"] = arxiv_sub["arxiv_id"].apply(find_fulltext_arxiv)

    # Filter ACL
    acl_mask = ACL_DF.apply(matches, axis=1)
    acl_sub = ACL_DF[acl_mask].copy()
    acl_sub["paper_id"] = acl_sub["acl_id"]
    acl_sub["source"] = "acl"
    acl_sub["fulltext_path"] = acl_sub["acl_id"].apply(find_fulltext_acl)

    # Unify columns
    cols = ["paper_id", "title", "abstract", "year", "citation_count", "source", "fulltext_path"]
    arxiv_out = arxiv_sub[[c for c in cols if c in arxiv_sub.columns]].copy()
    acl_out = acl_sub[[c for c in cols if c in acl_sub.columns]].copy()

    combined = pd.concat([arxiv_out, acl_out], ignore_index=True)
    combined = combined.dropna(subset=["citation_count", "year"])
    combined["year"] = combined["year"].astype(int)
    combined["citation_count"] = combined["citation_count"].astype(int)
    combined = combined[combined["year"] >= 2005]

    # Prefer papers with fulltext
    has_ft = combined[combined["fulltext_path"].notna()]
    no_ft = combined[combined["fulltext_path"].isna()]

    if len(has_ft) > max_papers:
        # Smart sample: keep high-citation up to 40% budget, sample rest
        high = has_ft[has_ft["citation_count"] >= 50]
        max_high = int(max_papers * 0.4)
        if len(high) > max_high:
            high = high.nlargest(max_high, "citation_count")
        rest = has_ft[~has_ft.index.isin(high.index)]
        n_rest = max_papers - len(high)
        if n_rest > 0 and len(rest) > n_rest:
            rest = rest.sample(n=n_rest, random_state=42)
        elif n_rest <= 0:
            rest = rest.iloc[:0]  # empty
        combined = pd.concat([high, rest]).reset_index(drop=True)
    elif len(has_ft) < max_papers and len(no_ft) > 0:
        n_add = min(max_papers - len(has_ft), len(no_ft))
        combined = pd.concat([has_ft, no_ft.head(n_add)]).reset_index(drop=True)
    else:
        combined = has_ft.reset_index(drop=True)

    return combined


# ============================================================================
# PHASE 2: CHUNKING + IDEA EXTRACTION
# ============================================================================
def chunk_text(text):
    """Split text into chunks by paragraphs."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            chunk = "\n\n".join(current)
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)
    if current:
        chunk = "\n\n".join(current)
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
    return chunks


async def extract_ideas_batch(papers_df):
    """Extract ideas from fulltext chunks (or abstracts as fallback)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(EXTRACTION_CONCURRENCY)

    # Build task list: (paper_id, title, chunk_text)
    tasks = []
    papers_with_ft = 0
    papers_abstract_only = 0
    for _, row in papers_df.iterrows():
        ft_path = row.get("fulltext_path")
        title = str(row.get("title", ""))
        pid = str(row["paper_id"])
        if ft_path and Path(ft_path).exists():
            text = Path(ft_path).read_text(encoding="utf-8", errors="replace")
            cks = chunk_text(text)
            if cks:
                papers_with_ft += 1
                for ci, ck in enumerate(cks):
                    tasks.append((pid, title, ck))
                continue
        # Fallback to abstract
        abstract = str(row.get("abstract", ""))
        if len(abstract) > 100:
            papers_abstract_only += 1
            tasks.append((pid, title, abstract))

    print(f"    {papers_with_ft} papers with fulltext, {papers_abstract_only} abstract-only")
    print(f"    {len(tasks)} total chunks to process")

    results = []
    total_in = total_out = errors = 0

    async def process_one(pid, title, chunk):
        nonlocal total_in, total_out, errors
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(title=title, chunk=chunk[:12000])},
        ]
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        response_format={"type": "json_object"},
                        max_completion_tokens=200,
                    )
                content = resp.choices[0].message.content
                data = json.loads(content)
                ideas = data.get("ideas", [])
                ideas = [i.strip() for i in ideas if isinstance(i, str) and 3 < len(i.strip()) < 300]
                total_in += resp.usage.prompt_tokens if resp.usage else 0
                total_out += resp.usage.completion_tokens if resp.usage else 0
                return {"paper_id": pid, "ideas": ideas}
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    await asyncio.sleep(min(2**attempt * 3, 30))
                elif attempt < 2:
                    await asyncio.sleep(1)
                else:
                    errors += 1
                    return {"paper_id": pid, "ideas": []}
        return {"paper_id": pid, "ideas": []}

    coros = [process_one(pid, title, chunk) for pid, title, chunk in tasks]
    done = 0
    t0 = time.time()
    for batch_start in range(0, len(coros), 500):
        batch = coros[batch_start:batch_start + 500]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            done += 1
            if isinstance(r, Exception):
                errors += 1
                results.append({"paper_id": "error", "ideas": []})
            else:
                results.append(r)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        cost = total_in / 1e6 * 0.10 + total_out / 1e6 * 0.40
        print(f"    [{done}/{len(coros)}] {rate:.0f}/s | ${cost:.3f} | err={errors}")

    stats = {
        "n_chunks": len(tasks), "n_ideas": sum(len(r["ideas"]) for r in results),
        "papers_fulltext": papers_with_ft, "papers_abstract": papers_abstract_only,
        "tokens_in": total_in, "tokens_out": total_out, "errors": errors,
        "elapsed": time.time() - t0,
    }
    return results, stats


# ============================================================================
# PHASE 3: EMBEDDING
# ============================================================================
async def embed_ideas(idea_texts):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(10)
    embeddings = [None] * len(idea_texts)

    async def process_batch(start, batch):
        async with semaphore:
            resp = await client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            return start, [d.embedding for d in resp.data]

    tasks = []
    for i in range(0, len(idea_texts), EMBEDDING_BATCH):
        batch = [t[:8000] for t in idea_texts[i:i + EMBEDDING_BATCH]]
        tasks.append(process_batch(i, batch))

    for coro in asyncio.as_completed(tasks):
        start, embs = await coro
        for j, emb in enumerate(embs):
            embeddings[start + j] = emb

    return np.array(embeddings, dtype=np.float32)


# ============================================================================
# PHASE 4: METRICS (batch-by-year, fast)
# ============================================================================
def compute_all_metrics(df, idea_results, idea_embeddings, idea_meta):
    paper_idea_idxs = {}
    for i, m in enumerate(idea_meta):
        pid = m["paper_id"]
        paper_idea_idxs.setdefault(pid, []).append(i)

    id_to_cit = dict(zip(df["paper_id"], df["citation_count"]))
    id_to_year = dict(zip(df["paper_id"], df["year"].astype(int)))

    # Paper embeddings: max-pool (best) + mean-pool
    paper_embs_max = {}
    paper_embs_mean = {}
    for pid, idxs in paper_idea_idxs.items():
        embs = idea_embeddings[idxs]
        mx = embs.max(axis=0)
        mx /= np.linalg.norm(mx) + 1e-10
        paper_embs_max[pid] = mx
        mn = embs.mean(axis=0)
        mn /= np.linalg.norm(mn) + 1e-10
        paper_embs_mean[pid] = mn

    valid_pids = [p for p in sorted(paper_idea_idxs) if p in id_to_cit and pd.notna(id_to_cit.get(p))]
    if len(valid_pids) < 50:
        print(f"    WARNING: Only {len(valid_pids)} valid papers")
        if len(valid_pids) < 20:
            return pd.DataFrame()

    SPLIT_YEAR = 2020
    test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]
    if len(test_pids) < 30:
        SPLIT_YEAR = int(np.median([id_to_year[p] for p in valid_pids]))
        test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]

    all_pids = valid_pids
    all_years = np.array([id_to_year[p] for p in all_pids])

    # === 1. kNN novelty (batch by year) ===
    def knn_by_year(pids, paper_embs, k=10):
        emb_matrix = np.array([paper_embs[p] for p in pids], dtype=np.float32)
        years = np.array([id_to_year[p] for p in pids])
        scores = np.full(len(pids), np.nan)
        for year in sorted(set(years)):
            prior = years < year
            curr = years == year
            if prior.sum() < k + 1:
                continue
            nn = NearestNeighbors(n_neighbors=min(k, prior.sum()), metric="cosine")
            nn.fit(emb_matrix[prior])
            dists, _ = nn.kneighbors(emb_matrix[curr])
            scores[curr] = dists.mean(axis=1)
        return dict(zip(pids, scores.tolist()))

    print("    Computing kNN novelty...")
    knn_max = knn_by_year(all_pids, paper_embs_max, k=10)
    knn_mean = knn_by_year(all_pids, paper_embs_mean, k=10)

    # === 2. Spectrum-based metrics ===
    print("    Computing spectrum metrics...")
    idea_embs_normed = normalize(idea_embeddings)
    K = min(200, len(idea_embeddings) // 10)
    K = max(K, 20)
    pca_dims = min(256, idea_embeddings.shape[1], len(idea_embeddings) - 1)
    pca = PCA(n_components=pca_dims, random_state=42)
    idea_pca = normalize(pca.fit_transform(idea_embs_normed))
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=min(4096, len(idea_pca)))
    labels = km.fit_predict(idea_pca)
    centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
    for c in range(K):
        m = labels == c
        if m.sum():
            centroids[c] = idea_embs_normed[m].mean(axis=0)
    centroids = normalize(centroids)

    # Mean spectrum
    def spectrum_mean(p_embs, cents):
        if len(p_embs) == 0:
            return np.zeros(len(cents))
        return np.mean(normalize(cents) @ normalize(p_embs).T, axis=1)

    all_spectra = np.array([
        spectrum_mean(idea_embs_normed[paper_idea_idxs.get(p, [])], centroids) for p in all_pids
    ], dtype=np.float32)

    # Temporal D_M
    dm_scores = np.full(len(all_pids), np.nan)
    r_scores = np.full(len(all_pids), np.nan)
    for year in sorted(set(all_years)):
        prior = all_years < year
        curr = all_years == year
        if prior.sum() < max(30, K + 5):
            continue
        p_spec = all_spectra[prior]
        mean = p_spec.mean(axis=0)
        cov = np.cov(p_spec.T) + np.eye(K) * 1e-4
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        for i in np.where(curr)[0]:
            try:
                dm_scores[i] = mahalanobis(all_spectra[i], mean, cov_inv)
            except Exception:
                pass
        # Rarity
        prev = p_spec.mean(axis=0)
        prev = prev / (prev.sum() + 1e-10)
        log_p = np.log(np.maximum(prev, 1e-10))
        r_scores[curr] = -np.sum(all_spectra[curr] * log_p, axis=1)

    c_scores = np.sum(all_spectra ** 2, axis=1)

    # Spectrum kNN for BPI
    nn_spec = np.full(len(all_pids), np.nan)
    for year in sorted(set(all_years)):
        prior = all_years < year
        curr = all_years == year
        if prior.sum() < 11:
            continue
        nn = NearestNeighbors(n_neighbors=min(10, prior.sum()), metric="euclidean")
        nn.fit(all_spectra[prior])
        dists, _ = nn.kneighbors(all_spectra[curr])
        nn_spec[curr] = dists.mean(axis=1)

    bpi_original = r_scores * c_scores * nn_spec

    knn_arr = np.array([knn_max.get(p, np.nan) for p in all_pids])
    combined = (np.power(np.maximum(knn_arr, 0), 0.5) *
                np.power(np.maximum(dm_scores, 0), 0.5) *
                np.power(np.maximum(r_scores, 0), 0.1))

    # Build DataFrame
    metrics_df = pd.DataFrame({
        "paper_id": all_pids,
        "title": [df.set_index("paper_id").loc[p, "title"] if p in df["paper_id"].values else "" for p in all_pids],
        "year": [int(id_to_year[p]) for p in all_pids],
        "citation_count": [int(id_to_cit[p]) for p in all_pids],
        "knn_max_pool": [knn_max.get(p, np.nan) for p in all_pids],
        "knn_mean_pool": [knn_mean.get(p, np.nan) for p in all_pids],
        "D_M": dm_scores, "R": r_scores, "C": c_scores,
        "BPI_original": bpi_original, "combined_best": combined,
        "n_ideas": [len(paper_idea_idxs.get(p, [])) for p in all_pids],
    })

    # Evaluate on test set
    test = metrics_df[metrics_df["year"] >= SPLIT_YEAR].dropna(subset=["knn_max_pool", "citation_count"])
    log_cit = np.log1p(test["citation_count"].values)

    eval_results = {}
    for col in ["knn_max_pool", "knn_mean_pool", "D_M", "R", "C", "BPI_original", "combined_best"]:
        vals = test[col].values
        valid = ~np.isnan(vals)
        if valid.sum() < 20:
            eval_results[col] = {"sp_r": float("nan"), "n": int(valid.sum())}
            continue
        r, p = spearmanr(vals[valid], log_cit[valid])
        eval_results[col] = {"sp_r": float(r), "sp_p": float(p), "n": int(valid.sum())}

    metrics_df.attrs["eval"] = eval_results
    metrics_df.attrs["split_year"] = SPLIT_YEAR
    metrics_df.attrs["n_test"] = len(test)
    metrics_df.attrs["K_clusters"] = K
    return metrics_df


# ============================================================================
# PHASE 5: VALIDATION
# ============================================================================
def validate_topic(metrics_df, golden_high, golden_medium, topic_name):
    eval_results = metrics_df.attrs.get("eval", {})
    split_year = metrics_df.attrs.get("split_year", 2020)

    result = {
        "topic": topic_name,
        "n_papers": len(metrics_df),
        "n_test": metrics_df.attrs.get("n_test", 0),
        "split_year": split_year,
        "metrics": eval_results,
        "golden_high_results": [],
        "golden_medium_results": [],
    }

    for col in ["knn_max_pool", "combined_best"]:
        if col in metrics_df.columns:
            metrics_df[f"{col}_pct"] = metrics_df[col].rank(pct=True) * 100

    for title_frag in golden_high:
        mask = metrics_df["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            row = metrics_df[mask].iloc[0]
            result["golden_high_results"].append({
                "query": title_frag[:60],
                "title": row["title"][:80],
                "year": int(row["year"]),
                "citations": int(row["citation_count"]),
                "knn_pct": float(row.get("knn_max_pool_pct", np.nan)),
                "combined_pct": float(row.get("combined_best_pct", np.nan)),
            })

    for title_frag in golden_medium:
        mask = metrics_df["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            row = metrics_df[mask].iloc[0]
            result["golden_medium_results"].append({
                "query": title_frag[:60],
                "title": row["title"][:80],
                "year": int(row["year"]),
                "citations": int(row["citation_count"]),
                "knn_pct": float(row.get("knn_max_pool_pct", np.nan)),
            })

    # Trash detection
    test = metrics_df[metrics_df["year"] >= split_year].dropna(subset=["knn_max_pool", "citation_count"])
    if len(test) > 50:
        q20 = test["knn_max_pool"].quantile(0.20)
        flagged = test[test["knn_max_pool"] <= q20]
        if len(flagged) > 0:
            result["trash_detection"] = {
                "n_flagged": len(flagged),
                "precision_le19": float((flagged["citation_count"] <= 19).sum() / len(flagged)),
                "max_cit_flagged": int(flagged["citation_count"].max()),
            }
    return result


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run_topic(topic_name, config):
    topic_dir = EXP_DIR / topic_name
    topic_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TOPIC: {topic_name}")
    print(f"{'='*70}")

    results_path = topic_dir / "results.json"
    if results_path.exists():
        print(f"  CACHED — loading")
        with open(results_path) as f:
            return json.load(f)

    # Phase 1: Filter
    papers_path = topic_dir / "papers.parquet"
    if papers_path.exists():
        print("  Loading cached papers...")
        df = pd.read_parquet(papers_path)
    else:
        print("  Phase 1: Filtering papers...")
        df = filter_topic(config["keywords"], config["max_papers"])
        df.to_parquet(papers_path, index=False)

    n_ft = df["fulltext_path"].notna().sum()
    print(f"  Papers: {len(df)}, with fulltext: {n_ft} ({n_ft/len(df)*100:.0f}%)")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}, median cit: {df['citation_count'].median():.0f}")

    # Check golden set
    for g in config.get("golden_high", []) + config.get("golden_medium", []):
        mask = df["title"].str.contains(g, case=False, na=False)
        if mask.any():
            row = df[mask].iloc[0]
            print(f"  GOLDEN: {row['title'][:55]}... ({row['citation_count']} cit)")
        else:
            print(f"  GOLDEN MISSING: {g[:50]}")

    # Phase 2: Extract ideas
    ideas_path = topic_dir / "ideas.json"
    if ideas_path.exists():
        print("  Loading cached ideas...")
        with open(ideas_path) as f:
            idea_results = json.load(f)
    else:
        print("  Phase 2: Extracting ideas...")
        idea_results, stats = asyncio.run(extract_ideas_batch(df))
        with open(ideas_path, "w") as f:
            json.dump(idea_results, f, ensure_ascii=False)
        with open(topic_dir / "extraction_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        cost = stats["tokens_in"] / 1e6 * 0.10 + stats["tokens_out"] / 1e6 * 0.40
        print(f"  Ideas: {stats['n_ideas']} from {stats['n_chunks']} chunks, ${cost:.3f}")

    # Flatten ideas
    idea_texts, idea_meta = [], []
    seen = set()
    for rec in idea_results:
        for idea in rec["ideas"]:
            ic = idea.strip().lower()
            if ic not in seen and len(idea.strip()) > 3:
                seen.add(ic)
                idea_texts.append(idea.strip())
                idea_meta.append({"paper_id": rec["paper_id"], "text": idea.strip()})

    print(f"  Unique ideas: {len(idea_texts)}")
    if len(idea_texts) < 50:
        print("  ERROR: Too few ideas. Skipping.")
        return {"topic": topic_name, "error": "too_few_ideas"}

    # Phase 3: Embed
    emb_path = topic_dir / "idea_embeddings.npy"
    if emb_path.exists():
        print("  Loading cached embeddings...")
        idea_embeddings = np.load(emb_path)
    else:
        print("  Phase 3: Embedding ideas...")
        idea_embeddings = asyncio.run(embed_ideas(idea_texts))
        np.save(emb_path, idea_embeddings)
    print(f"  Embeddings: {idea_embeddings.shape}")

    # Phase 4: Metrics
    print("  Phase 4: Computing metrics...")
    metrics_df = compute_all_metrics(df, idea_results, idea_embeddings, idea_meta)
    if len(metrics_df) == 0:
        return {"topic": topic_name, "error": "metrics_failed"}
    metrics_df.to_parquet(topic_dir / "metrics.parquet", index=False)

    # Phase 5: Validate
    print("  Phase 5: Validating...")
    result = validate_topic(metrics_df, config.get("golden_high", []), config.get("golden_medium", []), topic_name)

    print(f"\n  --- {topic_name} RESULTS ---")
    print(f"  Papers: {result['n_papers']}, Test: {result['n_test']} (>={result['split_year']})")
    for m, ev in result.get("metrics", {}).items():
        r = ev.get("sp_r", float("nan"))
        star = " ***" if isinstance(r, float) and r > 0.2 else ""
        print(f"    {m:20s}: r={r:+.4f} (n={ev.get('n',0)}){star}")
    for g in result.get("golden_high_results", []):
        print(f"    GOLDEN HIGH: {g['query'][:40]} pct={g.get('knn_pct',0):.1f}% (cit={g['citations']})")
    for g in result.get("golden_medium_results", []):
        print(f"    GOLDEN MED:  {g['query'][:40]} pct={g.get('knn_pct',0):.1f}% (cit={g['citations']})")
    if result.get("trash_detection"):
        td = result["trash_detection"]
        print(f"    TRASH: flagged={td['n_flagged']}, prec={td['precision_le19']:.1%}, max_cit={td['max_cit_flagged']}")

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def build_overall_subset(all_results):
    """Build mixed-topic subset from all collected papers."""
    print(f"\n{'='*70}")
    print("OVERALL: Mixed-topic subset")
    print(f"{'='*70}")

    overall_dir = EXP_DIR / "overall_mixed"
    overall_dir.mkdir(exist_ok=True)
    results_path = overall_dir / "results.json"
    if results_path.exists():
        print("  CACHED")
        with open(results_path) as f:
            return json.load(f)

    # Combine
    all_dfs, all_ideas = [], []
    for tn in TOPICS:
        pp = EXP_DIR / tn / "papers.parquet"
        ip = EXP_DIR / tn / "ideas.json"
        if pp.exists() and ip.exists():
            tdf = pd.read_parquet(pp)
            tdf["source_topic"] = tn
            all_dfs.append(tdf)
            with open(ip) as f:
                all_ideas.extend(json.load(f))

    if not all_dfs:
        return {"topic": "overall_mixed", "error": "no_data"}

    combined = pd.concat(all_dfs).drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
    print(f"  Combined unique: {len(combined)}")

    # Smart sample
    high = combined[combined["citation_count"] >= 500]
    med_high = combined[(combined["citation_count"] >= 100) & (combined["citation_count"] < 500)]
    med = combined[(combined["citation_count"] >= 20) & (combined["citation_count"] < 100)]
    low = combined[combined["citation_count"] < 20]
    n_target = 5000
    sample = pd.concat([
        high, med_high,
        med.sample(n=min(1500, len(med)), random_state=42) if len(med) > 1500 else med,
        low.sample(n=min(max(0, n_target - len(high) - len(med_high) - min(1500, len(med))), len(low)),
                   random_state=42) if len(low) > 0 else low,
    ]).drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
    sample.to_parquet(overall_dir / "papers.parquet", index=False)
    print(f"  Sample: {len(sample)} (500+: {(sample['citation_count']>=500).sum()}, "
          f"100-499: {((sample['citation_count']>=100)&(sample['citation_count']<500)).sum()}, "
          f"20-99: {((sample['citation_count']>=20)&(sample['citation_count']<100)).sum()}, "
          f"<20: {(sample['citation_count']<20).sum()})")

    # Filter ideas to sample
    sample_pids = set(sample["paper_id"])
    filtered = [r for r in all_ideas if r["paper_id"] in sample_pids]
    idea_texts, idea_meta = [], []
    seen = set()
    for rec in filtered:
        for idea in rec["ideas"]:
            ic = idea.strip().lower()
            if ic not in seen and len(idea.strip()) > 3:
                seen.add(ic)
                idea_texts.append(idea.strip())
                idea_meta.append({"paper_id": rec["paper_id"], "text": idea.strip()})
    print(f"  Unique ideas: {len(idea_texts)}")

    emb_path = overall_dir / "idea_embeddings.npy"
    if emb_path.exists():
        ie = np.load(emb_path)
    else:
        print("  Embedding ideas...")
        ie = asyncio.run(embed_ideas(idea_texts))
        np.save(emb_path, ie)

    print("  Computing metrics...")
    mdf = compute_all_metrics(sample, filtered, ie, idea_meta)
    if len(mdf) == 0:
        return {"topic": "overall_mixed", "error": "metrics_failed"}
    mdf.to_parquet(overall_dir / "metrics.parquet", index=False)

    all_gh = [t for c in TOPICS.values() for t in c.get("golden_high", [])]
    all_gm = [t for c in TOPICS.values() for t in c.get("golden_medium", [])]
    result = validate_topic(mdf, all_gh, all_gm, "overall_mixed")

    print(f"\n  --- OVERALL RESULTS ---")
    for m, ev in result.get("metrics", {}).items():
        print(f"    {m:20s}: r={ev.get('sp_r', float('nan')):+.4f}")

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def generate_report(all_results):
    report = f"# Multi-Topic Novelty Detection Replication Study\n\n"
    report += f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Data**: 442K arXiv + 71K ACL papers with fulltext\n"
    report += f"**Pipeline**: fulltext chunks -> gpt-5.4-nano extraction -> text-embedding-3-small -> max-pool kNN\n\n"

    valid = [r for r in all_results if "error" not in r]
    report += f"**Topics completed**: {len(valid)}/{len(all_results)}\n\n"

    report += "## Cross-Topic Results (Spearman r vs log-citations)\n\n"
    report += "| Topic | N | N test | kNN max | kNN mean | D_M | BPI orig | Combined |\n"
    report += "|-------|---|--------|---------|----------|-----|----------|----------|\n"
    for r in all_results:
        if "error" in r:
            report += f"| {r['topic']} | ERROR | | | | | | |\n"
            continue
        m = r.get("metrics", {})
        report += (f"| {r['topic']} | {r['n_papers']} | {r['n_test']} | "
                   f"{m.get('knn_max_pool',{}).get('sp_r',float('nan')):+.3f} | "
                   f"{m.get('knn_mean_pool',{}).get('sp_r',float('nan')):+.3f} | "
                   f"{m.get('D_M',{}).get('sp_r',float('nan')):+.3f} | "
                   f"{m.get('BPI_original',{}).get('sp_r',float('nan')):+.3f} | "
                   f"{m.get('combined_best',{}).get('sp_r',float('nan')):+.3f} |\n")

    # Averages
    def avg(metric):
        vals = [r["metrics"][metric]["sp_r"] for r in valid
                if metric in r.get("metrics", {}) and not np.isnan(r["metrics"][metric].get("sp_r", float("nan")))]
        return np.mean(vals) if vals else float("nan"), np.std(vals) if vals else float("nan"), len(vals)

    report += "| **MEAN** | | | "
    for metric in ["knn_max_pool", "knn_mean_pool", "D_M", "BPI_original", "combined_best"]:
        mu, std, n = avg(metric)
        report += f"**{mu:+.3f}** | "
    report += "\n"

    # Golden set
    report += "\n## Golden Set Validation\n\n"
    report += "| Topic | Paper | Year | Cit | kNN pct |\n"
    report += "|-------|-------|------|-----|--------|\n"
    for r in valid:
        for g in r.get("golden_high_results", []):
            report += f"| {r['topic']} | {g['query'][:45]} | {g['year']} | {g['citations']:,} | {g.get('knn_pct',0):.1f}% |\n"
        for g in r.get("golden_medium_results", []):
            report += f"| {r['topic']} | {g['query'][:45]} (med) | {g['year']} | {g['citations']:,} | {g.get('knn_pct',0):.1f}% |\n"

    # Trash detection
    report += "\n## Trash Detection (Bottom 20% by kNN)\n\n"
    report += "| Topic | Flagged | Precision (<=19 cit) | Max cit |\n"
    report += "|-------|---------|---------------------|---------|\n"
    for r in valid:
        if "trash_detection" in r:
            td = r["trash_detection"]
            report += f"| {r['topic']} | {td['n_flagged']} | {td['precision_le19']:.1%} | {td['max_cit_flagged']} |\n"

    # Conclusions
    report += "\n## Conclusions\n\n"
    knn_mu, knn_std, knn_n = avg("knn_max_pool")
    dm_mu, _, _ = avg("D_M")
    bpi_mu, _, _ = avg("BPI_original")
    knn_vals = [r["metrics"]["knn_max_pool"]["sp_r"] for r in valid if "knn_max_pool" in r.get("metrics", {})]
    pos = sum(1 for v in knn_vals if v > 0)
    strong = sum(1 for v in knn_vals if v > 0.2)
    report += f"- **kNN novelty (max-pool)**: Positive in {pos}/{len(knn_vals)} topics, strong (r>0.2) in {strong}/{len(knn_vals)}\n"
    report += f"- **Mean kNN r**: {knn_mu:+.3f} (std={knn_std:.3f}) vs original embeddings-topic r=+0.487\n"
    report += f"- **Mean D_M r**: {dm_mu:+.3f} vs original r=+0.384\n"
    report += f"- **Mean BPI_original r**: {bpi_mu:+.3f} (original was r=-0.179 on embeddings topic)\n"

    return report


# ============================================================================
if __name__ == "__main__":
    t_start = time.time()

    all_results = []
    total_cost = 0.0
    for topic_name, config in TOPICS.items():
        result = run_topic(topic_name, config)
        all_results.append(result)
        # Track cost
        stats_path = EXP_DIR / topic_name / "extraction_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                s = json.load(f)
            cost = s.get("tokens_in", 0) / 1e6 * 0.10 + s.get("tokens_out", 0) / 1e6 * 0.40
            total_cost += cost

    overall = build_overall_subset(all_results)
    all_results.append(overall)

    report = generate_report(all_results)
    (EXP_DIR / "RESULTS.md").write_text(report, encoding="utf-8")

    with open(EXP_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed/60:.1f} min, ~${total_cost:.2f} API cost")
    print(f"Report: {EXP_DIR / 'RESULTS.md'}")
    print(f"{'='*70}")
    print("\n" + report)
