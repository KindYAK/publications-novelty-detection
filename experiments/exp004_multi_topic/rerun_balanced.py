#!/usr/bin/env python3
"""Re-run metrics with BALANCED sampling (random, not citation-biased).

Uses cached ideas.json + idea_embeddings.npy from run_topics.py.
Only re-filters papers and re-computes metrics. No API calls.
"""
import json
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

PROJECT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data" / "processed" / "v3"

print("Loading datasets...")
ARXIV_DF = pd.read_parquet(DATA_DIR / "dataset_v3.parquet")
ACL_DF = pd.read_parquet(DATA_DIR / "acl_all.parquet")
FULLTEXT_DIR = DATA_DIR / "fulltext"
ACL_FULLTEXT_DIR = DATA_DIR / "acl_all_fulltext"


# Import topic definitions from main script
from run_topics import TOPICS, find_fulltext_arxiv, find_fulltext_acl


def filter_topic_balanced(keywords, max_papers):
    """Filter with RANDOM sampling — preserve natural citation distribution."""
    kw_lower = [k.lower() for k in keywords]

    def matches(row):
        text = str(row.get("title", "")).lower() + " " + str(row.get("abstract", "")).lower()
        return any(kw in text for kw in kw_lower)

    arxiv_sub = ARXIV_DF[ARXIV_DF.apply(matches, axis=1)].copy()
    arxiv_sub["paper_id"] = arxiv_sub["arxiv_id"]
    arxiv_sub["source"] = "arxiv"
    arxiv_sub["fulltext_path"] = arxiv_sub["arxiv_id"].apply(find_fulltext_arxiv)

    acl_sub = ACL_DF[ACL_DF.apply(matches, axis=1)].copy()
    acl_sub["paper_id"] = acl_sub["acl_id"]
    acl_sub["source"] = "acl"
    acl_sub["fulltext_path"] = acl_sub["acl_id"].apply(find_fulltext_acl)

    cols = ["paper_id", "title", "abstract", "year", "citation_count", "source", "fulltext_path"]
    combined = pd.concat([
        arxiv_sub[[c for c in cols if c in arxiv_sub.columns]],
        acl_sub[[c for c in cols if c in acl_sub.columns]],
    ], ignore_index=True)
    combined = combined.dropna(subset=["citation_count", "year"])
    combined["year"] = combined["year"].astype(int)
    combined["citation_count"] = combined["citation_count"].astype(int)
    combined = combined[combined["year"] >= 2005]
    combined = combined[combined["fulltext_path"].notna()]

    if len(combined) > max_papers:
        combined = combined.sample(n=max_papers, random_state=42)

    return combined.reset_index(drop=True)


def compute_all_metrics(df, idea_results, idea_embeddings, idea_meta):
    """Same as run_topics.py but imported for clarity."""
    paper_idea_idxs = {}
    for i, m in enumerate(idea_meta):
        pid = m["paper_id"]
        paper_idea_idxs.setdefault(pid, []).append(i)

    id_to_cit = dict(zip(df["paper_id"], df["citation_count"]))
    id_to_year = dict(zip(df["paper_id"], df["year"].astype(int)))

    paper_embs_max, paper_embs_mean = {}, {}
    for pid, idxs in paper_idea_idxs.items():
        embs = idea_embeddings[idxs]
        mx = embs.max(axis=0); mx /= np.linalg.norm(mx) + 1e-10
        paper_embs_max[pid] = mx
        mn = embs.mean(axis=0); mn /= np.linalg.norm(mn) + 1e-10
        paper_embs_mean[pid] = mn

    valid_pids = [p for p in sorted(paper_idea_idxs) if p in id_to_cit and pd.notna(id_to_cit.get(p))]
    if len(valid_pids) < 50:
        return pd.DataFrame()

    SPLIT_YEAR = 2019  # Use 2019 like original experiment
    test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]
    if len(test_pids) < 30:
        SPLIT_YEAR = int(np.median([id_to_year[p] for p in valid_pids]))
        test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]

    all_pids = valid_pids
    all_years = np.array([id_to_year[p] for p in all_pids])

    # kNN by year
    def knn_by_year(pids, pembs, k=10):
        em = np.array([pembs[p] for p in pids], dtype=np.float32)
        yrs = np.array([id_to_year[p] for p in pids])
        scores = np.full(len(pids), np.nan)
        for y in sorted(set(yrs)):
            pr, cu = yrs < y, yrs == y
            if pr.sum() < k + 1: continue
            nn = NearestNeighbors(n_neighbors=min(k, pr.sum()), metric="cosine")
            nn.fit(em[pr])
            d, _ = nn.kneighbors(em[cu])
            scores[cu] = d.mean(axis=1)
        return dict(zip(pids, scores.tolist()))

    print("    kNN...")
    knn_max = knn_by_year(all_pids, paper_embs_max, k=10)
    knn_mean = knn_by_year(all_pids, paper_embs_mean, k=10)

    # Spectrum
    print("    Spectrum...")
    ie_norm = normalize(idea_embeddings)
    K = min(200, len(ie_norm) // 10); K = max(K, 20)
    pca_d = min(256, ie_norm.shape[1], len(ie_norm) - 1)
    pca = PCA(n_components=pca_d, random_state=42)
    ie_pca = normalize(pca.fit_transform(ie_norm))
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=min(4096, len(ie_pca)))
    labels = km.fit_predict(ie_pca)
    cents = np.zeros((K, ie_norm.shape[1]), dtype=np.float32)
    for c in range(K):
        m = labels == c
        if m.sum(): cents[c] = ie_norm[m].mean(axis=0)
    cents = normalize(cents)

    all_spectra = np.array([
        np.mean(cents @ normalize(ie_norm[paper_idea_idxs.get(p, [])]).T, axis=1)
        if paper_idea_idxs.get(p) else np.zeros(K)
        for p in all_pids
    ], dtype=np.float32)

    dm_scores = np.full(len(all_pids), np.nan)
    r_scores = np.full(len(all_pids), np.nan)
    for y in sorted(set(all_years)):
        pr, cu = all_years < y, all_years == y
        if pr.sum() < max(30, K + 5): continue
        ps = all_spectra[pr]
        mean = ps.mean(axis=0)
        cov = np.cov(ps.T) + np.eye(K) * 1e-4
        try: ci = np.linalg.inv(cov)
        except: ci = np.linalg.pinv(cov)
        for i in np.where(cu)[0]:
            try: dm_scores[i] = mahalanobis(all_spectra[i], mean, ci)
            except: pass
        prev = ps.mean(axis=0); prev = prev / (prev.sum() + 1e-10)
        lp = np.log(np.maximum(prev, 1e-10))
        r_scores[cu] = -np.sum(all_spectra[cu] * lp, axis=1)

    c_scores = np.sum(all_spectra ** 2, axis=1)
    nn_spec = np.full(len(all_pids), np.nan)
    for y in sorted(set(all_years)):
        pr, cu = all_years < y, all_years == y
        if pr.sum() < 11: continue
        nn = NearestNeighbors(n_neighbors=min(10, pr.sum()), metric="euclidean")
        nn.fit(all_spectra[pr])
        d, _ = nn.kneighbors(all_spectra[cu])
        nn_spec[cu] = d.mean(axis=1)

    bpi = r_scores * c_scores * nn_spec
    knn_arr = np.array([knn_max.get(p, np.nan) for p in all_pids])
    combined = np.power(np.maximum(knn_arr, 0), 0.5) * np.power(np.maximum(dm_scores, 0), 0.5) * np.power(np.maximum(r_scores, 0), 0.1)

    mdf = pd.DataFrame({
        "paper_id": all_pids,
        "title": [df.set_index("paper_id").loc[p, "title"] if p in df["paper_id"].values else "" for p in all_pids],
        "year": [int(id_to_year[p]) for p in all_pids],
        "citation_count": [int(id_to_cit[p]) for p in all_pids],
        "knn_max_pool": list(knn_arr), "knn_mean_pool": [knn_mean.get(p, np.nan) for p in all_pids],
        "D_M": dm_scores, "R": r_scores, "C": c_scores,
        "BPI_original": bpi, "combined_best": combined,
    })

    test = mdf[mdf["year"] >= SPLIT_YEAR].dropna(subset=["knn_max_pool", "citation_count"])
    log_cit = np.log1p(test["citation_count"].values)
    ev = {}
    for col in ["knn_max_pool", "knn_mean_pool", "D_M", "R", "C", "BPI_original", "combined_best"]:
        v = test[col].values; ok = ~np.isnan(v)
        if ok.sum() < 20: ev[col] = {"sp_r": float("nan"), "n": int(ok.sum())}; continue
        r, p = spearmanr(v[ok], log_cit[ok])
        ev[col] = {"sp_r": float(r), "sp_p": float(p), "n": int(ok.sum())}

    mdf.attrs["eval"] = ev
    mdf.attrs["split_year"] = SPLIT_YEAR
    mdf.attrs["n_test"] = len(test)
    return mdf


def run_balanced_topic(topic_name, config):
    topic_dir = EXP_DIR / topic_name
    ideas_path = topic_dir / "ideas.json"
    emb_path = topic_dir / "idea_embeddings.npy"

    if not ideas_path.exists() or not emb_path.exists():
        print(f"  SKIP {topic_name} — no cached ideas/embeddings")
        return None

    print(f"\n{'='*60}")
    print(f"BALANCED: {topic_name}")
    print(f"{'='*60}")

    # Re-filter with random sampling
    df = filter_topic_balanced(config["keywords"], config["max_papers"])
    n_ft = df["fulltext_path"].notna().sum()
    print(f"  Papers: {len(df)}, fulltext: {n_ft}")
    print(f"  Cit distribution: 0-19={int((df['citation_count']<20).sum())}, "
          f"20-99={int(((df['citation_count']>=20)&(df['citation_count']<100)).sum())}, "
          f"100+={int((df['citation_count']>=100).sum())}")

    # Check golden
    for g in config.get("golden_high", []):
        mask = df["title"].str.contains(g, case=False, na=False)
        if mask.any():
            r = df[mask].iloc[0]
            print(f"  GOLDEN: {r['title'][:50]}... ({r['citation_count']} cit)")

    # Load cached ideas + embeddings
    with open(ideas_path) as f:
        idea_results = json.load(f)
    idea_embeddings = np.load(emb_path)

    # Filter ideas to papers in our balanced sample
    sample_pids = set(df["paper_id"])
    filtered_ideas = [r for r in idea_results if r["paper_id"] in sample_pids]

    idea_texts, idea_meta = [], []
    seen = set()
    for rec in filtered_ideas:
        for idea in rec["ideas"]:
            ic = idea.strip().lower()
            if ic not in seen and len(idea.strip()) > 3:
                seen.add(ic)
                idea_texts.append(idea.strip())
                idea_meta.append({"paper_id": rec["paper_id"], "text": idea.strip()})

    print(f"  Ideas in sample: {len(idea_texts)}")
    if len(idea_texts) < 50:
        print("  Too few ideas. SKIP.")
        return None

    # Re-embed only the filtered subset — but we have all embeddings cached.
    # Need to re-map: idea_meta -> indices in the full embedding array.
    # Actually simpler: just re-embed from the filtered texts.
    # But that costs money. Better: load full idea metadata and map.

    # Load FULL idea metadata to map text->index in the embedding array
    full_meta_texts = []
    full_seen = set()
    with open(ideas_path) as f:
        for rec in json.load(f):
            for idea in rec["ideas"]:
                ic = idea.strip().lower()
                if ic not in full_seen and len(idea.strip()) > 3:
                    full_seen.add(ic)
                    full_meta_texts.append(idea.strip())

    # Build text->index mapping for full embeddings
    text_to_idx = {t.strip().lower(): i for i, t in enumerate(full_meta_texts)}

    # Map filtered ideas to embedding indices
    filtered_emb_indices = []
    filtered_meta_final = []
    for m in idea_meta:
        idx = text_to_idx.get(m["text"].strip().lower())
        if idx is not None and idx < len(idea_embeddings):
            filtered_emb_indices.append(idx)
            filtered_meta_final.append(m)

    if len(filtered_emb_indices) < 50:
        print("  Too few mapped embeddings. SKIP.")
        return None

    filtered_embeddings = idea_embeddings[filtered_emb_indices]
    print(f"  Mapped embeddings: {filtered_embeddings.shape}")

    # Compute metrics
    print("  Computing metrics...")
    mdf = compute_all_metrics(df, filtered_ideas, filtered_embeddings, filtered_meta_final)
    if len(mdf) == 0:
        return None

    ev = mdf.attrs.get("eval", {})
    result = {
        "topic": topic_name + "_balanced",
        "n_papers": len(mdf), "n_test": mdf.attrs.get("n_test", 0),
        "split_year": mdf.attrs.get("split_year", 2019),
        "metrics": ev,
        "golden_high_results": [],
        "trash_detection": {},
    }

    # Golden set
    for col in ["knn_max_pool"]:
        mdf[f"{col}_pct"] = mdf[col].rank(pct=True) * 100
    for g in config.get("golden_high", []):
        mask = mdf["title"].str.contains(g, case=False, na=False)
        if mask.any():
            r = mdf[mask].iloc[0]
            result["golden_high_results"].append({
                "query": g[:60], "title": r["title"][:80],
                "year": int(r["year"]), "citations": int(r["citation_count"]),
                "knn_pct": float(r.get("knn_max_pool_pct", np.nan)),
            })

    # Trash
    sp = mdf.attrs.get("split_year", 2019)
    test = mdf[mdf["year"] >= sp].dropna(subset=["knn_max_pool", "citation_count"])
    if len(test) > 50:
        q20 = test["knn_max_pool"].quantile(0.20)
        flagged = test[test["knn_max_pool"] <= q20]
        if len(flagged) > 0:
            result["trash_detection"] = {
                "n_flagged": len(flagged),
                "precision_le19": float((flagged["citation_count"] <= 19).sum() / len(flagged)),
                "max_cit_flagged": int(flagged["citation_count"].max()),
            }

    # Print
    for m, v in ev.items():
        star = " ***" if isinstance(v.get("sp_r"), float) and v["sp_r"] > 0.2 else ""
        print(f"    {m:20s}: r={v.get('sp_r', float('nan')):+.4f} (n={v.get('n',0)}){star}")
    for g in result.get("golden_high_results", []):
        print(f"    GOLDEN: {g['query'][:40]} pct={g.get('knn_pct',0):.1f}% (cit={g['citations']})")
    if result.get("trash_detection"):
        td = result["trash_detection"]
        print(f"    TRASH: flagged={td['n_flagged']}, prec={td['precision_le19']:.1%}")

    mdf.to_parquet(topic_dir / "metrics_balanced.parquet", index=False)
    return result


if __name__ == "__main__":
    t0 = time.time()
    all_results = []
    for tn, cfg in TOPICS.items():
        r = run_balanced_topic(tn, cfg)
        if r:
            all_results.append(r)

    # Report
    print(f"\n{'='*70}")
    print("BALANCED SAMPLING CROSS-TOPIC SUMMARY")
    print(f"{'='*70}")
    print(f"| Topic | N | kNN max | kNN mean | D_M | BPI orig |")
    print(f"|-------|---|---------|----------|-----|----------|")
    for r in all_results:
        m = r["metrics"]
        print(f"| {r['topic'][:25]} | {r['n_papers']} | "
              f"{m.get('knn_max_pool',{}).get('sp_r',float('nan')):+.3f} | "
              f"{m.get('knn_mean_pool',{}).get('sp_r',float('nan')):+.3f} | "
              f"{m.get('D_M',{}).get('sp_r',float('nan')):+.3f} | "
              f"{m.get('BPI_original',{}).get('sp_r',float('nan')):+.3f} |")

    knn_vals = [r["metrics"]["knn_max_pool"]["sp_r"] for r in all_results if "knn_max_pool" in r["metrics"]]
    if knn_vals:
        print(f"\nMean kNN r: {np.mean(knn_vals):+.3f} (std={np.std(knn_vals):.3f})")

    with open(EXP_DIR / "balanced_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.0f}s")
