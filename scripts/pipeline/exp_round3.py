"""Round 3: ablations, golden set, NDCG, temporal decay, per-cluster, improved XGBoost.

Key question: Does idea extraction ($8) add value over free abstract embeddings?
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import entropy, kurtosis, skew, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

PROJECT = Path(__file__).resolve().parent.parent.parent
EXP = PROJECT / "experiments" / "exp001_embeddings" / "data"
EXP2 = PROJECT / "experiments" / "exp002_local"
EXP3 = PROJECT / "experiments" / "exp003_round3"
EXP3.mkdir(parents=True, exist_ok=True)

# === Load ===
print("Loading data...")
idea_embs = np.load(EXP / "idea_embeddings.npy")
with open(EXP / "idea_metadata.json", encoding="utf-8") as f:
    idea_meta = json.load(f)
subset = pd.read_parquet(EXP / "subset.parquet")

id_to_cit = dict(zip(subset["acl_id"], subset["citation_count"]))
id_to_year = dict(zip(subset["acl_id"], subset["year"].astype(int)))
id_to_title = dict(zip(subset["acl_id"], subset["title"]))
id_to_abstract = dict(zip(subset["acl_id"], subset["abstract"].fillna("")))

paper_idea_idxs = {}
for i, m in enumerate(idea_meta):
    pid = m["paper_id"]
    if pid not in paper_idea_idxs:
        paper_idea_idxs[pid] = []
    paper_idea_idxs[pid].append(i)

idea_embs_normed = normalize(idea_embs)

# Paper mean embeddings (from ideas)
paper_embs_idea = {}
for pid, idxs in paper_idea_idxs.items():
    emb = idea_embs[idxs].mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-10
    paper_embs_idea[pid] = emb

valid_pids = sorted([p for p in paper_idea_idxs if p in id_to_cit and pd.notna(id_to_cit[p]) and p in id_to_year])
SPLIT_YEAR = 2019
train_pids = [p for p in valid_pids if id_to_year[p] < SPLIT_YEAR]
test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]

# Load cached temporal metrics from round 2
with open(EXP2 / "temporal_metrics_cache.json", encoding="utf-8") as f:
    cached_metrics = json.load(f)
for pid in cached_metrics:
    for k in cached_metrics[pid]:
        if cached_metrics[pid][k] is None:
            cached_metrics[pid][k] = np.nan

print(f"Train: {len(train_pids)}, Test: {len(test_pids)}, Cached metrics: {len(cached_metrics)}")


# ========================================================================
# EXP 8: ABSTRACT-ONLY ABLATION (sentence-transformers, $0)
# ========================================================================
def run_abstract_ablation():
    print("\n" + "="*70)
    print("EXP 8: ABSTRACT-ONLY ABLATION (sentence-transformers)")
    print("="*70)

    from sentence_transformers import SentenceTransformer

    # Embed abstracts with local model
    cache_path = EXP3 / "abstract_embeddings.npy"
    pids_path = EXP3 / "abstract_pids.json"

    if cache_path.exists() and pids_path.exists():
        print("  Loading cached abstract embeddings...")
        abs_embs = np.load(cache_path)
        with open(pids_path) as f:
            abs_pids = json.load(f)
    else:
        print("  Encoding abstracts with all-MiniLM-L6-v2...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        abs_pids = [p for p in valid_pids if id_to_abstract.get(p, "").strip()]
        abstracts = [id_to_abstract[p][:2000] for p in abs_pids]
        t0 = time.time()
        abs_embs = model.encode(abstracts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        print(f"  Encoded {len(abs_pids)} abstracts in {time.time()-t0:.1f}s, shape={abs_embs.shape}")
        np.save(cache_path, abs_embs)
        with open(pids_path, "w") as f:
            json.dump(abs_pids, f)

    abs_emb_dict = {p: abs_embs[i] for i, p in enumerate(abs_pids)}
    abs_pids_set = set(abs_pids)

    # Compute kNN novelty from abstract embeddings
    test_abs = [p for p in test_pids if p in abs_pids_set]
    train_abs = [p for p in train_pids if p in abs_pids_set]
    all_abs = [p for p in valid_pids if p in abs_pids_set]
    all_sorted = sorted(all_abs, key=lambda p: id_to_year[p])

    results = []

    for k in [5, 10]:
        for window in [None, 2, 3]:
            scores = {}
            for pid in test_abs:
                year = id_to_year[pid]
                if window:
                    prior = [p for p in all_sorted if (year - window) <= id_to_year[p] < year and p in abs_emb_dict]
                else:
                    prior = [p for p in all_sorted if id_to_year[p] < year and p in abs_emb_dict]
                if len(prior) < k + 1:
                    scores[pid] = np.nan
                    continue
                pe = np.array([abs_emb_dict[p] for p in prior])
                nn = NearestNeighbors(n_neighbors=min(k, len(pe)), metric="cosine")
                nn.fit(pe)
                d, _ = nn.kneighbors(abs_emb_dict[pid].reshape(1, -1))
                scores[pid] = float(d[0].mean())

            vals = np.array([scores.get(p, np.nan) for p in test_abs])
            cit = np.array([np.log1p(id_to_cit[p]) for p in test_abs])
            mask = ~np.isnan(vals)
            r = spearmanr(vals[mask], cit[mask])[0] if mask.sum() > 20 else np.nan
            w = f"w={window}y" if window else "w=all"
            results.append({"source": "abstract", "k": k, "window": window or "all", "sp_r": r})
            print(f"  Abstract kNN(k={k}, {w}): sp_r={r:+.4f}")

    # Compare with idea-based kNN (same papers)
    print("\n  Idea-based kNN (same papers, for comparison):")
    for k in [5, 10]:
        for window in [None, 2]:
            scores = {}
            for pid in test_abs:
                year = id_to_year[pid]
                if window:
                    prior = [p for p in all_sorted if (year - window) <= id_to_year[p] < year and p in paper_embs_idea]
                else:
                    prior = [p for p in all_sorted if id_to_year[p] < year and p in paper_embs_idea]
                if len(prior) < k + 1:
                    scores[pid] = np.nan
                    continue
                pe = np.array([paper_embs_idea[p] for p in prior])
                nn = NearestNeighbors(n_neighbors=min(k, len(pe)), metric="cosine")
                nn.fit(pe)
                d, _ = nn.kneighbors(paper_embs_idea[pid].reshape(1, -1))
                scores[pid] = float(d[0].mean())

            vals = np.array([scores.get(p, np.nan) for p in test_abs])
            cit = np.array([np.log1p(id_to_cit[p]) for p in test_abs])
            mask = ~np.isnan(vals)
            r = spearmanr(vals[mask], cit[mask])[0] if mask.sum() > 20 else np.nan
            w = f"w={window}y" if window else "w=all"
            results.append({"source": "idea_mean", "k": k, "window": window or "all", "sp_r": r})
            print(f"  Idea kNN(k={k}, {w}): sp_r={r:+.4f}")

    # Abstract D_M (cluster abstracts then compute Mahalanobis)
    print("\n  Abstract-based D_M (cluster abstract embeddings):")
    for K in [100, 300, 500]:
        pca = PCA(n_components=min(128, abs_embs.shape[1]), random_state=42)
        emb_pca = normalize(pca.fit_transform(abs_embs))
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=2048)
        labels = km.fit_predict(emb_pca)
        centroids = np.zeros((K, abs_embs.shape[1]), dtype=np.float32)
        for c in range(K):
            m = labels == c
            if m.sum(): centroids[c] = abs_embs[m].mean(axis=0)
        centroids = normalize(centroids)

        # Compute mean spectra
        def abs_spectrum(pid):
            if pid not in abs_emb_dict: return np.zeros(K)
            return np.mean(normalize(centroids) @ abs_emb_dict[pid].reshape(-1, 1), axis=1).flatten()

        all_specs = {p: abs_spectrum(p) for p in all_abs}

        # Temporal D_M
        test_scores = {}
        for year in sorted(set(id_to_year[p] for p in test_abs)):
            prior = [p for p in all_abs if id_to_year[p] < year]
            year_papers = [p for p in test_abs if id_to_year[p] == year]
            if len(prior) < max(30, K + 5):
                continue
            prior_specs = np.array([all_specs[p] for p in prior])
            mean = prior_specs.mean(axis=0)
            cov = np.cov(prior_specs.T) + np.eye(K) * 1e-4
            try:
                cov_inv = np.linalg.inv(cov)
            except:
                cov_inv = np.linalg.pinv(cov)
            for pid in year_papers:
                try:
                    test_scores[pid] = mahalanobis(all_specs[pid], mean, cov_inv)
                except:
                    pass

        vals = np.array([test_scores.get(p, np.nan) for p in test_abs])
        cit = np.array([np.log1p(id_to_cit[p]) for p in test_abs])
        mask = ~np.isnan(vals)
        r = spearmanr(vals[mask], cit[mask])[0] if mask.sum() > 20 else np.nan
        results.append({"source": f"abstract_DM_K{K}", "k": K, "window": "all", "sp_r": r})
        print(f"  Abstract D_M(K={K}): sp_r={r:+.4f}")

    pd.DataFrame(results).to_csv(EXP3 / "abstract_ablation.csv", index=False)
    return results


# ========================================================================
# EXP 9: GOLDEN SET WITH K=100/200
# ========================================================================
def run_golden_small_k():
    print("\n" + "="*70)
    print("EXP 9: GOLDEN SET WITH SMALLER K")
    print("="*70)

    golden = [
        ("GloVe: Global Vectors", "high"),
        ("Linguistic Regularities in Continuous Space", "high"),
        ("Convolutional Neural Networks for Sentence Classification", "high"),
        ("Learning Phrase Representations", "high"),
        ("Learning Word Vectors for Sentiment", "high"),
        ("Contextual String Embeddings", "medium"),
        ("Universal Sentence Encoder", "medium"),
        ("word2vec Explained", "medium"),
    ]

    results = []
    for K in [50, 100, 200]:
        pca = PCA(n_components=256, random_state=42)
        emb_pca = normalize(pca.fit_transform(idea_embs_normed))
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
        km.fit(emb_pca)
        centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
        for c in range(K):
            m = km.labels_ == c
            if m.sum(): centroids[c] = idea_embs_normed[m].mean(axis=0)
        centroids = normalize(centroids)

        # Mean spectra for all
        def mean_spec(pid):
            idxs = paper_idea_idxs.get(pid, [])
            if not idxs: return np.zeros(K)
            return np.mean(normalize(centroids) @ normalize(idea_embs_normed[idxs]).T, axis=1)

        all_specs = {p: mean_spec(p) for p in valid_pids}

        # Temporal D_M for all papers
        scores = {}
        for year in sorted(set(id_to_year[p] for p in valid_pids)):
            prior = [p for p in valid_pids if id_to_year[p] < year]
            year_papers = [p for p in valid_pids if id_to_year[p] == year]
            if len(prior) < max(20, K + 5):
                continue
            prior_specs = np.array([all_specs[p] for p in prior])
            mean = prior_specs.mean(axis=0)
            cov = np.cov(prior_specs.T) + np.eye(K) * 1e-4
            try:
                cov_inv = np.linalg.inv(cov)
            except:
                cov_inv = np.linalg.pinv(cov)
            for pid in year_papers:
                try:
                    scores[pid] = mahalanobis(all_specs[pid], mean, cov_inv)
                except:
                    pass

        # Rank all scored papers
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        rank_dict = {p: i+1 for i, (p, _) in enumerate(ranked)}
        n_ranked = len(ranked)

        # Golden set
        print(f"\n  K={K} (min prior: {K+5}, scored: {n_ranked})")
        for title_frag, tier in golden:
            for pid in valid_pids:
                if title_frag.lower() in id_to_title.get(pid, "").lower():
                    r = rank_dict.get(pid)
                    pct = (1 - r/n_ranked)*100 if r else np.nan
                    cit = id_to_cit[pid]
                    yr = id_to_year[pid]
                    dm = scores.get(pid, np.nan)
                    status = f"rank={r}/{n_ranked} ({pct:.1f}%)" if r else "NOT SCORED"
                    print(f"    [{tier:>6}] yr={yr} cit={cit:>6.0f} D_M={dm:>7.2f} {status} | {id_to_title[pid][:55]}")
                    results.append({"K": K, "title": id_to_title[pid], "tier": tier, "year": yr,
                                    "citations": cit, "D_M": dm, "rank": r, "n_ranked": n_ranked,
                                    "percentile": pct})
                    break

        # Test set Spearman
        test_vals = np.array([scores.get(p, np.nan) for p in test_pids])
        test_cit = np.array([np.log1p(id_to_cit[p]) for p in test_pids])
        mask = ~np.isnan(test_vals)
        r_test = spearmanr(test_vals[mask], test_cit[mask])[0] if mask.sum() > 20 else np.nan
        print(f"  Test Spearman: {r_test:+.4f}")

    pd.DataFrame(results).to_csv(EXP3 / "golden_small_k.csv", index=False)
    return results


# ========================================================================
# EXP 10: NDCG + FUSION OPTIMIZATION
# ========================================================================
def run_ndcg_fusion():
    print("\n" + "="*70)
    print("EXP 10: NDCG + FUSION OPTIMIZATION")
    print("="*70)

    # Get metric arrays for test set
    metrics_list = ["D_M_mean", "p_knn5_2y", "p_knn10_all", "rarity_mean", "coherence"]
    test_vals = {m: np.array([cached_metrics.get(p, {}).get(m, np.nan) for p in test_pids]) for m in metrics_list}
    test_cit = np.array([id_to_cit[p] for p in test_pids])
    test_logcit = np.log1p(test_cit)

    # NDCG computation
    def compute_ndcg(scores, relevance, ks=[10, 50, 100, 500]):
        """Compute NDCG@k. Relevance = log(citations+1)."""
        mask = ~np.isnan(scores)
        s, r = scores[mask], relevance[mask]
        # Sort by score descending
        order = np.argsort(-s)
        r_sorted = r[order]
        results = {}
        for k in ks:
            if k > len(r_sorted):
                continue
            # DCG
            dcg = np.sum(r_sorted[:k] / np.log2(np.arange(2, k + 2)))
            # Ideal DCG
            ideal = np.sort(r)[::-1]
            idcg = np.sum(ideal[:k] / np.log2(np.arange(2, k + 2)))
            results[f"NDCG@{k}"] = dcg / max(idcg, 1e-10)
        return results

    print("\n  Individual metrics NDCG:")
    ndcg_results = []
    for m in metrics_list:
        ndcg = compute_ndcg(test_vals[m], test_logcit)
        r = spearmanr(np.nan_to_num(test_vals[m]), test_logcit)[0]
        print(f"    {m:20s}: Sp={r:+.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in ndcg.items()))
        ndcg_results.append({"metric": m, "spearman": r, **ndcg})

    # Rank fusion variants
    print("\n  Rank fusion NDCG:")
    combos = [
        ("D_M+kNN5", ["D_M_mean", "p_knn5_2y"]),
        ("D_M+kNN5+R", ["D_M_mean", "p_knn5_2y", "rarity_mean"]),
        ("D_M+kNN5+R+C", ["D_M_mean", "p_knn5_2y", "rarity_mean", "coherence"]),
        ("D_M+kNN10", ["D_M_mean", "p_knn10_all"]),
        ("ALL", metrics_list),
        ("D_M+kNN5+kNN10", ["D_M_mean", "p_knn5_2y", "p_knn10_all"]),
    ]
    for name, cols in combos:
        ranks = []
        for m in cols:
            v = test_vals[m].copy()
            v[np.isnan(v)] = 0
            ranks.append(pd.Series(v).rank(ascending=True, pct=True).values)
        fused = np.mean(ranks, axis=0)
        ndcg = compute_ndcg(fused, test_logcit)
        r = spearmanr(fused, test_logcit)[0]
        print(f"    {name:25s}: Sp={r:+.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in ndcg.items()))
        ndcg_results.append({"metric": f"fusion_{name}", "spearman": r, **ndcg})

    # Year-normalized rank fusion
    print("\n  Year-normalized fusion:")
    for name, cols in [("znorm_D_M+kNN5+R", ["D_M_mean", "p_knn5_2y", "rarity_mean"]),
                       ("znorm_D_M+kNN5", ["D_M_mean", "p_knn5_2y"])]:
        znorms = []
        test_years = np.array([id_to_year[p] for p in test_pids])
        for m in cols:
            v = test_vals[m].copy()
            # Z-normalize within year
            for yr in np.unique(test_years):
                mask = test_years == yr
                vals_yr = v[mask]
                valid = ~np.isnan(vals_yr)
                if valid.sum() > 5:
                    mu, std = vals_yr[valid].mean(), vals_yr[valid].std()
                    v[mask] = (vals_yr - mu) / max(std, 1e-10)
            znorms.append(np.nan_to_num(v))
        fused = np.mean(znorms, axis=0)
        ndcg = compute_ndcg(fused, test_logcit)
        r = spearmanr(fused, test_logcit)[0]
        print(f"    {name:25s}: Sp={r:+.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in ndcg.items()))
        ndcg_results.append({"metric": name, "spearman": r, **ndcg})

    pd.DataFrame(ndcg_results).to_csv(EXP3 / "ndcg_fusion.csv", index=False)
    return ndcg_results


# ========================================================================
# EXP 11: TEMPORAL DECAY + AGGREGATION VARIANTS
# ========================================================================
def run_temporal_variants():
    print("\n" + "="*70)
    print("EXP 11: TEMPORAL DECAY + AGGREGATION VARIANTS")
    print("="*70)

    all_sorted = sorted(valid_pids, key=lambda p: id_to_year[p])
    results = []

    # Max-pool paper embedding (instead of mean)
    print("  Computing max-pool paper embeddings...")
    paper_embs_max = {}
    for pid, idxs in paper_idea_idxs.items():
        if not idxs: continue
        embs = idea_embs[idxs]
        emb = embs.max(axis=0)
        emb /= np.linalg.norm(emb) + 1e-10
        paper_embs_max[pid] = emb

    # Weighted mean (weight by idea embedding norm — proxy for specificity)
    paper_embs_wnorm = {}
    for pid, idxs in paper_idea_idxs.items():
        if not idxs: continue
        embs = idea_embs[idxs]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        weighted = embs * norms  # weight by norm
        emb = weighted.mean(axis=0)
        emb /= np.linalg.norm(emb) + 1e-10
        paper_embs_wnorm[pid] = emb

    for emb_name, emb_dict in [("mean", paper_embs_idea), ("max", paper_embs_max), ("wnorm", paper_embs_wnorm)]:
        for k in [5]:
            for window in [2, None]:
                scores = {}
                for pid in test_pids:
                    if pid not in emb_dict: continue
                    year = id_to_year[pid]
                    if window:
                        prior = [p for p in all_sorted if (year - window) <= id_to_year[p] < year and p in emb_dict]
                    else:
                        prior = [p for p in all_sorted if id_to_year[p] < year and p in emb_dict]
                    if len(prior) < k + 1:
                        continue
                    pe = np.array([emb_dict[p] for p in prior])
                    nn = NearestNeighbors(n_neighbors=min(k, len(pe)), metric="cosine")
                    nn.fit(pe)
                    d, _ = nn.kneighbors(emb_dict[pid].reshape(1, -1))
                    scores[pid] = float(d[0].mean())

                vals = np.array([scores.get(p, np.nan) for p in test_pids])
                cit = np.array([np.log1p(id_to_cit[p]) for p in test_pids])
                mask = ~np.isnan(vals)
                r = spearmanr(vals[mask], cit[mask])[0] if mask.sum() > 20 else np.nan
                w = f"w={window}y" if window else "w=all"
                results.append({"aggregation": emb_name, "k": k, "window": window or "all", "sp_r": r})
                print(f"  {emb_name:6s} kNN(k={k}, {w:6s}): sp_r={r:+.4f}")

    # Exponential decay kNN
    print("\n  Exponential decay prior weighting:")
    for decay in [0.5, 1.0, 2.0]:
        scores = {}
        for pid in test_pids:
            year = id_to_year[pid]
            prior = [p for p in all_sorted if id_to_year[p] < year and p in paper_embs_idea]
            if len(prior) < 10:
                continue
            pe = np.array([paper_embs_idea[p] for p in prior])
            # Weighted distances by recency
            ages = np.array([year - id_to_year[p] for p in prior], dtype=np.float32)
            weights = np.exp(-decay * ages / ages.max())
            # Compute cosine distances
            emb = paper_embs_idea[pid]
            dists = 1 - pe @ emb  # cosine distance
            # Weighted mean of k-nearest
            k = 5
            idx = np.argsort(dists)[:k]
            scores[pid] = float(np.average(dists[idx], weights=weights[idx]))

        vals = np.array([scores.get(p, np.nan) for p in test_pids])
        cit = np.array([np.log1p(id_to_cit[p]) for p in test_pids])
        mask = ~np.isnan(vals)
        r = spearmanr(vals[mask], cit[mask])[0] if mask.sum() > 20 else np.nan
        results.append({"aggregation": f"decay_{decay}", "k": 5, "window": "all", "sp_r": r})
        print(f"  decay={decay:.1f}: sp_r={r:+.4f}")

    # Citation velocity: citations / years_since_pub
    print("\n  Citation velocity (citations/year) as target:")
    current_year = 2026
    for metric_name in ["D_M_mean", "p_knn5_2y"]:
        vals = np.array([cached_metrics.get(p, {}).get(metric_name, np.nan) for p in test_pids])
        years_since = np.array([max(current_year - id_to_year[p], 1) for p in test_pids])
        cit_velocity = np.array([id_to_cit[p] / max(current_year - id_to_year[p], 1) for p in test_pids])
        log_vel = np.log1p(cit_velocity)
        mask = ~np.isnan(vals)
        r = spearmanr(vals[mask], log_vel[mask])[0] if mask.sum() > 20 else np.nan
        results.append({"aggregation": f"{metric_name}_vs_velocity", "sp_r": r})
        print(f"  {metric_name} vs cit_velocity: sp_r={r:+.4f}")

    pd.DataFrame(results).to_csv(EXP3 / "temporal_variants.csv", index=False)
    return results


# ========================================================================
# EXP 12: PER-CLUSTER ANALYSIS
# ========================================================================
def run_cluster_analysis():
    print("\n" + "="*70)
    print("EXP 12: PER-CLUSTER FEATURE IMPORTANCE")
    print("="*70)

    K = 300
    pca = PCA(n_components=256, random_state=42)
    emb_pca = normalize(pca.fit_transform(idea_embs_normed))
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
    km.fit(emb_pca)
    centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
    for c in range(K):
        m = km.labels_ == c
        if m.sum(): centroids[c] = idea_embs_normed[m].mean(axis=0)
    centroids = normalize(centroids)

    # Load cluster descriptions from K=100 (approximate)
    with open(EXP / "cluster_descriptions.json", encoding="utf-8") as f:
        desc_100 = json.load(f)

    # Mean spectra
    spectra = {}
    for pid in valid_pids:
        idxs = paper_idea_idxs.get(pid, [])
        if not idxs:
            spectra[pid] = np.zeros(K)
        else:
            spectra[pid] = np.mean(normalize(centroids) @ normalize(idea_embs_normed[idxs]).T, axis=1)

    # Per-cluster correlation with citations
    test_spec = np.array([spectra[p] for p in test_pids])
    test_cit = np.array([np.log1p(id_to_cit[p]) for p in test_pids])

    cluster_corrs = []
    for i in range(K):
        col = test_spec[:, i]
        r, p = spearmanr(col, test_cit)
        cluster_corrs.append({"cluster": i, "sp_r": r, "sp_p": p, "mean_activation": col.mean()})

    corr_df = pd.DataFrame(cluster_corrs).sort_values("sp_r", ascending=False)

    print("\n  Top 15 clusters (POSITIVE correlation with citations):")
    for _, row in corr_df.head(15).iterrows():
        print(f"    Cluster {int(row['cluster']):>3}: r={row['sp_r']:+.4f}, mean_act={row['mean_activation']:.4f}")

    print("\n  Bottom 15 clusters (NEGATIVE correlation):")
    for _, row in corr_df.tail(15).iterrows():
        print(f"    Cluster {int(row['cluster']):>3}: r={row['sp_r']:+.4f}, mean_act={row['mean_activation']:.4f}")

    corr_df.to_csv(EXP3 / "cluster_correlations.csv", index=False)

    # Train GB with spectrum features to get feature importance
    X_train = np.array([spectra[p] for p in train_pids])
    y_train = np.array([np.log1p(id_to_cit[p]) for p in train_pids])
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(test_spec)
    r_gb = spearmanr(pred, test_cit)[0]
    print(f"\n  GB on K=300 mean spectrum: test sp_r={r_gb:+.4f}")

    imp = sorted(zip(range(K), gb.feature_importances_), key=lambda x: -x[1])
    print("  Top 10 important clusters:")
    for cl, fi in imp[:10]:
        print(f"    Cluster {cl:>3}: importance={fi:.4f}")

    return corr_df


# ========================================================================
# EXP 13: IMPROVED XGBOOST (temporal metrics + abstract PCA)
# ========================================================================
def run_improved_xgboost():
    print("\n" + "="*70)
    print("EXP 13: IMPROVED XGBOOST")
    print("="*70)

    # Features: temporal metric values + abstract PCA + year + n_ideas
    abs_path = EXP3 / "abstract_embeddings.npy"
    pids_path = EXP3 / "abstract_pids.json"
    if abs_path.exists():
        abs_embs = np.load(abs_path)
        with open(pids_path) as f:
            abs_pids = json.load(f)
        abs_dict = {p: abs_embs[i] for i, p in enumerate(abs_pids)}
        # PCA
        pca_abs = PCA(n_components=20, random_state=42)
        abs_pca = pca_abs.fit_transform(abs_embs)
        abs_pca_dict = {p: abs_pca[i] for i, p in enumerate(abs_pids)}
    else:
        abs_dict = {}
        abs_pca_dict = {}

    def build_features(pids):
        rows = []
        for pid in pids:
            row = {
                "year": id_to_year[pid],
                "n_ideas": len(paper_idea_idxs.get(pid, [])),
            }
            # Temporal metrics
            m = cached_metrics.get(pid, {})
            for k in ["D_M_mean", "p_knn5_2y", "p_knn10_all", "rarity_mean", "coherence"]:
                row[k] = m.get(k, np.nan)
            # Abstract PCA
            if pid in abs_pca_dict:
                for j in range(20):
                    row[f"abs_pca_{j}"] = float(abs_pca_dict[pid][j])
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)

    X_train = build_features(train_pids)
    X_test = build_features(test_pids)
    y_train = np.array([np.log1p(id_to_cit[p]) for p in train_pids])
    y_test = np.array([np.log1p(id_to_cit[p]) for p in test_pids])

    results = []
    configs = [
        ("GB_temporal_only", [c for c in X_train.columns if c not in ["year", "n_ideas"] and not c.startswith("abs_")] + ["year", "n_ideas"]),
        ("GB_temporal+abstract", list(X_train.columns)),
        ("GB_temporal+year", ["D_M_mean", "p_knn5_2y", "p_knn10_all", "rarity_mean", "coherence", "year"]),
        ("RF_temporal+abstract", list(X_train.columns)),
    ]

    for name, cols in configs:
        Xtr = X_train[cols].values
        Xte = X_test[cols].values
        if name.startswith("RF"):
            model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        r = spearmanr(pred, y_test)[0]
        results.append({"model": name, "sp_r": r, "n_features": len(cols)})
        print(f"  {name:30s}: sp_r={r:+.4f} ({len(cols)} features)")

        if hasattr(model, "feature_importances_") and "temporal+abstract" in name and not name.startswith("RF"):
            imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])
            print("    Top features:")
            for fn, fi in imp[:10]:
                print(f"      {fn:25s}: {fi:.4f}")

    pd.DataFrame(results).to_csv(EXP3 / "improved_xgboost.csv", index=False)
    return results


# ========================================================================
if __name__ == "__main__":
    t0 = time.time()

    run_abstract_ablation()
    run_golden_small_k()
    run_ndcg_fusion()
    run_temporal_variants()
    run_cluster_analysis()
    run_improved_xgboost()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ROUND 3 DONE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")

    import subprocess
    subprocess.run(["git", "add", "experiments/exp003_round3/", "scripts/pipeline/exp_round3.py"],
                   cwd=str(PROJECT), capture_output=True)
    print("Git added.")
