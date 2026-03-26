"""Round 2 experiments: combinations, ablations, robustness, classification.

Uses existing embeddings. No API calls.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import entropy, kurtosis, skew, spearmanr, zscore
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

PROJECT = Path(__file__).resolve().parent.parent.parent
EXP = PROJECT / "experiments" / "exp001_embeddings" / "data"
EXP2 = PROJECT / "experiments" / "exp002_local"
EXP2.mkdir(parents=True, exist_ok=True)

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

# Paper mean embeddings
paper_embs = {}
for pid, idxs in paper_idea_idxs.items():
    emb = idea_embs[idxs].mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-10
    paper_embs[pid] = emb

valid_pids = sorted([p for p in paper_idea_idxs if p in id_to_cit and pd.notna(id_to_cit[p]) and p in id_to_year])
SPLIT_YEAR = 2019
train_pids = [p for p in valid_pids if id_to_year[p] < SPLIT_YEAR]
test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]
print(f"Train: {len(train_pids)}, Test: {len(test_pids)}")


# === Precompute best metrics for all papers ===
def compute_temporal_metrics(pids_to_score, all_pids, K=750):
    """Compute D_M(mean, K=750) and paper kNN(k=5, w=2y) for given papers."""
    # Clustering
    pca = PCA(n_components=256, random_state=42)
    emb_pca = normalize(pca.fit_transform(idea_embs_normed))
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
    km.fit(emb_pca)
    centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
    for c in range(K):
        m = km.labels_ == c
        if m.sum(): centroids[c] = idea_embs_normed[m].mean(axis=0)
    centroids = normalize(centroids)

    # Mean spectra for all papers
    def mean_spectrum(pid):
        idxs = paper_idea_idxs.get(pid, [])
        if not idxs: return np.zeros(K)
        return np.mean(normalize(centroids) @ normalize(idea_embs_normed[idxs]).T, axis=1)

    all_spectra = {p: mean_spectrum(p) for p in all_pids}

    results = {}
    for pid in pids_to_score:
        year = id_to_year[pid]
        spec = all_spectra[pid]

        # D_M against prior
        prior = [p for p in all_pids if id_to_year[p] < year]
        if len(prior) < max(30, K + 5):
            results[pid] = {"D_M_mean": np.nan, "p_knn5_2y": np.nan, "p_knn10_all": np.nan, "rarity_mean": np.nan, "coherence": np.nan}
            continue

        prior_specs = np.array([all_spectra[p] for p in prior])
        mean = prior_specs.mean(axis=0)
        cov = np.cov(prior_specs.T) + np.eye(K) * 1e-4
        try:
            cov_inv = np.linalg.inv(cov)
            d_m = float(mahalanobis(spec, mean, cov_inv))
        except Exception:
            d_m = float(np.sqrt(np.sum((spec - mean) ** 2)))

        prevalence = mean / (mean.sum() + 1e-10)
        rarity = float(-np.sum(spec * np.log(np.maximum(prevalence, 1e-10))))
        coherence = float(np.sum(spec**2))

        # Paper kNN (k=5, 2y window)
        prior_2y = [p for p in all_pids if (year - 2) <= id_to_year[p] < year and p in paper_embs]
        if len(prior_2y) >= 5:
            pe = np.array([paper_embs[p] for p in prior_2y])
            nn = NearestNeighbors(n_neighbors=min(5, len(pe)), metric="cosine")
            nn.fit(pe)
            d, _ = nn.kneighbors(paper_embs[pid].reshape(1, -1))
            p_knn = float(d[0].mean())
        else:
            p_knn = np.nan

        # Paper kNN all prior
        prior_all_emb = [p for p in all_pids if id_to_year[p] < year and p in paper_embs]
        if len(prior_all_emb) >= 10:
            pe_all = np.array([paper_embs[p] for p in prior_all_emb])
            nn_all = NearestNeighbors(n_neighbors=min(10, len(pe_all)), metric="cosine")
            nn_all.fit(pe_all)
            d_all, _ = nn_all.kneighbors(paper_embs[pid].reshape(1, -1))
            p_knn_all = float(d_all[0].mean())
        else:
            p_knn_all = np.nan

        results[pid] = {
            "D_M_mean": d_m, "p_knn5_2y": p_knn, "p_knn10_all": p_knn_all,
            "rarity_mean": rarity, "coherence": coherence,
        }
    return results


METRICS_CACHE = EXP2 / "temporal_metrics_cache.json"
if METRICS_CACHE.exists():
    print("Loading cached temporal metrics...")
    with open(METRICS_CACHE, encoding="utf-8") as f:
        all_metrics = json.load(f)
    # Convert string nan to float nan
    for pid in all_metrics:
        for k in all_metrics[pid]:
            if all_metrics[pid][k] is None:
                all_metrics[pid][k] = np.nan
    print(f"  Loaded {len(all_metrics)} papers from cache")
else:
    print("\n=== Computing temporal metrics for all papers ===")
    t0 = time.time()
    all_metrics = compute_temporal_metrics(valid_pids, valid_pids, K=750)
    print(f"  Done in {time.time()-t0:.0f}s")
    # Cache
    cache = {}
    for pid, vals in all_metrics.items():
        cache[pid] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in vals.items()}
    with open(METRICS_CACHE, "w") as f:
        json.dump(cache, f)
    print(f"  Cached to {METRICS_CACHE}")


# ========================================================================
# EXP 4: METRIC COMBINATION (rank fusion, weighted, XGBoost on temporal features)
# ========================================================================
def run_combinations():
    print("\n" + "="*70)
    print("EXP 4: METRIC COMBINATION")
    print("="*70)

    metrics_list = ["D_M_mean", "p_knn5_2y", "p_knn10_all", "rarity_mean", "coherence"]

    # Build arrays
    def get_arrays(pids):
        vals = {m: np.array([all_metrics[p][m] for p in pids]) for m in metrics_list}
        cit = np.array([np.log1p(id_to_cit[p]) for p in pids])
        return vals, cit

    train_vals, y_train = get_arrays(train_pids)
    test_vals, y_test = get_arrays(test_pids)

    results = []

    # Individual metrics (baseline)
    print("\n  Individual metrics (test set):")
    for m in metrics_list:
        v = test_vals[m]
        mask = ~np.isnan(v)
        r = spearmanr(v[mask], y_test[mask])[0] if mask.sum() > 20 else np.nan
        results.append({"method": f"individual_{m}", "sp_r": r})
        print(f"    {m:20s}: sp_r={r:+.4f}")

    # Rank fusion: average ranks
    print("\n  Rank fusion (test set):")
    for combo in [("D_M_mean", "p_knn5_2y"), ("D_M_mean", "p_knn10_all"),
                  ("D_M_mean", "p_knn5_2y", "rarity_mean")]:
        ranks = []
        for m in combo:
            v = test_vals[m].copy()
            v[np.isnan(v)] = 0
            ranks.append(pd.Series(v).rank(ascending=True, pct=True).values)
        avg_rank = np.mean(ranks, axis=0)
        r = spearmanr(avg_rank, y_test)[0]
        name = "+".join(combo)
        results.append({"method": f"rank_fusion_{name}", "sp_r": r})
        print(f"    {name[:50]:50s}: sp_r={r:+.4f}")

    # Weighted combination (grid search on train)
    print("\n  Weighted combination (optimized on train):")
    best_w, best_r = None, -999
    dm_train = train_vals["D_M_mean"].copy(); dm_train[np.isnan(dm_train)] = 0
    knn_train = train_vals["p_knn5_2y"].copy(); knn_train[np.isnan(knn_train)] = 0
    # Normalize to [0,1]
    dm_train_n = (dm_train - dm_train.min()) / (dm_train.max() - dm_train.min() + 1e-10)
    knn_train_n = (knn_train - knn_train.min()) / (knn_train.max() - knn_train.min() + 1e-10)

    for w in np.arange(0, 1.05, 0.05):
        combo = w * dm_train_n + (1-w) * knn_train_n
        r = spearmanr(combo, y_train)[0]
        if r > best_r:
            best_r = r
            best_w = w

    # Apply best weight to test
    dm_test = test_vals["D_M_mean"].copy(); dm_test[np.isnan(dm_test)] = 0
    knn_test = test_vals["p_knn5_2y"].copy(); knn_test[np.isnan(knn_test)] = 0
    dm_test_n = (dm_test - dm_test.min()) / (dm_test.max() - dm_test.min() + 1e-10)
    knn_test_n = (knn_test - knn_test.min()) / (knn_test.max() - knn_test.min() + 1e-10)
    combo_test = best_w * dm_test_n + (1-best_w) * knn_test_n
    r_test = spearmanr(combo_test, y_test)[0]
    results.append({"method": f"weighted_DM_kNN_w={best_w:.2f}", "sp_r": r_test})
    print(f"    Best weight: D_M*{best_w:.2f} + kNN*{1-best_w:.2f} → train r={best_r:.4f}, test r={r_test:+.4f}")

    # Reciprocal Rank Fusion (RRF)
    print("\n  Reciprocal Rank Fusion:")
    dm_ranks = pd.Series(dm_test).rank(ascending=True).values
    knn_ranks = pd.Series(knn_test).rank(ascending=True).values
    rrf = 1/(60 + dm_ranks) + 1/(60 + knn_ranks)
    r_rrf = spearmanr(rrf, y_test)[0]
    results.append({"method": "RRF_DM_kNN", "sp_r": r_rrf})
    print(f"    RRF(D_M, kNN): sp_r={r_rrf:+.4f}")

    # XGBoost on temporal metrics as features
    print("\n  XGBoost on temporal metric features:")
    X_train = np.column_stack([train_vals[m] for m in metrics_list])
    X_test = np.column_stack([test_vals[m] for m in metrics_list])
    X_train = np.nan_to_num(X_train, 0)
    X_test = np.nan_to_num(X_test, 0)
    # Add year
    X_train = np.column_stack([X_train, [id_to_year[p] for p in train_pids]])
    X_test = np.column_stack([X_test, [id_to_year[p] for p in test_pids]])

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    r_gb = spearmanr(pred, y_test)[0]
    results.append({"method": "GB_temporal_features", "sp_r": r_gb})
    print(f"    GB(temporal features): sp_r={r_gb:+.4f}")
    imp = sorted(zip(metrics_list + ["year"], gb.feature_importances_), key=lambda x: -x[1])
    for name, val in imp:
        print(f"      {name:20s}: {val:.4f}")

    return pd.DataFrame(results)


# ========================================================================
# EXP 5: ROBUSTNESS (different splits, bootstrap)
# ========================================================================
def run_robustness():
    print("\n" + "="*70)
    print("EXP 5: ROBUSTNESS")
    print("="*70)

    results = []

    # Different split years
    print("\n  Different split years:")
    for split in [2017, 2018, 2019, 2020, 2021]:
        tr = [p for p in valid_pids if id_to_year[p] < split]
        te = [p for p in valid_pids if id_to_year[p] >= split]
        if len(te) < 100 or len(tr) < 100:
            continue

        # D_M(mean)
        dm_te = np.array([all_metrics[p]["D_M_mean"] for p in te])
        knn_te = np.array([all_metrics[p]["p_knn5_2y"] for p in te])
        cit_te = np.array([np.log1p(id_to_cit[p]) for p in te])

        mask_dm = ~np.isnan(dm_te)
        mask_knn = ~np.isnan(knn_te)
        r_dm = spearmanr(dm_te[mask_dm], cit_te[mask_dm])[0] if mask_dm.sum() > 20 else np.nan
        r_knn = spearmanr(knn_te[mask_knn], cit_te[mask_knn])[0] if mask_knn.sum() > 20 else np.nan

        results.append({"split": split, "n_train": len(tr), "n_test": len(te),
                        "r_DM_mean": r_dm, "r_kNN_2y": r_knn})
        print(f"    split={split}: train={len(tr):,} test={len(te):,} | D_M={r_dm:+.4f} kNN={r_knn:+.4f}")

    # Bootstrap CI (on best split)
    print("\n  Bootstrap 95% CI (split=2019):")
    te = [p for p in valid_pids if id_to_year[p] >= 2019]
    dm_te = np.array([all_metrics[p]["D_M_mean"] for p in te])
    knn_te = np.array([all_metrics[p]["p_knn5_2y"] for p in te])
    cit_te = np.array([np.log1p(id_to_cit[p]) for p in te])

    n_boot = 1000
    for name, vals in [("D_M_mean", dm_te), ("kNN_2y", knn_te)]:
        mask = ~np.isnan(vals)
        v, c = vals[mask], cit_te[mask]
        boot_r = []
        for _ in range(n_boot):
            idx = np.random.choice(len(v), len(v), replace=True)
            boot_r.append(spearmanr(v[idx], c[idx])[0])
        boot_r = np.array(boot_r)
        lo, hi = np.percentile(boot_r, [2.5, 97.5])
        results.append({"metric": name, "mean_r": np.mean(boot_r), "ci_lo": lo, "ci_hi": hi})
        print(f"    {name:15s}: r={np.mean(boot_r):.4f} [{lo:.4f}, {hi:.4f}]")

    return pd.DataFrame(results)


# ========================================================================
# EXP 6: BREAKTHROUGH CLASSIFICATION
# ========================================================================
def run_classification():
    print("\n" + "="*70)
    print("EXP 6: BREAKTHROUGH CLASSIFICATION")
    print("="*70)

    results = []

    for threshold_name, threshold in [("100+ citations", 100), ("500+ citations", 500), ("50+ citations", 50)]:
        print(f"\n  --- {threshold_name} ---")

        # Labels
        y_train = np.array([1 if id_to_cit[p] >= threshold else 0 for p in train_pids])
        y_test = np.array([1 if id_to_cit[p] >= threshold else 0 for p in test_pids])
        print(f"  Train: {y_train.sum()} positive / {len(y_train)} total ({y_train.mean()*100:.1f}%)")
        print(f"  Test:  {y_test.sum()} positive / {len(y_test)} total ({y_test.mean()*100:.1f}%)")

        if y_test.sum() < 5 or y_train.sum() < 5:
            print("  Too few positives, skipping")
            continue

        # Features: temporal metrics
        metrics_list = ["D_M_mean", "p_knn5_2y", "p_knn10_all", "rarity_mean", "coherence"]
        X_train = np.column_stack([
            np.array([all_metrics[p].get(m, 0) for p in train_pids]) for m in metrics_list
        ] + [np.array([id_to_year[p] for p in train_pids])])
        X_test = np.column_stack([
            np.array([all_metrics[p].get(m, 0) for p in test_pids]) for m in metrics_list
        ] + [np.array([id_to_year[p] for p in test_pids])])
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)

        # GradientBoosting
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        pred_proba = gb.predict_proba(X_test)[:, 1]
        pred = gb.predict(X_test)

        auc = roc_auc_score(y_test, pred_proba) if len(set(y_test)) > 1 else np.nan
        ap = average_precision_score(y_test, pred_proba) if len(set(y_test)) > 1 else np.nan
        f1 = f1_score(y_test, pred, zero_division=0)

        results.append({"task": threshold_name, "model": "GB_temporal",
                        "AUC": auc, "AP": ap, "F1": f1,
                        "precision": (pred & y_test).sum() / max(pred.sum(), 1),
                        "recall": (pred & y_test).sum() / max(y_test.sum(), 1)})

        print(f"  GB: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}")

        # Simple threshold on best metric (D_M_mean)
        dm_test = np.array([all_metrics[p]["D_M_mean"] for p in test_pids])
        dm_test = np.nan_to_num(dm_test, 0)
        # Find best threshold on train
        dm_train = np.array([all_metrics[p]["D_M_mean"] for p in train_pids])
        dm_train = np.nan_to_num(dm_train, 0)
        best_f1, best_t = 0, 0
        for pct in range(50, 100):
            t = np.percentile(dm_train, pct)
            pred_simple = (dm_train >= t).astype(int)
            f1_t = f1_score(y_train, pred_simple, zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        pred_simple_test = (dm_test >= best_t).astype(int)
        f1_simple = f1_score(y_test, pred_simple_test, zero_division=0)
        prec_simple = (pred_simple_test & y_test).sum() / max(pred_simple_test.sum(), 1)
        rec_simple = (pred_simple_test & y_test).sum() / max(y_test.sum(), 1)
        results.append({"task": threshold_name, "model": "threshold_DM",
                        "F1": f1_simple, "precision": prec_simple, "recall": rec_simple})
        print(f"  D_M threshold: F1={f1_simple:.3f}, prec={prec_simple:.3f}, rec={rec_simple:.3f}")

        # RandomForest
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, class_weight="balanced")
        rf.fit(X_train, y_train)
        pred_rf = rf.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, pred_rf) if len(set(y_test)) > 1 else np.nan
        ap_rf = average_precision_score(y_test, pred_rf) if len(set(y_test)) > 1 else np.nan
        results.append({"task": threshold_name, "model": "RF_balanced",
                        "AUC": auc_rf, "AP": ap_rf})
        print(f"  RF(balanced): AUC={auc_rf:.3f}, AP={ap_rf:.3f}")

    return pd.DataFrame(results)


# ========================================================================
# EXP 7: GOLDEN SET + ERROR ANALYSIS
# ========================================================================
def run_error_analysis():
    print("\n" + "="*70)
    print("EXP 7: GOLDEN SET + ERROR ANALYSIS")
    print("="*70)

    # Build rankings
    dm_scores = {p: all_metrics[p]["D_M_mean"] for p in valid_pids if not np.isnan(all_metrics[p]["D_M_mean"])}
    knn_scores = {p: all_metrics[p]["p_knn5_2y"] for p in valid_pids if not np.isnan(all_metrics[p]["p_knn5_2y"])}

    dm_ranked = sorted(dm_scores.items(), key=lambda x: -x[1])
    knn_ranked = sorted(knn_scores.items(), key=lambda x: -x[1])

    dm_rank = {p: i+1 for i, (p, _) in enumerate(dm_ranked)}
    knn_rank = {p: i+1 for i, (p, _) in enumerate(knn_ranked)}
    n_dm = len(dm_ranked)
    n_knn = len(knn_ranked)

    # Golden set
    golden = [
        ("GloVe: Global Vectors", "high"),
        ("Linguistic Regularities in Continuous Space", "high"),
        ("Convolutional Neural Networks for Sentence Classification", "high"),
        ("Learning Phrase Representations", "high"),
        ("Learning Word Vectors for Sentiment", "high"),
        ("Contextual String Embeddings", "medium"),
        ("Universal Sentence Encoder", "medium"),
        ("Dependency-Based Word Embeddings", "medium"),
        ("Named Entity Recognition with Bidirectional", "low"),
        ("Relation Classification via Convolutional Deep Neural", "low"),
    ]

    print("\n  === Golden Set Rankings ===")
    print(f"  {'Title':<55} {'Tier':>6} {'Cit':>6} {'D_M rank':>10} {'D_M %':>8} {'kNN rank':>10} {'kNN %':>8}")
    golden_results = []
    for title_frag, tier in golden:
        for pid in valid_pids:
            if title_frag.lower() in id_to_title.get(pid, "").lower():
                cit = id_to_cit[pid]
                dr = dm_rank.get(pid, None)
                kr = knn_rank.get(pid, None)
                dp = (1 - dr/n_dm)*100 if dr else np.nan
                kp = (1 - kr/n_knn)*100 if kr else np.nan
                print(f"  {id_to_title[pid][:55]:<55} {tier:>6} {cit:>6.0f} {dr or 'N/A':>10} {dp:>7.1f}% {kr or 'N/A':>10} {kp:>7.1f}%")
                golden_results.append({
                    "title": id_to_title[pid], "tier": tier, "citations": cit,
                    "DM_rank": dr, "DM_pct": dp, "kNN_rank": kr, "kNN_pct": kp,
                    "DM_score": dm_scores.get(pid), "kNN_score": knn_scores.get(pid),
                })
                break

    # Top 30 by D_M that are NOT highly cited (false positives?)
    print("\n  === Top 30 D_M (potential false positives) ===")
    print(f"  {'Rank':>4} {'D_M':>8} {'Cit':>6} {'Year':>5} {'Title'}")
    fp_rows = []
    for i, (pid, score) in enumerate(dm_ranked[:30]):
        cit = id_to_cit[pid]
        year = id_to_year[pid]
        is_fp = "FP?" if cit < 20 else ""
        print(f"  {i+1:>4} {score:>8.2f} {cit:>6.0f} {year:>5} {id_to_title[pid][:70]} {is_fp}")
        fp_rows.append({"rank": i+1, "D_M": score, "citations": cit, "year": year,
                        "title": id_to_title[pid], "false_positive": cit < 20})

    # Bottom of high-cited (false negatives: high cit but low D_M)
    print("\n  === Highly cited but LOW D_M (false negatives) ===")
    high_cit = [(p, id_to_cit[p]) for p in valid_pids if id_to_cit[p] >= 200 and p in dm_rank]
    high_cit_by_dm = sorted(high_cit, key=lambda x: dm_rank.get(x[0], 99999), reverse=True)
    fn_rows = []
    for pid, cit in high_cit_by_dm[:15]:
        dr = dm_rank[pid]
        dp = (1 - dr/n_dm)*100
        print(f"  D_M rank={dr:>5} ({dp:>5.1f}%) | cit={cit:>6.0f} | {id_to_title[pid][:65]}")
        fn_rows.append({"DM_rank": dr, "DM_pct": dp, "citations": cit, "title": id_to_title[pid]})

    return pd.DataFrame(golden_results), pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)


# ========================================================================
if __name__ == "__main__":
    t0 = time.time()

    r4 = run_combinations()
    r5 = run_robustness()
    r6 = run_classification()
    golden_df, fp_df, fn_df = run_error_analysis()

    # Save all results
    r4.to_csv(EXP2 / "combinations.csv", index=False)
    r5.to_csv(EXP2 / "robustness.csv", index=False)
    r6.to_csv(EXP2 / "classification.csv", index=False)
    golden_df.to_csv(EXP2 / "golden_set_analysis.csv", index=False)
    fp_df.to_csv(EXP2 / "false_positives.csv", index=False)
    fn_df.to_csv(EXP2 / "false_negatives.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ROUND 2 DONE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")

    # Git add
    import subprocess
    subprocess.run(["git", "add", "experiments/exp002_local/", "scripts/pipeline/exp_round2.py"],
                   cwd=str(PROJECT), capture_output=True)
    print("Git added.")
