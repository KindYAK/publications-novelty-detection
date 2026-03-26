"""Experiment runner — all local compute, no API calls.

Uses existing:
- idea_embeddings.npy (62750, 1536)
- idea_metadata.json (idea → paper mapping)
- subset.parquet (5234 papers with citations/year)

Runs experiments on clustering, spectra, metrics, and supervised models.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import entropy, kurtosis, skew, spearmanr, zscore
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# === Paths ===
PROJECT = Path(__file__).resolve().parent.parent.parent
EXP = PROJECT / "experiments" / "exp001_embeddings" / "data"
REPORTS = PROJECT / "experiments" / "exp001_embeddings" / "reports"
EXP2 = PROJECT / "experiments" / "exp002_local"
EXP2.mkdir(parents=True, exist_ok=True)
(EXP2 / "reports").mkdir(exist_ok=True)

# === Load data (once) ===
print("Loading data...")
t0 = time.time()
idea_embs = np.load(EXP / "idea_embeddings.npy")
with open(EXP / "idea_metadata.json", encoding="utf-8") as f:
    idea_meta = json.load(f)
subset = pd.read_parquet(EXP / "subset.parquet")
print(f"  Loaded in {time.time()-t0:.1f}s: {idea_embs.shape[0]} ideas, {len(subset)} papers")

# Paper metadata
id_to_cit = dict(zip(subset["acl_id"], subset["citation_count"]))
id_to_year = dict(zip(subset["acl_id"], subset["year"].astype(int)))
id_to_title = dict(zip(subset["acl_id"], subset["title"]))

# Paper → idea indices
paper_idea_idxs = {}
for i, m in enumerate(idea_meta):
    pid = m["paper_id"]
    if pid not in paper_idea_idxs:
        paper_idea_idxs[pid] = []
    paper_idea_idxs[pid].append(i)

# Paper-level mean embeddings (L2-normalized)
print("Computing paper embeddings...")
paper_ids_all = sorted(paper_idea_idxs.keys())
paper_embs = {}
for pid in paper_ids_all:
    idxs = paper_idea_idxs[pid]
    emb = idea_embs[idxs].mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-10
    paper_embs[pid] = emb

# Filter to papers with citations and year
valid_pids = [p for p in paper_ids_all if p in id_to_cit and p in id_to_year
              and pd.notna(id_to_cit[p]) and pd.notna(id_to_year[p])]
print(f"  {len(valid_pids)} papers with embeddings + metadata")

# === TRAIN / TEST SPLIT (temporal: train <2019, test >=2019) ===
SPLIT_YEAR = 2019
train_pids = [p for p in valid_pids if id_to_year[p] < SPLIT_YEAR]
test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]
print(f"  Train (<{SPLIT_YEAR}): {len(train_pids)} papers")
print(f"  Test (>={SPLIT_YEAR}): {len(test_pids)} papers")

# Precompute normalized idea embeddings
idea_embs_normed = normalize(idea_embs)

# PCA (reusable)
print("PCA(256) on idea embeddings...")
pca = PCA(n_components=256, random_state=42)
idea_embs_pca = pca.fit_transform(idea_embs_normed)
idea_embs_pca = normalize(idea_embs_pca)
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def compute_spectrum_max(paper_idea_embs, centroids):
    """Original: s_i = max_j cos_sim(centroid_i, idea_j)"""
    if len(paper_idea_embs) == 0:
        return np.zeros(len(centroids))
    p = normalize(paper_idea_embs)
    c = normalize(centroids)
    sim = c @ p.T  # (K, M)
    return np.max(sim, axis=1)


def compute_spectrum_mean(paper_idea_embs, centroids):
    """Mean similarity instead of max"""
    if len(paper_idea_embs) == 0:
        return np.zeros(len(centroids))
    p = normalize(paper_idea_embs)
    c = normalize(centroids)
    sim = c @ p.T
    return np.mean(sim, axis=1)


def compute_spectrum_softmax(paper_idea_embs, centroids, temperature=0.1):
    """Soft assignment: for each idea, distribute weight across clusters by softmax similarity"""
    if len(paper_idea_embs) == 0:
        return np.zeros(len(centroids))
    p = normalize(paper_idea_embs)
    c = normalize(centroids)
    sim = c @ p.T  # (K, M)
    # Softmax over clusters for each idea
    weights = np.exp(sim / temperature)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-10
    # Sum across ideas
    spectrum = weights.sum(axis=1)
    spectrum /= spectrum.sum() + 1e-10
    return spectrum


def compute_spectrum_topk(paper_idea_embs, centroids, k=3):
    """For each idea, assign to top-k nearest clusters"""
    if len(paper_idea_embs) == 0:
        return np.zeros(len(centroids))
    p = normalize(paper_idea_embs)
    c = normalize(centroids)
    sim = c @ p.T  # (K, M)
    spectrum = np.zeros(len(centroids))
    for j in range(sim.shape[1]):
        top_k = np.argsort(sim[:, j])[::-1][:k]
        for idx in top_k:
            spectrum[idx] += sim[idx, j]
    spectrum /= spectrum.sum() + 1e-10
    return spectrum


def cluster_and_get_centroids(K, embs_pca=idea_embs_pca, embs_orig=idea_embs_normed):
    """K-Means clustering, return centroids in original space."""
    t0 = time.time()
    if len(embs_pca) > 30000:
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
    else:
        km = KMeans(n_clusters=K, n_init=3, random_state=42)
    labels = km.fit_predict(embs_pca)
    # Centroids in original space
    centroids = np.zeros((K, embs_orig.shape[1]), dtype=np.float32)
    for c in range(K):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = embs_orig[mask].mean(axis=0)
    centroids = normalize(centroids)
    elapsed = time.time() - t0
    return centroids, labels, elapsed


def compute_all_spectra(pids, centroids, method="max"):
    """Compute spectra for a list of papers."""
    func = {"max": compute_spectrum_max, "mean": compute_spectrum_mean,
            "softmax": compute_spectrum_softmax, "topk": compute_spectrum_topk}[method]
    spectra = []
    for pid in pids:
        idxs = paper_idea_idxs.get(pid, [])
        embs = idea_embs_normed[idxs] if idxs else np.zeros((0, idea_embs_normed.shape[1]))
        spectra.append(func(embs, centroids))
    return np.array(spectra, dtype=np.float32)


def spectrum_features(spectrum):
    """Extract statistical features from a spectrum vector."""
    s = spectrum
    s_nonzero = s[s > 0.01]
    return {
        "spec_max": s.max(),
        "spec_mean": s.mean(),
        "spec_std": s.std(),
        "spec_entropy": float(entropy(s + 1e-10)),
        "spec_gini": float(1 - np.sum(s**2) / (np.sum(s)**2 + 1e-10)),
        "spec_kurtosis": float(kurtosis(s)),
        "spec_skew": float(skew(s)),
        "spec_n_active": int((s > 0.01).sum()),
        "spec_top1_ratio": float(s.max() / (s.sum() + 1e-10)),
        "spec_top3_ratio": float(np.sort(s)[-3:].sum() / (s.sum() + 1e-10)),
        "spec_hhi": float(np.sum(s**2)),  # Herfindahl-Hirschman (= coherence C)
    }


def paper_embedding_features(pid, pids_prior, embs_dict, k=10):
    """Compute paper-level novelty features from embeddings."""
    emb = embs_dict[pid]
    if not pids_prior:
        return {"p_cos_dist": np.nan, "p_nn_novelty": np.nan, "p_density": np.nan}

    prior_embs = np.array([embs_dict[p] for p in pids_prior if p in embs_dict])
    if len(prior_embs) < 5:
        return {"p_cos_dist": np.nan, "p_nn_novelty": np.nan, "p_density": np.nan}

    centroid = prior_embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-10
    cos_d = float(cosine(emb, centroid))

    nn = NearestNeighbors(n_neighbors=min(k, len(prior_embs)), metric="cosine")
    nn.fit(prior_embs)
    dists, _ = nn.kneighbors(emb.reshape(1, -1))
    nn_nov = float(dists[0].mean())
    density = 1.0 / max(nn_nov, 1e-10)

    return {"p_cos_dist": cos_d, "p_nn_novelty": nn_nov, "p_density": density}


def temporal_spectrum_features(spectrum, prior_spectra):
    """Compute novelty features from spectrum relative to prior spectra."""
    if len(prior_spectra) < 10:
        return {"s_mahal": np.nan, "s_rarity": np.nan, "s_nn_dist": np.nan}

    mean = prior_spectra.mean(axis=0)
    cov = np.cov(prior_spectra.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(len(mean)) * 1e-4
    try:
        cov_inv = np.linalg.inv(cov)
        d_m = float(mahalanobis(spectrum, mean, cov_inv))
    except Exception:
        d_m = float(np.sqrt(np.sum((spectrum - mean) ** 2)))

    prevalence = mean / (mean.sum() + 1e-10)
    rarity = float(-np.sum(spectrum * np.log(np.maximum(prevalence, 1e-10))))
    coherence = float(np.sum(spectrum**2))

    nn = NearestNeighbors(n_neighbors=min(10, len(prior_spectra)), metric="euclidean")
    nn.fit(prior_spectra)
    dists, _ = nn.kneighbors(spectrum.reshape(1, -1))
    nn_dist = float(dists[0].mean())

    return {"s_mahal": d_m, "s_rarity": rarity, "s_coherence": coherence, "s_nn_dist": nn_dist}


def evaluate(y_true, y_pred, label=""):
    """Evaluate predictions: Spearman, bucket accuracy."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) < 20:
        return {"label": label, "sp_r": np.nan, "n": len(y_t)}
    sp_r, sp_p = spearmanr(y_p, y_t)
    return {"label": label, "sp_r": sp_r, "sp_p": sp_p, "n": len(y_t)}


# ========================================================================
# EXPERIMENT 1: CLUSTERING SWEEP
# ========================================================================
def run_clustering_sweep():
    print("\n" + "="*70)
    print("EXPERIMENT 1: CLUSTERING SWEEP + SPECTRUM VARIANTS")
    print("="*70)

    Ks = [50, 100, 200, 300, 500, 750, 1000, 2000]
    methods = ["max", "mean", "softmax", "topk"]
    results = []

    log_cit_test = np.array([np.log1p(id_to_cit[p]) for p in test_pids])

    for K in Ks:
        print(f"\n--- K={K} ---")
        centroids, labels, elapsed = cluster_and_get_centroids(K)
        print(f"  Clustered in {elapsed:.1f}s")

        for method in methods:
            spectra_test = compute_all_spectra(test_pids, centroids, method=method)

            # Raw spectrum correlation (mean of spectrum values vs citations)
            spec_means = spectra_test.mean(axis=1)

            # Compute temporal novelty from spectra
            spectra_train = compute_all_spectra(train_pids, centroids, method=method)

            # For test papers, compute features against train spectra
            novelty_scores = []
            for i, pid in enumerate(test_pids):
                year = id_to_year[pid]
                prior_mask = [j for j, tp in enumerate(train_pids) if id_to_year[tp] < year]
                if len(prior_mask) < 10:
                    # Also use earlier test papers
                    earlier_test = [j for j, tp in enumerate(test_pids[:i]) if id_to_year[tp] < year]
                    if earlier_test:
                        prior_specs = np.vstack([spectra_train[prior_mask]] + [spectra_test[earlier_test]]) if prior_mask else spectra_test[earlier_test]
                    elif prior_mask:
                        prior_specs = spectra_train[prior_mask]
                    else:
                        novelty_scores.append(np.nan)
                        continue
                else:
                    prior_specs = spectra_train[prior_mask]

                feats = temporal_spectrum_features(spectra_test[i], prior_specs)
                novelty_scores.append(feats["s_mahal"])

            novelty_arr = np.array(novelty_scores)

            # Evaluate
            ev_mahal = evaluate(log_cit_test, novelty_arr, f"K={K}_{method}_mahal")
            ev_rarity = evaluate(log_cit_test, np.array([
                temporal_spectrum_features(spectra_test[i],
                    spectra_train[[j for j, tp in enumerate(train_pids) if id_to_year[tp] < id_to_year[test_pids[i]]]]
                    if len([j for j, tp in enumerate(train_pids) if id_to_year[tp] < id_to_year[test_pids[i]]]) >= 10
                    else np.zeros((0, K))
                ).get("s_rarity", np.nan) for i in range(len(test_pids))
            ]), f"K={K}_{method}_rarity") if K <= 500 else {"label": "skip", "sp_r": np.nan}

            r = {"K": K, "method": method, "sp_mahal": ev_mahal["sp_r"]}
            results.append(r)
            print(f"  {method:8s}: mahal sp_r={ev_mahal['sp_r']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(EXP2 / "clustering_sweep.csv", index=False)
    print(f"\nSaved: {EXP2}/clustering_sweep.csv")
    return results_df


# ========================================================================
# EXPERIMENT 2: PAPER EMBEDDING NOVELTY VARIANTS
# ========================================================================
def run_paper_novelty_variants():
    print("\n" + "="*70)
    print("EXPERIMENT 2: PAPER EMBEDDING NOVELTY VARIANTS")
    print("="*70)

    log_cit_test = np.array([np.log1p(id_to_cit[p]) for p in test_pids])
    results = []

    # Sort all papers by year for temporal processing
    all_pids_sorted = sorted(valid_pids, key=lambda p: id_to_year[p])
    pid_to_sorted_idx = {p: i for i, p in enumerate(all_pids_sorted)}

    # Precompute all paper embeddings matrix
    all_embs = np.array([paper_embs[p] for p in all_pids_sorted])

    for k in [3, 5, 10, 20, 50]:
        for window in [None, 2, 3, 5]:
            scores = []
            for pid in test_pids:
                year = id_to_year[pid]
                if window:
                    prior = [p for p in all_pids_sorted if (year - window) <= id_to_year[p] < year and p in paper_embs]
                else:
                    prior = [p for p in all_pids_sorted if id_to_year[p] < year and p in paper_embs]

                if len(prior) < k + 1:
                    scores.append(np.nan)
                    continue

                prior_embs_arr = np.array([paper_embs[p] for p in prior])
                emb = paper_embs[pid]

                nn = NearestNeighbors(n_neighbors=min(k, len(prior_embs_arr)), metric="cosine")
                nn.fit(prior_embs_arr)
                dists, _ = nn.kneighbors(emb.reshape(1, -1))
                scores.append(float(dists[0].mean()))

            scores_arr = np.array(scores)
            ev = evaluate(log_cit_test, scores_arr)
            w_str = f"w={window}y" if window else "w=all"
            label = f"k={k}_{w_str}"
            results.append({"k": k, "window": window or "all", "sp_r": ev["sp_r"]})
            marker = " <<<" if ev["sp_r"] > 0.33 else ""
            print(f"  kNN(k={k:2d}, {w_str:6s}): sp_r={ev['sp_r']:.4f}{marker}")

    # Also try: percentile-based (what % of prior papers are further from centroid than this paper?)
    print("\n  Percentile-based novelty:")
    for window in [None, 3, 5]:
        scores = []
        for pid in test_pids:
            year = id_to_year[pid]
            if window:
                prior = [p for p in all_pids_sorted if (year - window) <= id_to_year[p] < year and p in paper_embs]
            else:
                prior = [p for p in all_pids_sorted if id_to_year[p] < year and p in paper_embs]

            if len(prior) < 20:
                scores.append(np.nan)
                continue

            prior_embs_arr = np.array([paper_embs[p] for p in prior])
            centroid = prior_embs_arr.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-10

            emb = paper_embs[pid]
            paper_dist = cosine(emb, centroid)
            prior_dists = [cosine(e, centroid) for e in prior_embs_arr]
            percentile = float(np.mean([d < paper_dist for d in prior_dists]))
            scores.append(percentile)

        scores_arr = np.array(scores)
        ev = evaluate(log_cit_test, scores_arr)
        w_str = f"w={window}y" if window else "w=all"
        results.append({"k": "pctile", "window": window or "all", "sp_r": ev["sp_r"]})
        print(f"  percentile({w_str:6s}): sp_r={ev['sp_r']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(EXP2 / "paper_novelty_variants.csv", index=False)
    print(f"\nSaved: {EXP2}/paper_novelty_variants.csv")
    return results_df


# ========================================================================
# EXPERIMENT 3: XGBOOST ON FEATURES
# ========================================================================
def run_xgboost_experiments():
    print("\n" + "="*70)
    print("EXPERIMENT 3: XGBOOST SUPERVISED MODELS")
    print("="*70)

    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    except ImportError:
        print("  sklearn ensemble not available")
        return None

    # Best clustering from sweep (use K=200 as reasonable default)
    K = 200
    print(f"  Using K={K} for spectrum features")
    centroids, labels, _ = cluster_and_get_centroids(K)

    # Build feature matrices
    def build_features(pids, centroids):
        """Build feature matrix for a set of papers."""
        spectra = compute_all_spectra(pids, centroids, method="max")
        spectra_mean = compute_all_spectra(pids, centroids, method="mean")

        rows = []
        for i, pid in enumerate(pids):
            year = id_to_year[pid]

            # Spectrum stats
            sf = spectrum_features(spectra[i])
            sf_mean = spectrum_features(spectra_mean[i])
            sf_mean = {f"mean_{k}": v for k, v in sf_mean.items()}

            # Paper embedding features (against all prior)
            prior = [p for p in valid_pids if id_to_year[p] < year and p in paper_embs]
            pf = paper_embedding_features(pid, prior, paper_embs, k=10)

            # 3-year window
            prior_3y = [p for p in prior if id_to_year[p] >= year - 3]
            pf_3y = paper_embedding_features(pid, prior_3y, paper_embs, k=10)
            pf_3y = {f"{k}_3y": v for k, v in pf_3y.items()}

            # Temporal features
            n_ideas = len(paper_idea_idxs.get(pid, []))

            row = {
                "year": year,
                "n_ideas": n_ideas,
                **sf,
                **sf_mean,
                **pf,
                **pf_3y,
            }
            # Also add raw spectrum as features (top-K principal components)
            for j in range(min(20, K)):
                row[f"spec_{j}"] = float(spectra[i, j])

            rows.append(row)

        return pd.DataFrame(rows), spectra

    print("  Building train features...")
    X_train_df, _ = build_features(train_pids, centroids)
    y_train = np.array([np.log1p(id_to_cit[p]) for p in train_pids])

    print("  Building test features...")
    X_test_df, _ = build_features(test_pids, centroids)
    y_test = np.array([np.log1p(id_to_cit[p]) for p in test_pids])

    # Drop NaN columns
    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

    X_train = X_train_df.values
    X_test = X_test_df.values
    feature_names = list(X_train_df.columns)

    results = []

    # Model 1: GradientBoosting on all features
    print("\n  Training GradientBoosting (all features)...")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    ev = evaluate(y_test, pred_gb, "GB_all")
    results.append({"model": "GradientBoosting_all", "sp_r": ev["sp_r"], "n_features": len(feature_names)})
    print(f"    Spearman r = {ev['sp_r']:.4f}")

    # Feature importance
    imp = sorted(zip(feature_names, gb.feature_importances_), key=lambda x: -x[1])
    print("    Top 15 features:")
    for name, importance in imp[:15]:
        print(f"      {name:30s}: {importance:.4f}")

    # Model 2: Only spectrum stats (no raw spectrum, no paper embedding)
    spec_cols = [c for c in feature_names if c.startswith("spec_") and not c.startswith("spec_mean")]
    stat_cols = [c for c in feature_names if not c.startswith("spec_") and c not in ["year", "n_ideas", "p_cos_dist", "p_nn_novelty", "p_density"] and "_3y" not in c and "p_" not in c]
    stat_cols += ["year", "n_ideas"]

    print("\n  Training GB (spectrum stats only)...")
    X_tr_stats = X_train_df[stat_cols].values
    X_te_stats = X_test_df[stat_cols].values
    gb2 = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gb2.fit(X_tr_stats, y_train)
    pred_gb2 = gb2.predict(X_te_stats)
    ev2 = evaluate(y_test, pred_gb2, "GB_specstats")
    results.append({"model": "GB_spectrum_stats", "sp_r": ev2["sp_r"], "n_features": len(stat_cols)})
    print(f"    Spearman r = {ev2['sp_r']:.4f}")

    # Model 3: Only paper embedding features
    emb_cols = [c for c in feature_names if "p_" in c or c in ["year", "n_ideas"]]
    print(f"\n  Training GB (paper embedding features only, {len(emb_cols)} features)...")
    X_tr_emb = X_train_df[emb_cols].values
    X_te_emb = X_test_df[emb_cols].values
    gb3 = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gb3.fit(X_tr_emb, y_train)
    pred_gb3 = gb3.predict(X_te_emb)
    ev3 = evaluate(y_test, pred_gb3, "GB_paper_emb")
    results.append({"model": "GB_paper_emb", "sp_r": ev3["sp_r"], "n_features": len(emb_cols)})
    print(f"    Spearman r = {ev3['sp_r']:.4f}")

    # Model 4: RandomForest
    print("\n  Training RandomForest (all features)...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    ev4 = evaluate(y_test, pred_rf, "RF_all")
    results.append({"model": "RandomForest_all", "sp_r": ev4["sp_r"], "n_features": len(feature_names)})
    print(f"    Spearman r = {ev4['sp_r']:.4f}")

    # Model 5: Raw spectrum only (K-dim vector → predict citations)
    print(f"\n  Training GB (raw {K}-dim spectrum only)...")
    spectra_train = compute_all_spectra(train_pids, centroids, method="max")
    spectra_test = compute_all_spectra(test_pids, centroids, method="max")
    gb5 = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb5.fit(spectra_train, y_train)
    pred_gb5 = gb5.predict(spectra_test)
    ev5 = evaluate(y_test, pred_gb5, "GB_raw_spectrum")
    results.append({"model": f"GB_raw_spectrum_K{K}", "sp_r": ev5["sp_r"], "n_features": K})
    print(f"    Spearman r = {ev5['sp_r']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(EXP2 / "xgboost_results.csv", index=False)

    # Save feature importance
    imp_df = pd.DataFrame(imp, columns=["feature", "importance"])
    imp_df.to_csv(EXP2 / "feature_importance.csv", index=False)

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        "paper_id": test_pids,
        "title": [id_to_title.get(p, "") for p in test_pids],
        "year": [id_to_year[p] for p in test_pids],
        "citations": [id_to_cit[p] for p in test_pids],
        "log_cit": y_test,
        "pred_gb_all": pred_gb,
        "pred_gb_specstats": pred_gb2,
        "pred_gb_paper_emb": pred_gb3,
        "pred_rf_all": pred_rf,
        "pred_gb_raw_spectrum": pred_gb5,
    })
    pred_df.to_parquet(EXP2 / "predictions.parquet", index=False)

    print(f"\nSaved: xgboost_results.csv, feature_importance.csv, predictions.parquet")
    return results_df, imp_df, pred_df


# ========================================================================
# MAIN
# ========================================================================
if __name__ == "__main__":
    total_t0 = time.time()

    r1 = run_clustering_sweep()
    r2 = run_paper_novelty_variants()
    r3_results, r3_imp, r3_pred = run_xgboost_experiments()

    elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS DONE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")

    # Summary
    print("\n=== SUMMARY ===")
    print("\nClustering sweep (best per method, test set):")
    if r1 is not None:
        for method in r1["method"].unique():
            best = r1[r1["method"] == method].sort_values("sp_mahal", ascending=False).iloc[0]
            print(f"  {method:8s}: K={int(best['K']):>4d}, sp_r={best['sp_mahal']:.4f}")

    print("\nPaper novelty variants (test set):")
    if r2 is not None:
        best_row = r2.sort_values("sp_r", ascending=False).iloc[0]
        print(f"  Best: k={best_row['k']}, window={best_row['window']}, sp_r={best_row['sp_r']:.4f}")

    print("\nXGBoost models (test set):")
    if r3_results is not None:
        for _, row in r3_results.iterrows():
            print(f"  {row['model']:30s}: sp_r={row['sp_r']:.4f} ({int(row['n_features'])} features)")
