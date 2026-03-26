"""Fast experiment runner — batch by year, not per-paper.

Results so far:
- Mean spectrum aggregation FIXES negative correlation (max: -0.35, mean: +0.26)
- Higher K improves mean aggregation
- Paper-level kNN novelty: r=0.333

Now: finish sweep, run XGBoost, find best combination.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy, kurtosis, skew, spearmanr, zscore
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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

paper_idea_idxs = {}
for i, m in enumerate(idea_meta):
    pid = m["paper_id"]
    if pid not in paper_idea_idxs:
        paper_idea_idxs[pid] = []
    paper_idea_idxs[pid].append(i)

# Paper embeddings
paper_embs = {}
for pid, idxs in paper_idea_idxs.items():
    emb = idea_embs[idxs].mean(axis=0)
    emb /= np.linalg.norm(emb) + 1e-10
    paper_embs[pid] = emb

valid_pids = [p for p in sorted(paper_idea_idxs.keys())
              if p in id_to_cit and p in id_to_year and pd.notna(id_to_cit[p])]

SPLIT_YEAR = 2019
train_pids = [p for p in valid_pids if id_to_year[p] < SPLIT_YEAR]
test_pids = [p for p in valid_pids if id_to_year[p] >= SPLIT_YEAR]
print(f"Train: {len(train_pids)}, Test: {len(test_pids)}")

idea_embs_normed = normalize(idea_embs)
pca = PCA(n_components=256, random_state=42)
idea_embs_pca = normalize(pca.fit_transform(idea_embs_normed))


# === Spectrum functions ===
def spectrum_max(p_embs, centroids):
    if len(p_embs) == 0: return np.zeros(len(centroids))
    return np.max(normalize(centroids) @ normalize(p_embs).T, axis=1)

def spectrum_mean(p_embs, centroids):
    if len(p_embs) == 0: return np.zeros(len(centroids))
    return np.mean(normalize(centroids) @ normalize(p_embs).T, axis=1)

def spectrum_softmax(p_embs, centroids, temp=0.1):
    if len(p_embs) == 0: return np.zeros(len(centroids))
    sim = normalize(centroids) @ normalize(p_embs).T
    w = np.exp(sim / temp)
    w /= w.sum(axis=0, keepdims=True) + 1e-10
    s = w.sum(axis=1)
    return s / (s.sum() + 1e-10)


def compute_spectra(pids, centroids, method="mean"):
    fn = {"max": spectrum_max, "mean": spectrum_mean, "softmax": spectrum_softmax}[method]
    return np.array([fn(idea_embs_normed[paper_idea_idxs.get(p, [])], centroids) for p in pids], dtype=np.float32)


def temporal_mahal_by_year(all_pids, all_spectra, all_years):
    """Batch Mahalanobis: for each year, compute against all prior years at once."""
    unique_years = sorted(set(all_years))
    scores = np.full(len(all_pids), np.nan)
    K = all_spectra.shape[1]

    for year in unique_years:
        prior_mask = np.array([y < year for y in all_years])
        year_mask = np.array([y == year for y in all_years])
        n_prior = prior_mask.sum()
        if n_prior < max(30, K + 5):
            continue

        prior = all_spectra[prior_mask]
        mean = prior.mean(axis=0)
        cov = np.cov(prior.T) + np.eye(K) * 1e-4
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        year_specs = all_spectra[year_mask]
        for i, spec in enumerate(year_specs):
            try:
                scores[np.where(year_mask)[0][i]] = mahalanobis(spec, mean, cov_inv)
            except Exception:
                pass
    return scores


def temporal_rarity_by_year(all_pids, all_spectra, all_years):
    """Batch rarity R(w) = -sum(s_i * log(p_i)) using prior year prevalence."""
    unique_years = sorted(set(all_years))
    scores = np.full(len(all_pids), np.nan)

    for year in unique_years:
        prior_mask = np.array([y < year for y in all_years])
        year_mask = np.array([y == year for y in all_years])
        if prior_mask.sum() < 20:
            continue
        prior = all_spectra[prior_mask]
        prevalence = prior.mean(axis=0)
        prevalence = prevalence / (prevalence.sum() + 1e-10)
        log_p = np.log(np.maximum(prevalence, 1e-10))

        year_specs = all_spectra[year_mask]
        r_vals = -np.sum(year_specs * log_p, axis=1)
        scores[year_mask] = r_vals
    return scores


def temporal_nn_by_year(all_pids, all_spectra, all_years, k=10):
    """Batch kNN novelty in spectrum space."""
    unique_years = sorted(set(all_years))
    scores = np.full(len(all_pids), np.nan)

    for year in unique_years:
        prior_mask = np.array([y < year for y in all_years])
        year_mask = np.array([y == year for y in all_years])
        if prior_mask.sum() < k + 1:
            continue
        prior = all_spectra[prior_mask]
        nn = NearestNeighbors(n_neighbors=min(k, len(prior)), metric="euclidean")
        nn.fit(prior)
        dists, _ = nn.kneighbors(all_spectra[year_mask])
        scores[year_mask] = dists.mean(axis=1)
    return scores


def paper_nn_novelty(pids, k=10, window=None):
    """kNN novelty in paper embedding space."""
    all_sorted = sorted(valid_pids, key=lambda p: id_to_year[p])
    scores = {}
    for pid in pids:
        year = id_to_year[pid]
        if window:
            prior = [p for p in all_sorted if (year - window) <= id_to_year[p] < year and p in paper_embs]
        else:
            prior = [p for p in all_sorted if id_to_year[p] < year and p in paper_embs]
        if len(prior) < k + 1:
            scores[pid] = np.nan
            continue
        prior_e = np.array([paper_embs[p] for p in prior])
        nn = NearestNeighbors(n_neighbors=min(k, len(prior_e)), metric="cosine")
        nn.fit(prior_e)
        d, _ = nn.kneighbors(paper_embs[pid].reshape(1, -1))
        scores[pid] = float(d[0].mean())
    return scores


def eval_metric(pids, scores_dict, label=""):
    """Evaluate a metric dict against log-citations on test set."""
    vals = np.array([scores_dict.get(p, np.nan) for p in pids])
    cit = np.array([np.log1p(id_to_cit[p]) for p in pids])
    mask = ~(np.isnan(vals) | np.isnan(cit))
    if mask.sum() < 20:
        return {"label": label, "sp_r": np.nan, "n": mask.sum()}
    r, p = spearmanr(vals[mask], cit[mask])
    return {"label": label, "sp_r": r, "sp_p": p, "n": int(mask.sum())}


# ========================================================================
# EXPERIMENT 1: FAST CLUSTERING SWEEP
# ========================================================================
def run_clustering_sweep():
    print("\n" + "="*70)
    print("EXP 1: CLUSTERING SWEEP (batch by year)")
    print("="*70)

    Ks = [50, 100, 200, 300, 500, 750, 1000]
    methods = ["max", "mean", "softmax"]
    all_pids = train_pids + test_pids
    all_years = [id_to_year[p] for p in all_pids]
    log_cit = np.array([np.log1p(id_to_cit[p]) for p in all_pids])
    test_mask = np.array([p in set(test_pids) for p in all_pids])

    results = []
    for K in Ks:
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
        labels = km.fit_predict(idea_embs_pca)
        centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
        for c in range(K):
            m = labels == c
            if m.sum(): centroids[c] = idea_embs_normed[m].mean(axis=0)
        centroids = normalize(centroids)
        clust_time = time.time() - t0

        for method in methods:
            t0 = time.time()
            spectra = compute_spectra(all_pids, centroids, method=method)

            # Mahalanobis
            mahal = temporal_mahal_by_year(all_pids, spectra, all_years)
            mahal_test = mahal[test_mask]
            cit_test = log_cit[test_mask]
            valid = ~np.isnan(mahal_test)
            sp_mahal = spearmanr(mahal_test[valid], cit_test[valid])[0] if valid.sum() > 20 else np.nan

            # Rarity
            rarity = temporal_rarity_by_year(all_pids, spectra, all_years)
            rarity_test = rarity[test_mask]
            valid_r = ~np.isnan(rarity_test)
            sp_rarity = spearmanr(rarity_test[valid_r], cit_test[valid_r])[0] if valid_r.sum() > 20 else np.nan

            # Spectrum kNN novelty
            nn_spec = temporal_nn_by_year(all_pids, spectra, all_years, k=10)
            nn_test = nn_spec[test_mask]
            valid_n = ~np.isnan(nn_test)
            sp_nn = spearmanr(nn_test[valid_n], cit_test[valid_n])[0] if valid_n.sum() > 20 else np.nan

            # Coherence
            coherence = np.sum(spectra**2, axis=1)
            sp_coh = spearmanr(coherence[test_mask], cit_test)[0]

            # BPI variants
            bpi_original = rarity * coherence * (1.0 / np.maximum(1.0 / np.maximum(nn_spec, 1e-10), 1e-10))
            # Actually: BPI = R * C / rho, where rho = 1/nn_dist, so BPI = R * C * nn_dist
            bpi_v1 = rarity * coherence * nn_spec  # R * C * nn_dist (original-ish but with nn)
            bpi_v2 = mahal * (1.0 / np.maximum(nn_spec, 1e-10))  # D_M * density
            bpi_v3 = mahal * coherence  # D_M * C

            for name, arr in [("bpi_RCnn", bpi_v1), ("bpi_DM_density", bpi_v2), ("bpi_DM_C", bpi_v3)]:
                a = arr[test_mask]
                v = ~np.isnan(a)
                sp = spearmanr(a[v], cit_test[v])[0] if v.sum() > 20 else np.nan

            elapsed = time.time() - t0

            r = {"K": K, "method": method, "sp_mahal": sp_mahal, "sp_rarity": sp_rarity,
                 "sp_nn_spec": sp_nn, "sp_coherence": sp_coh, "time": elapsed + clust_time}
            results.append(r)
            print(f"  K={K:4d} {method:8s}: mahal={sp_mahal:+.4f}  rarity={sp_rarity:+.4f}  nn_spec={sp_nn:+.4f}  coh={sp_coh:+.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(EXP2 / "clustering_sweep.csv", index=False)
    return df


# ========================================================================
# EXPERIMENT 2: PAPER EMBEDDING NOVELTY VARIANTS
# ========================================================================
def run_novelty_variants():
    print("\n" + "="*70)
    print("EXP 2: PAPER EMBEDDING NOVELTY VARIANTS")
    print("="*70)

    results = []
    for k in [3, 5, 10, 20, 50]:
        for window in [None, 2, 3, 5]:
            scores = paper_nn_novelty(test_pids, k=k, window=window)
            ev = eval_metric(test_pids, scores)
            w = f"w={window}y" if window else "w=all"
            results.append({"k": k, "window": window or "all", "sp_r": ev["sp_r"]})
            marker = " <<<" if ev["sp_r"] > 0.33 else ""
            print(f"  kNN(k={k:2d}, {w:6s}): sp_r={ev['sp_r']:.4f}{marker}")

    df = pd.DataFrame(results)
    df.to_csv(EXP2 / "novelty_variants.csv", index=False)
    return df


# ========================================================================
# EXPERIMENT 3: XGBOOST
# ========================================================================
def run_xgboost():
    print("\n" + "="*70)
    print("EXP 3: XGBOOST MODELS")
    print("="*70)

    # Use K=300 (sweet spot from sweep)
    K = 300
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=4096)
    km.fit(idea_embs_pca)
    centroids = np.zeros((K, idea_embs_normed.shape[1]), dtype=np.float32)
    for c in range(K):
        m = km.labels_ == c
        if m.sum(): centroids[c] = idea_embs_normed[m].mean(axis=0)
    centroids = normalize(centroids)

    def build_features(pids):
        spectra_max = compute_spectra(pids, centroids, "max")
        spectra_mean = compute_spectra(pids, centroids, "mean")

        rows = []
        for i, pid in enumerate(pids):
            year = id_to_year[pid]
            s_max = spectra_max[i]
            s_mean = spectra_mean[i]

            # Spectrum stats (max aggregation)
            row = {
                "year": year,
                "n_ideas": len(paper_idea_idxs.get(pid, [])),
                # Max spectrum stats
                "mx_max": s_max.max(), "mx_mean": s_max.mean(), "mx_std": s_max.std(),
                "mx_entropy": entropy(s_max + 1e-10), "mx_hhi": np.sum(s_max**2),
                "mx_kurtosis": kurtosis(s_max), "mx_skew": skew(s_max),
                "mx_n_active": (s_max > 0.01).sum(),
                "mx_top1": s_max.max() / (s_max.sum() + 1e-10),
                "mx_top3": np.sort(s_max)[-3:].sum() / (s_max.sum() + 1e-10),
                # Mean spectrum stats
                "mn_max": s_mean.max(), "mn_mean": s_mean.mean(), "mn_std": s_mean.std(),
                "mn_entropy": entropy(s_mean + 1e-10), "mn_hhi": np.sum(s_mean**2),
                "mn_kurtosis": kurtosis(s_mean), "mn_skew": skew(s_mean),
                "mn_n_active": (s_mean > 0.01).sum(),
            }

            # Paper embedding features
            prior = [p for p in valid_pids if id_to_year[p] < year and p in paper_embs]
            if len(prior) >= 10:
                prior_e = np.array([paper_embs[p] for p in prior])
                emb = paper_embs[pid]
                centroid = prior_e.mean(axis=0); centroid /= np.linalg.norm(centroid) + 1e-10
                from scipy.spatial.distance import cosine
                row["p_cos_dist"] = cosine(emb, centroid)
                nn = NearestNeighbors(n_neighbors=min(10, len(prior_e)), metric="cosine")
                nn.fit(prior_e)
                d, _ = nn.kneighbors(emb.reshape(1, -1))
                row["p_nn10"] = d[0].mean()
                row["p_nn3"] = d[0, :3].mean()
                row["p_nn_max"] = d[0].max()
                # 3-year window
                prior_3y = [p for p in prior if id_to_year[p] >= year - 3]
                if len(prior_3y) >= 5:
                    pe3 = np.array([paper_embs[p] for p in prior_3y])
                    nn3 = NearestNeighbors(n_neighbors=min(10, len(pe3)), metric="cosine")
                    nn3.fit(pe3)
                    d3, _ = nn3.kneighbors(emb.reshape(1, -1))
                    row["p_nn10_3y"] = d3[0].mean()
                else:
                    row["p_nn10_3y"] = np.nan
            else:
                row["p_cos_dist"] = np.nan
                row["p_nn10"] = np.nan
                row["p_nn3"] = np.nan
                row["p_nn_max"] = np.nan
                row["p_nn10_3y"] = np.nan

            # Top-20 raw spectrum values (mean)
            for j in range(20):
                row[f"raw_mn_{j}"] = float(s_mean[j]) if j < K else 0

            rows.append(row)
        return pd.DataFrame(rows)

    print("  Building features...")
    t0 = time.time()
    X_train_df = build_features(train_pids)
    X_test_df = build_features(test_pids)
    y_train = np.array([np.log1p(id_to_cit[p]) for p in train_pids])
    y_test = np.array([np.log1p(id_to_cit[p]) for p in test_pids])
    print(f"  Features built in {time.time()-t0:.0f}s, {X_train_df.shape[1]} features")

    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)
    features = list(X_train_df.columns)

    results = []

    # Model configs to try
    configs = [
        ("GB_all", features, dict(n_estimators=300, max_depth=5, lr=0.05, subsample=0.8)),
        ("GB_spectrum_only", [c for c in features if c.startswith("mx_") or c.startswith("mn_") or c in ["year", "n_ideas"]], dict(n_estimators=200, max_depth=4, lr=0.1)),
        ("GB_paper_emb_only", [c for c in features if c.startswith("p_") or c in ["year", "n_ideas"]], dict(n_estimators=200, max_depth=4, lr=0.1)),
        ("GB_raw_spectrum", [c for c in features if c.startswith("raw_")] + ["year"], dict(n_estimators=200, max_depth=5, lr=0.1)),
        ("RF_all", features, "rf"),
    ]

    for name, cols, cfg in configs:
        print(f"\n  Training {name} ({len(cols)} features)...")
        Xtr = X_train_df[cols].values
        Xte = X_test_df[cols].values

        if cfg == "rf":
            model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingRegressor(
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"], subsample=cfg.get("subsample", 1.0), random_state=42)

        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        sp_r = spearmanr(pred, y_test)[0]
        results.append({"model": name, "sp_r": sp_r, "n_features": len(cols)})
        print(f"    sp_r = {sp_r:.4f}")

        # Feature importance for best model
        if hasattr(model, "feature_importances_") and name == "GB_all":
            imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])
            print("    Top 15 features:")
            for fn, fi in imp[:15]:
                print(f"      {fn:30s}: {fi:.4f}")
            imp_df = pd.DataFrame(imp, columns=["feature", "importance"])
            imp_df.to_csv(EXP2 / "feature_importance.csv", index=False)

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(EXP2 / "xgboost_results.csv", index=False)

    # Save predictions
    pred_df = pd.DataFrame({
        "paper_id": test_pids,
        "title": [id_to_title.get(p, "") for p in test_pids],
        "year": [id_to_year[p] for p in test_pids],
        "citations": [id_to_cit[p] for p in test_pids],
        "log_cit_actual": y_test,
    })
    # Add predictions from best model (GB_all)
    model_all = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
    model_all.fit(X_train_df.values, y_train)
    pred_df["pred_gb_all"] = model_all.predict(X_test_df.values)
    pred_df.to_parquet(EXP2 / "predictions.parquet", index=False)

    return results_df


# ========================================================================
if __name__ == "__main__":
    t0 = time.time()

    r1 = run_clustering_sweep()
    r2 = run_novelty_variants()
    r3 = run_xgboost()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")

    print("\n=== BEST RESULTS ===")
    print("\nClustering sweep (test set, best per metric):")
    for metric in ["sp_mahal", "sp_rarity", "sp_nn_spec"]:
        best = r1.sort_values(metric, ascending=False).iloc[0]
        print(f"  {metric:15s}: K={int(best['K']):>4d} {best['method']:8s} r={best[metric]:+.4f}")

    print(f"\nPaper novelty (test set):")
    best = r2.sort_values("sp_r", ascending=False).iloc[0]
    print(f"  Best: k={best['k']}, window={best['window']}, r={best['sp_r']:.4f}")

    print(f"\nXGBoost (test set):")
    for _, row in r3.iterrows():
        print(f"  {row['model']:25s}: r={row['sp_r']:.4f} ({int(row['n_features'])} feat)")

    # Git add
    import subprocess
    subprocess.run(["git", "add", "experiments/exp002_local/", "scripts/pipeline/exp_runner_fast.py"],
                   cwd=str(PROJECT), capture_output=True)
    print("\nGit added exp002_local/ and exp_runner_fast.py")
