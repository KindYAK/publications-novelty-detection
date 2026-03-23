"""Step 4c: Paper-level novelty from mean idea embeddings + year-normalized metrics.

Key insight from 4b: year-normalized BPI has POSITIVE correlation with citations.
This script tries:
1. Paper-level embeddings (mean of idea embeddings) for direct novelty
2. Year-normalized spectrum metrics
3. Windowed prior (3-year window instead of all prior years)
4. Combined signals
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, zscore
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from config import EXP_DATA, EXP_REPORTS, GOLDEN_HIGH, GOLDEN_LOW, GOLDEN_MEDIUM, K_NEIGHBORS

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def build_paper_embeddings(idea_embs, metadata):
    """Compute mean idea embedding per paper."""
    paper_ideas = {}
    for i, meta in enumerate(metadata):
        pid = meta["paper_id"]
        if pid not in paper_ideas:
            paper_ideas[pid] = []
        paper_ideas[pid].append(i)

    paper_embs = {}
    for pid, idxs in paper_ideas.items():
        embs = idea_embs[idxs]
        mean_emb = embs.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-10
        paper_embs[pid] = mean_emb
    return paper_embs


def compute_paper_novelty(paper_embs, paper_ids, years, k_neighbors=K_NEIGHBORS, window=None):
    """Compute novelty metrics from paper-level embeddings.

    If window is set, only compare against papers from last `window` years.
    """
    emb_matrix = np.array([paper_embs[pid] for pid in paper_ids], dtype=np.float32)
    unique_years = sorted(set(years))
    results = []

    for year in unique_years:
        if window:
            prior_mask = np.array([(year - window) <= y < year for y in years])
        else:
            prior_mask = np.array([y < year for y in years])
        year_mask = np.array([y == year for y in years])

        n_prior = prior_mask.sum()
        if n_prior < 20:
            for i in np.where(year_mask)[0]:
                results.append({"idx": i, "cos_dist": np.nan, "density": np.nan, "nn_novelty": np.nan})
            continue

        prior_embs = emb_matrix[prior_mask]
        year_embs = emb_matrix[year_mask]

        # Field centroid
        centroid = prior_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-10

        # k-NN
        k = min(k_neighbors, len(prior_embs) - 1)
        nn = NearestNeighbors(n_neighbors=max(k, 1), metric="cosine")
        nn.fit(prior_embs)

        for i_rel, emb in enumerate(year_embs):
            i_abs = np.where(year_mask)[0][i_rel]

            # Cosine distance to centroid
            cos_d = float(cosine(emb, centroid))

            # Mean k-NN distance (novelty: how far from nearest prior papers)
            dists, _ = nn.kneighbors(emb.reshape(1, -1))
            nn_novelty = float(np.mean(dists[0]))  # higher = more novel
            density = 1.0 / max(nn_novelty, 1e-10)  # higher = more similar papers nearby

            results.append({"idx": i_abs, "cos_dist": cos_d, "density": density, "nn_novelty": nn_novelty})

    return results


def evaluate(df, col, label=""):
    """Quick evaluation of a metric column."""
    valid = df.dropna(subset=[col, "citation_count"])
    valid = valid[valid["citation_count"] >= 0]
    if len(valid) < 50:
        return {"col": col, "sp_r": np.nan}

    log_cit = np.log1p(valid["citation_count"].values)
    vals = valid[col].values
    sp_r, sp_p = spearmanr(vals, log_cit)

    # Golden set percentiles
    valid["_pct"] = valid[col].rank(pct=True) * 100
    high_pcts = []
    for title_frag in GOLDEN_HIGH:
        mask = valid["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            high_pcts.append(float(valid.loc[mask, "_pct"].iloc[0]))

    return {
        "col": col,
        "sp_r": sp_r,
        "sp_p": sp_p,
        "golden_mean_pct": np.mean(high_pcts) if high_pcts else np.nan,
        "golden_top25": sum(1 for p in high_pcts if p >= 75) / max(len(high_pcts), 1) * 100,
    }


def main():
    print("Loading data...")
    idea_embs = np.load(EXP_DATA / "idea_embeddings.npy")
    with open(EXP_DATA / "idea_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    subset = pd.read_parquet(EXP_DATA / "subset.parquet")

    # Load existing spectrum-based metrics
    metrics_v2 = pd.read_parquet(EXP_DATA / "metrics_v2.parquet")

    # Build paper embeddings
    print("Computing paper-level embeddings...")
    paper_embs = build_paper_embeddings(idea_embs, metadata)
    print(f"  {len(paper_embs)} papers with embeddings")

    # Get paper IDs and years aligned with metrics
    df = metrics_v2.copy()
    paper_ids = df["paper_id"].tolist()
    years = df["year"].astype(int).tolist()

    # Filter to papers that have embeddings
    has_emb = [pid in paper_embs for pid in paper_ids]
    df = df[has_emb].copy()
    paper_ids = [p for p, h in zip(paper_ids, has_emb) if h]
    years = [y for y, h in zip(years, has_emb) if h]
    print(f"  {len(df)} papers with both metrics and embeddings")

    # === Paper-level novelty (all prior) ===
    print("\nComputing paper-level novelty (all prior)...")
    paper_nov = compute_paper_novelty(paper_embs, paper_ids, years, window=None)
    for r in paper_nov:
        idx = r["idx"]
        row_mask = df.index == df.index[idx] if idx < len(df) else None
        if row_mask is not None:
            df.loc[df.index[idx], "p_cos_dist"] = r["cos_dist"]
            df.loc[df.index[idx], "p_density"] = r["density"]
            df.loc[df.index[idx], "p_nn_novelty"] = r["nn_novelty"]

    # === Paper-level novelty (3-year window) ===
    print("Computing paper-level novelty (3-year window)...")
    paper_nov_3y = compute_paper_novelty(paper_embs, paper_ids, years, window=3)
    for r in paper_nov_3y:
        idx = r["idx"]
        if idx < len(df):
            df.loc[df.index[idx], "p_cos_dist_3y"] = r["cos_dist"]
            df.loc[df.index[idx], "p_density_3y"] = r["density"]
            df.loc[df.index[idx], "p_nn_novelty_3y"] = r["nn_novelty"]

    # === Year-normalized spectrum metrics ===
    print("Computing year-normalized metrics...")
    for col in ["D_M", "R", "C", "rho", "BPI_original"]:
        if col in df.columns:
            df[f"{col}_znorm"] = df.groupby("year")[col].transform(lambda x: zscore(x, nan_policy="omit"))

    # === Composite formulas ===
    df["paper_novelty_x_density"] = df["p_cos_dist"] * df["p_density"]
    df["paper_nn_novelty_x_density"] = df["p_nn_novelty"] * df["p_density"]
    df["DM_znorm_x_rho_znorm"] = df.get("D_M_znorm", 0) * df.get("rho_znorm", 0)
    df["paper_cos_dist_3y"] = df.get("p_cos_dist_3y", np.nan)
    df["paper_nn_novelty_3y"] = df.get("p_nn_novelty_3y", np.nan)

    # Year-normalize paper metrics too
    for col in ["p_cos_dist", "p_nn_novelty", "p_density", "p_cos_dist_3y", "p_nn_novelty_3y"]:
        if col in df.columns:
            df[f"{col}_znorm"] = df.groupby("year")[col].transform(lambda x: zscore(x, nan_policy="omit"))

    # === Evaluate all columns ===
    eval_cols = [
        # Spectrum-based (raw)
        "BPI_original", "BPI_v5_DM_only", "BPI_v6_DM_C", "BPI_v8_logDM_rho",
        # Spectrum-based (year-normalized)
        "BPI_original_znorm", "D_M_znorm", "R_znorm", "C_znorm",
        # Paper-level (raw)
        "p_cos_dist", "p_nn_novelty", "p_density",
        # Paper-level (3y window)
        "p_cos_dist_3y", "p_nn_novelty_3y",
        # Paper-level (year-normalized)
        "p_cos_dist_znorm", "p_nn_novelty_znorm", "p_density_znorm",
        # Composites
        "paper_novelty_x_density", "DM_znorm_x_rho_znorm",
    ]

    print(f"\n{'='*90}")
    print(f"{'Metric':<35} {'Spearman r':>10} {'p-value':>12} {'Gold HIGH %':>12} {'Gold top25':>10}")
    print(f"{'='*90}")

    all_evals = []
    for col in eval_cols:
        if col not in df.columns:
            continue
        ev = evaluate(df, col)
        all_evals.append(ev)
        sp = ev["sp_r"]
        gp = ev.get("golden_mean_pct", np.nan)
        gt = ev.get("golden_top25", np.nan)
        marker = " <<<" if sp > 0.05 else ""
        print(f"  {col:<33} {sp:>10.4f} {ev.get('sp_p', np.nan):>12.2e} {gp:>12.1f} {gt:>10.0f}%{marker}")

    # Best positive correlation
    positive = [e for e in all_evals if e["sp_r"] > 0]
    if positive:
        best = max(positive, key=lambda e: e["sp_r"])
        print(f"\n=== Best positive: {best['col']} (r={best['sp_r']:.4f}) ===")

        # Show top 20 for best
        best_col = best["col"]
        top20 = df.nlargest(20, best_col)
        print(f"\nTop 20 by {best_col}:")
        for _, row in top20.iterrows():
            print(f"  {row[best_col]:.4f} | cit={row.get('citation_count', '?')} | {row.get('year', '?')} | {str(row.get('title', ''))[:65]}")

    # Save
    df.to_parquet(EXP_DATA / "metrics_v3.parquet", index=False)

    # Report
    report = "# Paper-Level Novelty & Year-Normalized Metrics\n\n"
    report += "## All Metrics Evaluation\n\n"
    report += f"| Metric | Spearman r | p-value | Golden HIGH % | Gold top25 |\n"
    report += f"|--------|-----------|---------|--------------|------------|\n"
    for ev in sorted(all_evals, key=lambda e: -e["sp_r"]):
        report += f"| {ev['col']} | {ev['sp_r']:.4f} | {ev.get('sp_p', np.nan):.2e} | {ev.get('golden_mean_pct', np.nan):.1f} | {ev.get('golden_top25', np.nan):.0f}% |\n"

    if positive:
        report += f"\n## Best: {best['col']} (r={best['sp_r']:.4f})\n\n"
        report += "### Top 20\n"
        report += "| Score | Citations | Year | Title |\n|-------|-----------|------|-------|\n"
        for _, row in top20.iterrows():
            report += f"| {row[best_col]:.4f} | {row.get('citation_count', '?')} | {row.get('year', '?')} | {str(row.get('title', ''))[:55]} |\n"

    (EXP_REPORTS / "04c_paper_novelty.md").write_text(report, encoding="utf-8")
    print("\nReport: reports/04c_paper_novelty.md")


if __name__ == "__main__":
    main()
