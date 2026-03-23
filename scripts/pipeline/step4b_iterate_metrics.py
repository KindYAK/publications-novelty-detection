"""Step 4b: Iterate on metric formulations using existing spectra.

Problem: BPI = R * C * (1/rho) rewards OBSCURE papers, not breakthroughs.
- R (rarity) and C (coherence) are highest for niche papers
- 1/rho rewards isolation (no similar papers)

Hypotheses to test:
1. Use rho (density) instead of 1/rho — novel + popular = impactful
2. Try D_M-based formulas — structural distance captures unusual combinations
3. Year-normalized z-scores to handle temporal drift
4. Hybrid: novelty * density (papers that are different from crowd in crowded fields)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import kendalltau, mannwhitneyu, spearmanr, zscore
from sklearn.neighbors import NearestNeighbors

from config import COV_REG, EXP_DATA, EXP_REPORTS, GOLDEN_HIGH, GOLDEN_LOW, GOLDEN_MEDIUM, K_NEIGHBORS, MIN_PRIOR_PAPERS

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def compute_all_metrics_temporal(spectra, years, k_neighbors=K_NEIGHBORS, min_prior=MIN_PRIOR_PAPERS):
    """Compute raw metrics with temporal ordering, return per-paper dict."""
    unique_years = sorted(set(years))
    results = []  # list of dicts with all raw metrics

    for year in unique_years:
        prior_mask = np.array([y < year for y in years])
        year_mask = np.array([y == year for y in years])
        n_prior = prior_mask.sum()

        if n_prior < min_prior:
            for i in np.where(year_mask)[0]:
                results.append({"idx": i, "year": year, "D_M": np.nan, "R": np.nan, "C": np.nan, "rho": np.nan})
            continue

        prior_specs = spectra[prior_mask]
        year_specs = spectra[year_mask]

        mean = prior_specs.mean(axis=0)
        cov = np.cov(prior_specs.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(len(mean)) * COV_REG
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        prevalence = mean / (mean.sum() + 1e-10)
        log_p = np.log(np.maximum(prevalence, 1e-10))

        k = min(k_neighbors, len(prior_specs) - 1)
        nn = NearestNeighbors(n_neighbors=max(k, 1), metric="euclidean")
        nn.fit(prior_specs)

        for i_rel, spec in enumerate(year_specs):
            i_abs = np.where(year_mask)[0][i_rel]

            try:
                d_m = float(mahalanobis(spec, mean, cov_inv))
            except Exception:
                d_m = float(np.sqrt(np.sum((spec - mean) ** 2)))

            r = float(-np.sum(spec * log_p))
            c = float(np.sum(spec**2))

            dists, _ = nn.kneighbors(spec.reshape(1, -1))
            mean_dist = np.mean(dists[0])
            rho = 1.0 / max(mean_dist, 1e-10)

            results.append({"idx": i_abs, "year": year, "D_M": d_m, "R": r, "C": c, "rho": rho})

    return results


# === BPI formulas to try ===
FORMULAS = {
    "BPI_original": lambda m: m["R"] * m["C"] * (1.0 / max(m["rho"], 1e-10)),  # Original: R*C/rho
    "BPI_v2_DM_rho": lambda m: m["D_M"] * m["rho"],  # Novelty * density
    "BPI_v3_R_rho": lambda m: m["R"] * m["rho"],  # Rarity * density
    "BPI_v4_DM_C_rho": lambda m: m["D_M"] * m["C"] * m["rho"],  # Novelty * coherence * density
    "BPI_v5_DM_only": lambda m: m["D_M"],  # Pure structural novelty
    "BPI_v6_DM_C": lambda m: m["D_M"] * m["C"],  # Novelty * coherence (no density)
    "BPI_v7_R_div_rho_inv": lambda m: m["R"] / max(1.0 / max(m["rho"], 1e-10), 1e-10),  # R * rho (= R / (1/rho))
    "BPI_v8_logDM_rho": lambda m: np.log1p(m["D_M"]) * m["rho"],  # log-novelty * density
}


def evaluate_formula(df, formula_col, golden_high_titles, golden_low_titles):
    """Evaluate a BPI formula against golden set and citations."""
    valid = df.dropna(subset=[formula_col]).copy()
    if len(valid) == 0:
        return {}

    valid["_rank"] = valid[formula_col].rank(ascending=False, method="min").astype(int)
    valid["_pct"] = valid[formula_col].rank(pct=True) * 100

    # Golden set
    high_pcts = []
    for title_frag in golden_high_titles:
        mask = valid["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            high_pcts.append(float(valid.loc[mask, "_pct"].iloc[0]))

    low_pcts = []
    for title_frag in golden_low_titles:
        mask = valid["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            low_pcts.append(float(valid.loc[mask, "_pct"].iloc[0]))

    # Citation correlation
    cit_valid = valid.dropna(subset=["citation_count"])
    cit_valid = cit_valid[cit_valid["citation_count"] >= 0]
    log_cit = np.log1p(cit_valid["citation_count"].values)
    vals = cit_valid[formula_col].values

    sp_r, sp_p = spearmanr(vals, log_cit) if len(vals) > 10 else (np.nan, np.nan)

    # Mann-Whitney: top 25% vs bottom 25%
    q75 = np.percentile(vals, 75)
    q25 = np.percentile(vals, 25)
    high_bpi_cit = cit_valid[cit_valid[formula_col] >= q75]["citation_count"]
    low_bpi_cit = cit_valid[cit_valid[formula_col] <= q25]["citation_count"]
    mw_p = np.nan
    median_high, median_low = np.nan, np.nan
    if len(high_bpi_cit) > 5 and len(low_bpi_cit) > 5:
        _, mw_p = mannwhitneyu(high_bpi_cit, low_bpi_cit, alternative="greater")
        median_high = float(high_bpi_cit.median())
        median_low = float(low_bpi_cit.median())

    # Year-normalized version
    valid["_znorm"] = valid.groupby("year")[formula_col].transform(lambda x: zscore(x, nan_policy="omit"))
    sp_z, _ = spearmanr(valid.dropna(subset=["_znorm", "citation_count"])["_znorm"],
                         np.log1p(valid.dropna(subset=["_znorm", "citation_count"])["citation_count"])) if len(valid) > 10 else (np.nan, np.nan)

    return {
        "formula": formula_col,
        "spearman_r": sp_r,
        "spearman_p": sp_p,
        "spearman_znorm": sp_z,
        "golden_high_mean_pct": np.mean(high_pcts) if high_pcts else np.nan,
        "golden_high_in_top25": sum(1 for p in high_pcts if p >= 75) / max(len(high_pcts), 1) * 100,
        "golden_low_mean_pct": np.mean(low_pcts) if low_pcts else np.nan,
        "mw_p": mw_p,
        "median_cit_high_bpi": median_high,
        "median_cit_low_bpi": median_low,
    }


def main():
    print("Loading data...")
    spectra = np.load(EXP_DATA / "spectra.npy")
    with open(EXP_DATA / "spectra_paper_ids.json") as f:
        paper_ids = json.load(f)
    subset = pd.read_parquet(EXP_DATA / "subset.parquet")

    # Build year mapping
    id_to_row = {row["acl_id"]: row for _, row in subset.iterrows()}
    years = [int(id_to_row[pid]["year"]) if pid in id_to_row and pd.notna(id_to_row[pid].get("year")) else 2020 for pid in paper_ids]

    print(f"Computing raw metrics for {len(spectra)} papers...")
    raw_metrics = compute_all_metrics_temporal(spectra, years)

    # Build dataframe
    rows = []
    for m in raw_metrics:
        pid = paper_ids[m["idx"]]
        meta = id_to_row.get(pid, {})
        row = {
            "paper_id": pid,
            "year": m["year"],
            "D_M": m["D_M"],
            "R": m["R"],
            "C": m["C"],
            "rho": m["rho"],
            "title": meta.get("title", ""),
            "citation_count": meta.get("citation_count"),
        }
        # Apply all BPI formulas
        for fname, func in FORMULAS.items():
            try:
                row[fname] = func(m) if not np.isnan(m["D_M"]) else np.nan
            except Exception:
                row[fname] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Metrics computed for {len(df)} papers")

    # === Evaluate each formula ===
    print("\n=== Formula Comparison ===")
    print(f"{'Formula':<25} {'Spearman':>10} {'Sp(znorm)':>10} {'GoldH%':>8} {'GoldH top25%':>12} {'GoldL%':>8} {'MedCitHi':>10} {'MedCitLo':>10}")
    print("-" * 105)

    eval_results = []
    for fname in FORMULAS:
        ev = evaluate_formula(df, fname, GOLDEN_HIGH, GOLDEN_LOW)
        eval_results.append(ev)
        print(
            f"{fname:<25} {ev['spearman_r']:>10.3f} {ev['spearman_znorm']:>10.3f} "
            f"{ev['golden_high_mean_pct']:>8.1f} {ev['golden_high_in_top25']:>12.0f}% "
            f"{ev['golden_low_mean_pct']:>8.1f} {ev['median_cit_high_bpi']:>10.0f} {ev['median_cit_low_bpi']:>10.0f}"
        )

    # === Pick best formula (highest Spearman with citations) ===
    best = max(eval_results, key=lambda e: e["spearman_r"] if not np.isnan(e["spearman_r"]) else -999)
    print(f"\n=== Best formula: {best['formula']} (Spearman r={best['spearman_r']:.3f}) ===")

    # Show top 20 for best formula
    best_col = best["formula"]
    top20 = df.nlargest(20, best_col)
    print(f"\nTop 20 by {best_col}:")
    for _, row in top20.iterrows():
        print(f"  {best_col}={row[best_col]:.4f} | cit={row.get('citation_count', '?')} | {str(row.get('title', ''))[:70]}")

    # === Save best metrics ===
    df.to_parquet(EXP_DATA / "metrics_v2.parquet", index=False)

    # Report
    report = f"# Metric Iteration Report\n\n"
    report += f"## Formula Comparison\n\n"
    report += f"| Formula | Spearman r | Sp(z-norm) | Golden HIGH % | HIGH in top25 | Golden LOW % | Med Cit (high BPI) | Med Cit (low BPI) |\n"
    report += f"|---------|-----------|-----------|--------------|--------------|-------------|-------------------|-------------------|\n"
    for ev in eval_results:
        report += f"| {ev['formula']} | {ev['spearman_r']:.3f} | {ev['spearman_znorm']:.3f} | {ev['golden_high_mean_pct']:.1f} | {ev['golden_high_in_top25']:.0f}% | {ev['golden_low_mean_pct']:.1f} | {ev['median_cit_high_bpi']:.0f} | {ev['median_cit_low_bpi']:.0f} |\n"

    report += f"\n## Best: {best['formula']}\n"
    report += f"- Spearman r = {best['spearman_r']:.3f}\n"
    report += f"- Golden HIGH mean percentile = {best['golden_high_mean_pct']:.1f}%\n\n"

    report += f"## Top 20 by {best_col}\n\n"
    report += "| Rank | Score | Citations | Year | Title |\n|------|-------|-----------|------|-------|\n"
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        report += f"| {rank} | {row[best_col]:.2f} | {row.get('citation_count', '?')} | {row.get('year', '?')} | {str(row.get('title', ''))[:55]} |\n"

    (EXP_REPORTS / "04b_metric_iteration.md").write_text(report, encoding="utf-8")
    print(f"\nReport: reports/04b_metric_iteration.md")


if __name__ == "__main__":
    main()
