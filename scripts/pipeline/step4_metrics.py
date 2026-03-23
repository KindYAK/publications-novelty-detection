"""Step 4: Compute idea spectra and novelty metrics (with temporal ordering)."""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors

from config import COV_REG, EXP_DATA, EXP_REPORTS, K_NEIGHBORS, MIN_PRIOR_PAPERS


def build_paper_idea_map(ideas_path, embeddings_path, metadata_path):
    """Map each paper to its idea embedding indices."""
    embeddings = np.load(embeddings_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Build paper → idea indices mapping
    paper_ideas = {}  # paper_id → list of indices into embeddings
    for i, meta in enumerate(metadata):
        pid = meta["paper_id"]
        if pid not in paper_ideas:
            paper_ideas[pid] = []
        paper_ideas[pid].append(i)

    return embeddings, paper_ideas


def compute_spectrum(paper_idea_embs, base_idea_embs):
    """Compute K-dim idea spectrum for a paper.

    s_i(w) = max_j cos_sim(base_i, idea_j)
    """
    if len(paper_idea_embs) == 0:
        return np.zeros(len(base_idea_embs), dtype=np.float32)

    # Normalize
    p_norm = paper_idea_embs / (np.linalg.norm(paper_idea_embs, axis=1, keepdims=True) + 1e-10)
    b_norm = base_idea_embs / (np.linalg.norm(base_idea_embs, axis=1, keepdims=True) + 1e-10)

    sim = b_norm @ p_norm.T  # (K, M)
    return np.max(sim, axis=1).astype(np.float32)


def compute_metrics_temporal(spectra, years, paper_ids, k_neighbors=K_NEIGHBORS, min_prior=MIN_PRIOR_PAPERS):
    """Compute novelty metrics with temporal ordering.

    For each paper, metrics are computed against all papers from PRIOR years.
    """
    unique_years = sorted(set(years))
    results = []

    for year in unique_years:
        prior_mask = np.array([y < year for y in years])
        year_mask = np.array([y == year for y in years])

        n_prior = prior_mask.sum()
        if n_prior < min_prior:
            # Not enough history — mark as N/A
            for i in np.where(year_mask)[0]:
                results.append({
                    "paper_id": paper_ids[i],
                    "year": year,
                    "D_M": np.nan,
                    "R": np.nan,
                    "C": np.nan,
                    "rho": np.nan,
                    "BPI": np.nan,
                    "n_prior": n_prior,
                })
            continue

        prior_specs = spectra[prior_mask]
        year_specs = spectra[year_mask]
        year_pids = [paper_ids[i] for i in np.where(year_mask)[0]]

        # Field statistics from prior years
        mean = prior_specs.mean(axis=0)
        cov = np.cov(prior_specs.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(len(mean)) * COV_REG
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        # Prevalence = mean spectrum (normalized)
        prevalence = mean / (mean.sum() + 1e-10)

        # k-NN model for density
        k = min(k_neighbors, len(prior_specs) - 1)
        if k < 1:
            k = 1
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(prior_specs)

        for i, (spec, pid) in enumerate(zip(year_specs, year_pids)):
            # D_M: Mahalanobis distance
            try:
                d_m = float(mahalanobis(spec, mean, cov_inv))
            except Exception:
                d_m = float(np.sqrt(np.sum((spec - mean) ** 2)))

            # R: Rarity
            log_p = np.log(np.maximum(prevalence, 1e-10))
            r = float(-np.sum(spec * log_p))

            # C: Coherence
            c = float(np.sum(spec**2))

            # ρ: Local density
            dists, _ = nn.kneighbors(spec.reshape(1, -1))
            mean_dist = np.mean(dists[0])
            rho = 1.0 / max(mean_dist, 1e-10)

            # BPI
            bpi = r * c * (1.0 / max(rho, 1e-10))

            results.append({
                "paper_id": pid,
                "year": year,
                "D_M": d_m,
                "R": r,
                "C": c,
                "rho": rho,
                "BPI": bpi,
                "n_prior": n_prior,
            })

        print(f"  Year {year}: {year_mask.sum()} papers, {n_prior} prior | mean BPI={np.mean([r['BPI'] for r in results[-year_mask.sum():] if not np.isnan(r['BPI'])]):.4f}")

    return results


def main():
    print("Loading data...")
    subset = pd.read_parquet(EXP_DATA / "subset.parquet")
    base_idea_embs = np.load(EXP_DATA / "base_idea_embeddings.npy")
    idea_embeddings, paper_ideas = build_paper_idea_map(
        EXP_DATA / "ideas.jsonl",
        EXP_DATA / "idea_embeddings.npy",
        EXP_DATA / "idea_metadata.json",
    )

    K = len(base_idea_embs)
    print(f"Base ideas (K): {K}")
    print(f"Papers with ideas: {len(paper_ideas)}")

    # Compute spectra
    print("Computing idea spectra...")
    paper_ids = []
    years = []
    spectra = []

    for _, row in subset.iterrows():
        pid = row["acl_id"]
        year = row.get("year", None)
        if pid not in paper_ideas or year is None or np.isnan(year):
            continue

        idxs = paper_ideas[pid]
        p_embs = idea_embeddings[idxs]
        spec = compute_spectrum(p_embs, base_idea_embs)

        paper_ids.append(pid)
        years.append(int(year))
        spectra.append(spec)

    spectra = np.array(spectra, dtype=np.float32)
    print(f"Spectra shape: {spectra.shape}")

    # Save spectra
    np.save(EXP_DATA / "spectra.npy", spectra)
    with open(EXP_DATA / "spectra_paper_ids.json", "w") as f:
        json.dump(paper_ids, f)

    # Compute temporal metrics
    print("\nComputing metrics with temporal ordering...")
    metrics = compute_metrics_temporal(spectra, years, paper_ids)

    # Merge with paper metadata
    metrics_df = pd.DataFrame(metrics)
    subset_slim = subset[["acl_id", "title", "citation_count", "influential_citation_count", "venue"]].copy()
    merged = metrics_df.merge(subset_slim, left_on="paper_id", right_on="acl_id", how="left")

    # Save
    merged.to_parquet(EXP_DATA / "metrics.parquet", index=False)
    print(f"\nSaved metrics for {len(merged)} papers")

    # Quick stats
    valid = merged.dropna(subset=["BPI"])
    if len(valid) > 0:
        print(f"\n=== BPI Stats (n={len(valid)}) ===")
        print(f"  Mean: {valid['BPI'].mean():.4f}")
        print(f"  Median: {valid['BPI'].median():.4f}")
        print(f"  Std: {valid['BPI'].std():.4f}")
        print(f"  Max: {valid['BPI'].max():.4f}")

        # Top 20 by BPI
        top = valid.nlargest(20, "BPI")
        print("\n=== Top 20 by BPI ===")
        for _, row in top.iterrows():
            cit = row.get("citation_count", "?")
            print(f"  BPI={row['BPI']:.4f} | cit={cit} | {str(row.get('title', ''))[:70]}")

    # Report
    report = f"""# Metrics Report

## Config
- Base ideas (K): {K}
- k-neighbors: {K_NEIGHBORS}
- Min prior papers: {MIN_PRIOR_PAPERS}

## Spectra
- Papers with spectra: {len(spectra)}
- Year range: {min(years)}-{max(years)}

## Metric Stats
"""
    if len(valid) > 0:
        for col in ["D_M", "R", "C", "rho", "BPI"]:
            report += f"- {col}: mean={valid[col].mean():.4f}, std={valid[col].std():.4f}, median={valid[col].median():.4f}\n"
        report += "\n## Top 20 by BPI\n"
        report += "| Rank | BPI | Citations | Title |\n|------|-----|-----------|-------|\n"
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            report += f"| {rank} | {row['BPI']:.4f} | {row.get('citation_count', '?')} | {str(row.get('title', ''))[:60]} |\n"

    (EXP_REPORTS / "04_metrics_report.md").write_text(report, encoding="utf-8")
    print("Report: reports/04_metrics_report.md")


if __name__ == "__main__":
    main()
