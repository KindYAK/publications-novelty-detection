"""Step 5: Validate BPI against golden set and citation buckets."""
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

from config import EXP_DATA, EXP_REPORTS, GOLDEN_HIGH, GOLDEN_LOW, GOLDEN_MEDIUM


def find_golden_papers(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Find golden set papers in the metrics dataframe."""
    golden = {"high": [], "medium": [], "low": [], "not_found": []}

    def _find(title_frag, tier):
        mask = df["title"].str.contains(title_frag, case=False, na=False)
        if mask.any():
            row = df[mask].iloc[0]
            golden[tier].append({
                "title_query": title_frag,
                "title": row["title"],
                "paper_id": row["paper_id"],
                "year": row.get("year"),
                "citation_count": row.get("citation_count"),
                "BPI": row.get("BPI"),
                "D_M": row.get("D_M"),
                "R": row.get("R"),
                "C": row.get("C"),
                "rho": row.get("rho"),
                "BPI_rank": None,
                "BPI_percentile": None,
            })
        else:
            golden["not_found"].append(f"[{tier}] {title_frag}")

    for t in GOLDEN_HIGH:
        _find(t, "high")
    for t in GOLDEN_MEDIUM:
        _find(t, "medium")
    for t in GOLDEN_LOW:
        _find(t, "low")

    return golden


def citation_bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze BPI by citation buckets."""
    df = df.dropna(subset=["BPI", "citation_count"]).copy()
    df["log_cit"] = np.log1p(df["citation_count"])

    # Harsh buckets — most papers are low-impact
    bins = [0, 1, 5, 20, 100, 500, float("inf")]
    labels = ["0 (zero)", "1-4 (noise)", "5-19 (minor)", "20-99 (decent)", "100-499 (strong)", "500+ (breakthrough)"]
    df["bucket"] = pd.cut(df["citation_count"], bins=bins, labels=labels, right=False)

    stats = df.groupby("bucket", observed=True).agg(
        count=("BPI", "size"),
        mean_BPI=("BPI", "mean"),
        median_BPI=("BPI", "median"),
        std_BPI=("BPI", "std"),
        mean_DM=("D_M", "mean"),
        mean_R=("R", "mean"),
        mean_C=("C", "mean"),
        mean_rho=("rho", "mean"),
        mean_cit=("citation_count", "mean"),
    ).reset_index()
    return stats


def main():
    print("Loading metrics...")
    df = pd.read_parquet(EXP_DATA / "metrics.parquet")
    valid = df.dropna(subset=["BPI"]).copy()
    print(f"Papers with metrics: {len(valid)}")

    if len(valid) == 0:
        print("ERROR: No valid metrics. Check previous steps.")
        return

    # Compute ranks and percentiles
    valid["BPI_rank"] = valid["BPI"].rank(ascending=False, method="min").astype(int)
    valid["BPI_percentile"] = valid["BPI"].rank(pct=True) * 100

    # === Golden Set ===
    print("\n=== Golden Set Evaluation ===")
    golden = find_golden_papers(valid)

    # Fill ranks
    for tier in ["high", "medium", "low"]:
        for paper in golden[tier]:
            mask = valid["paper_id"] == paper["paper_id"]
            if mask.any():
                row = valid[mask].iloc[0]
                paper["BPI_rank"] = int(row["BPI_rank"])
                paper["BPI_percentile"] = float(row["BPI_percentile"])

    for tier in ["high", "medium", "low"]:
        print(f"\n  [{tier.upper()} expected]")
        for p in golden[tier]:
            bpi_val = p['BPI'] if p['BPI'] is not None and not (isinstance(p['BPI'], float) and np.isnan(p['BPI'])) else 0
            print(
                f"    {p['title'][:60]}... | BPI={bpi_val:.4f} "
                f"rank={p['BPI_rank']}/{len(valid)} "
                f"top {100 - (p['BPI_percentile'] or 0):.1f}% | cit={p['citation_count']}"
            )

    if golden["not_found"]:
        print(f"\n  [NOT FOUND] {golden['not_found']}")

    # Golden set score
    golden_high_pcts = [p["BPI_percentile"] for p in golden["high"] if p["BPI_percentile"] is not None]
    golden_low_pcts = [p["BPI_percentile"] for p in golden["low"] if p["BPI_percentile"] is not None]
    if golden_high_pcts:
        in_top10 = sum(1 for p in golden_high_pcts if p >= 90) / len(golden_high_pcts) * 100
        in_top25 = sum(1 for p in golden_high_pcts if p >= 75) / len(golden_high_pcts) * 100
        mean_pct = np.mean(golden_high_pcts)
        print(f"\n  Golden HIGH in top 10%: {in_top10:.0f}%")
        print(f"  Golden HIGH in top 25%: {in_top25:.0f}%")
        print(f"  Golden HIGH mean percentile: {mean_pct:.1f}%")
    if golden_low_pcts:
        low_in_bottom50 = sum(1 for p in golden_low_pcts if p <= 50) / len(golden_low_pcts) * 100
        print(f"  Golden LOW in bottom 50%: {low_in_bottom50:.0f}%")
        print(f"  Golden LOW mean percentile: {np.mean(golden_low_pcts):.1f}%")

    # === Citation Bucket Analysis ===
    print("\n=== Citation Bucket Analysis ===")
    bucket_stats = citation_bucket_analysis(valid)
    print(bucket_stats.to_string(index=False))

    # === Correlation Analysis ===
    print("\n=== Correlation: BPI vs Citations ===")
    cit_valid = valid.dropna(subset=["citation_count"])
    cit_valid = cit_valid[cit_valid["citation_count"] >= 0]

    log_cit = np.log1p(cit_valid["citation_count"].values)
    bpi = cit_valid["BPI"].values

    spearman_r, spearman_p = spearmanr(bpi, log_cit)
    kendall_t, kendall_p = kendalltau(bpi, log_cit)
    print(f"  Spearman: r={spearman_r:.3f}, p={spearman_p:.2e}")
    print(f"  Kendall:  τ={kendall_t:.3f}, p={kendall_p:.2e}")

    # Mann-Whitney: top 25% BPI vs bottom 25% BPI — do they differ in citations?
    q75 = np.percentile(bpi, 75)
    q25 = np.percentile(bpi, 25)
    high_bpi_cit = cit_valid[cit_valid["BPI"] >= q75]["citation_count"]
    low_bpi_cit = cit_valid[cit_valid["BPI"] <= q25]["citation_count"]
    if len(high_bpi_cit) > 5 and len(low_bpi_cit) > 5:
        mw_stat, mw_p = mannwhitneyu(high_bpi_cit, low_bpi_cit, alternative="greater")
        print(f"  Mann-Whitney (high vs low BPI citations): U={mw_stat:.0f}, p={mw_p:.2e}")
        print(f"    High BPI median cit: {high_bpi_cit.median():.0f}, Low BPI: {low_bpi_cit.median():.0f}")

    # === Per-metric correlations ===
    print("\n=== Per-Metric Correlations with log(citations) ===")
    for metric in ["D_M", "R", "C", "rho", "BPI"]:
        vals = cit_valid[metric].values
        r, p = spearmanr(vals, log_cit)
        print(f"  {metric:5s}: Spearman r={r:.3f} (p={p:.2e})")

    # === Save full report ===
    report = "# Validation Report\n\n"

    report += "## Golden Set\n\n"
    report += "### Expected HIGH BPI\n"
    report += "| Paper | Year | Citations | BPI | Rank | Percentile |\n"
    report += "|-------|------|-----------|-----|------|------------|\n"
    for p in golden["high"]:
        report += f"| {p['title'][:50]}... | {p.get('year', '?')} | {p.get('citation_count', '?')} | {p.get('BPI', 0):.4f} | {p.get('BPI_rank', '?')}/{len(valid)} | {p.get('BPI_percentile', 0):.1f}% |\n"

    for tier_name in ["MEDIUM", "LOW"]:
        tier_key = tier_name.lower()
        report += f"\n### Expected {tier_name} BPI\n"
        report += "| Paper | Year | Citations | BPI | Rank | Percentile |\n"
        report += "|-------|------|-----------|-----|------|------------|\n"
        for p in golden[tier_key]:
            bpi_val = p.get('BPI', 0) or 0
            report += f"| {p['title'][:50]}... | {p.get('year', '?')} | {p.get('citation_count', '?')} | {bpi_val:.4f} | {p.get('BPI_rank', '?')}/{len(valid)} | {p.get('BPI_percentile', 0):.1f}% |\n"

    if golden["not_found"]:
        report += f"\n### Not Found\n"
        for t in golden["not_found"]:
            report += f"- {t}\n"

    if golden_high_pcts:
        report += f"\n### Golden Set Score\n"
        report += f"- HIGH papers in top 10%: {in_top10:.0f}%\n"
        report += f"- HIGH papers in top 25%: {in_top25:.0f}%\n"
        report += f"- HIGH mean percentile: {mean_pct:.1f}%\n"

    report += "\n## Citation Bucket Analysis\n\n"
    report += bucket_stats.to_string(index=False)

    report += "\n\n## Correlations\n"
    report += f"- Spearman (BPI vs log_cit): r={spearman_r:.3f}, p={spearman_p:.2e}\n"
    report += f"- Kendall (BPI vs log_cit): τ={kendall_t:.3f}, p={kendall_p:.2e}\n"

    report += "\n### Per-Metric Correlations\n"
    report += "| Metric | Spearman r | p-value |\n|--------|-----------|----------|\n"
    for metric in ["D_M", "R", "C", "rho", "BPI"]:
        vals = cit_valid[metric].values
        r, p = spearmanr(vals, log_cit)
        report += f"| {metric} | {r:.3f} | {p:.2e} |\n"

    report += f"\n## Top 30 Papers by BPI\n\n"
    report += "| Rank | BPI | Cit | Year | Title |\n"
    report += "|------|-----|-----|------|-------|\n"
    top30 = valid.nlargest(30, "BPI")
    for rank, (_, row) in enumerate(top30.iterrows(), 1):
        report += f"| {rank} | {row['BPI']:.4f} | {row.get('citation_count', '?')} | {row.get('year', '?')} | {str(row.get('title', ''))[:55]} |\n"

    report += f"\n## Bottom 30 Papers by BPI\n\n"
    report += "| Rank | BPI | Cit | Year | Title |\n"
    report += "|------|-----|-----|------|-------|\n"
    bot30 = valid.nsmallest(30, "BPI")
    for rank, (_, row) in enumerate(bot30.iterrows(), 1):
        report += f"| {rank} | {row['BPI']:.4f} | {row.get('citation_count', '?')} | {row.get('year', '?')} | {str(row.get('title', ''))[:55]} |\n"

    (EXP_REPORTS / "05_validation_report.md").write_text(report, encoding="utf-8")
    print("\nFull report: reports/05_validation_report.md")

    # Save golden set results as JSON for easy re-use
    with open(EXP_DATA / "golden_results.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
