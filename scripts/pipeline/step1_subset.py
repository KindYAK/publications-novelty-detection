"""Step 1: Select embeddings-related subset from ACL dataset."""
import re
from pathlib import Path

import pandas as pd

from config import (
    ACL_FULLTEXT,
    ACL_PARQUET,
    EXP_DATA,
    EXP_REPORTS,
    GOLDEN_HIGH,
    GOLDEN_MEDIUM,
    SUBSET_KEYWORDS,
)


def matches_any_keyword(text: str, keywords: list[str]) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def find_fulltext(acl_id) -> Path | None:
    """Map acl_id to fulltext file path."""
    if not isinstance(acl_id, str) or not acl_id:
        return None
    fname = acl_id.replace(".", "_").replace("/", "_") + ".txt"
    path = ACL_FULLTEXT / fname
    return path if path.exists() else None


def main():
    print("Loading ACL dataset...")
    df = pd.read_parquet(ACL_PARQUET)
    print(f"  Total papers: {len(df):,}")

    # Keyword filter on title + abstract
    kw_lower = [k.lower() for k in SUBSET_KEYWORDS]
    mask = df.apply(
        lambda r: matches_any_keyword(r.get("title", ""), kw_lower)
        or matches_any_keyword(r.get("abstract", ""), kw_lower),
        axis=1,
    )
    subset = df[mask].copy()
    print(f"  After keyword filter: {len(subset):,}")

    # Check fulltext availability
    subset["fulltext_path"] = subset["acl_id"].apply(
        lambda x: (str(p) if (p := find_fulltext(x)) else None)
    )
    subset["has_fulltext"] = subset["fulltext_path"].notna()
    n_ft = subset["has_fulltext"].sum()
    print(f"  With fulltext: {n_ft:,} ({n_ft / len(subset) * 100:.1f}%)")

    # Keep only papers with fulltext
    subset = subset[subset["has_fulltext"]].copy()
    print(f"  Final subset: {len(subset):,}")

    # === Check golden set ===
    print("\n=== Golden Set Check ===")
    all_golden = GOLDEN_HIGH + GOLDEN_MEDIUM
    for title_frag in all_golden:
        found = subset[subset["title"].str.contains(title_frag, case=False, na=False)]
        tier = "HIGH" if title_frag in GOLDEN_HIGH else "MED"
        if len(found) > 0:
            row = found.iloc[0]
            print(f"  [{tier}] FOUND: {row['title'][:80]} (year={row.get('year', '?')}, cit={row.get('citation_count', '?')})")
        else:
            # Check full dataset
            found_full = df[df["title"].str.contains(title_frag, case=False, na=False)]
            if len(found_full) > 0:
                print(f"  [{tier}] IN FULL BUT NOT SUBSET: {found_full.iloc[0]['title'][:80]}")
            else:
                print(f"  [{tier}] NOT FOUND: {title_frag[:60]}")

    # === Citation stats ===
    cit = subset["citation_count"].dropna()
    print(f"\n=== Citation Distribution (n={len(cit):,}) ===")
    print(f"  Mean: {cit.mean():.1f}, Median: {cit.median():.0f}")
    print(f"  Max: {cit.max():.0f}, Min: {cit.min():.0f}")
    buckets = {
        "0 (zero)": (cit == 0).sum(),
        "1-4 (rare)": ((cit >= 1) & (cit <= 4)).sum(),
        "5-19 (moderate)": ((cit >= 5) & (cit <= 19)).sum(),
        "20-99 (solid)": ((cit >= 20) & (cit <= 99)).sum(),
        "100-499 (high)": ((cit >= 100) & (cit <= 499)).sum(),
        "500+ (breakthrough)": (cit >= 500).sum(),
    }
    for label, count in buckets.items():
        print(f"  {label}: {count:,} ({count / len(cit) * 100:.1f}%)")

    # Year distribution
    print(f"\n=== Year Range ===")
    yr = subset["year"].dropna()
    print(f"  {int(yr.min())} - {int(yr.max())}")
    for y in sorted(yr.unique()):
        n = (yr == y).sum()
        if n > 10:
            print(f"  {int(y)}: {n:,}")

    # Save
    out_path = EXP_DATA / "subset.parquet"
    subset.to_parquet(out_path, index=False)
    print(f"\nSaved subset to {out_path}")

    # Write report
    report = f"""# Subset Selection Report

## Source
- ACL dataset: {len(df):,} papers
- Keywords: {len(SUBSET_KEYWORDS)} terms ({', '.join(SUBSET_KEYWORDS[:5])}...)

## Result
- Matched: {mask.sum():,} papers
- With fulltext: {len(subset):,} papers
- Year range: {int(yr.min())}-{int(yr.max())}

## Citation Distribution
| Bucket | Count | % |
|--------|-------|---|
"""
    for label, count in buckets.items():
        report += f"| {label} | {count:,} | {count / len(cit) * 100:.1f}% |\n"

    report += f"\n## Golden Set Coverage\n"
    for title_frag in all_golden:
        found = subset[subset["title"].str.contains(title_frag, case=False, na=False)]
        status = "FOUND" if len(found) > 0 else "MISSING"
        report += f"- [{status}] {title_frag[:60]}\n"

    (EXP_REPORTS / "01_subset_stats.md").write_text(report, encoding="utf-8")
    print("Report saved to reports/01_subset_stats.md")


if __name__ == "__main__":
    main()
