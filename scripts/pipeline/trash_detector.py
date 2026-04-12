"""Trash paper detector: flag bottom N% by novelty as low-impact.

Usage:
    python trash_detector.py                    # Run on test set, print results
    python trash_detector.py --percentile 20    # Flag bottom 20%

Results (test set, papers >= 2019):
    Bottom 10%: 98.3% precision for <=19 citations
    Bottom 20%: 97.8% precision, 0% false kills above 100 citations
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

PROJECT = Path(__file__).resolve().parent.parent.parent
EXP = PROJECT / "experiments" / "exp001_embeddings" / "data"


def load_data():
    idea_embs = np.load(EXP / "idea_embeddings.npy")
    with open(EXP / "idea_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    subset = pd.read_parquet(EXP / "subset.parquet")

    id_to_cit = dict(zip(subset["acl_id"], subset["citation_count"]))
    id_to_year = dict(zip(subset["acl_id"], subset["year"].astype(int)))
    id_to_title = dict(zip(subset["acl_id"], subset["title"]))

    paper_idea_idxs = {}
    for i, m in enumerate(meta):
        pid = m["paper_id"]
        if pid not in paper_idea_idxs:
            paper_idea_idxs[pid] = []
        paper_idea_idxs[pid].append(i)

    # Max-pool paper embeddings
    paper_embs = {}
    for pid, idxs in paper_idea_idxs.items():
        if not idxs:
            continue
        e = idea_embs[idxs].max(axis=0)
        e /= np.linalg.norm(e) + 1e-10
        paper_embs[pid] = e

    valid = sorted([p for p in paper_idea_idxs if p in id_to_cit and pd.notna(id_to_cit[p])])
    return valid, paper_embs, id_to_cit, id_to_year, id_to_title


def compute_novelty(pids, all_pids, paper_embs, id_to_year, k=10):
    """Compute max-pool kNN novelty for each paper against all prior papers."""
    all_sorted = sorted(all_pids, key=lambda p: id_to_year[p])
    scores = {}
    for pid in pids:
        year = id_to_year[pid]
        prior = [p for p in all_sorted if id_to_year[p] < year and p in paper_embs]
        if len(prior) < k + 1:
            continue
        pe = np.array([paper_embs[p] for p in prior])
        nn = NearestNeighbors(n_neighbors=min(k, len(pe)), metric="cosine")
        nn.fit(pe)
        d, _ = nn.kneighbors(paper_embs[pid].reshape(1, -1))
        scores[pid] = float(d[0].mean())
    return scores


def flag_trash(pids, scores, percentile=20):
    """Flag bottom N% by novelty score as trash."""
    scored = [(p, scores[p]) for p in pids if p in scores]
    vals = [s for _, s in scored]
    threshold = np.percentile(vals, percentile)
    flagged = [p for p, s in scored if s <= threshold]
    not_flagged = [p for p, s in scored if s > threshold]
    return flagged, not_flagged, threshold


def evaluate_trash_detection(flagged, not_flagged, id_to_cit, id_to_title):
    """Evaluate trash detection quality."""
    flagged_cit = [id_to_cit[p] for p in flagged]
    not_flagged_cit = [id_to_cit[p] for p in not_flagged]

    results = {"n_flagged": len(flagged), "n_kept": len(not_flagged)}

    for thresh_name, thresh in [("<=4", 4), ("<=9", 9), ("<=19", 19)]:
        truly_trash = sum(1 for c in flagged_cit if c <= thresh)
        precision = truly_trash / max(len(flagged), 1)
        total_trash = sum(1 for c in flagged_cit + not_flagged_cit if c <= thresh)
        recall = truly_trash / max(total_trash, 1)
        results[f"precision_{thresh_name}"] = precision
        results[f"recall_{thresh_name}"] = recall

    # Citation distribution of flagged
    flagged_arr = np.array(flagged_cit)
    results["flagged_zero_pct"] = (flagged_arr == 0).mean()
    results["flagged_under5_pct"] = (flagged_arr <= 4).mean()
    results["flagged_under20_pct"] = (flagged_arr <= 19).mean()
    results["flagged_max_cit"] = int(flagged_arr.max()) if len(flagged_arr) else 0

    # False kills
    false_kills = [(p, id_to_cit[p]) for p in flagged if id_to_cit[p] > 19]
    false_kills.sort(key=lambda x: -x[1])

    return results, false_kills


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Flag low-novelty papers as trash")
    parser.add_argument("--percentile", type=int, default=20, help="Flag bottom N%% (default: 20)")
    parser.add_argument("--split-year", type=int, default=2019, help="Test set split year")
    args = parser.parse_args()

    print("Loading data...")
    valid, paper_embs, id_to_cit, id_to_year, id_to_title = load_data()

    test = [p for p in valid if id_to_year[p] >= args.split_year]
    print(f"Test papers (>={args.split_year}): {len(test)}")

    print("Computing novelty scores...")
    t0 = time.time()
    scores = compute_novelty(test, valid, paper_embs, id_to_year, k=10)
    print(f"  Scored {len(scores)} papers in {time.time()-t0:.0f}s")

    # Sweep percentiles
    print(f"\n{'Pctl':>5} {'Flagged':>8} {'Prec<=4':>8} {'Prec<=9':>8} {'Prec<=19':>9} {'MaxCit':>7} {'FalseKills':>11}")
    print("-" * 62)
    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        flagged, kept, thresh = flag_trash(test, scores, percentile=pct)
        res, fk = evaluate_trash_detection(flagged, kept, id_to_cit, id_to_title)
        print(
            f"{pct:>4}% {res['n_flagged']:>8} "
            f"{res['precision_<=4']:.1%}{'':<2} {res['precision_<=9']:.1%}{'':<2} "
            f"{res['precision_<=19']:.1%}{'':<3} {res['flagged_max_cit']:>7} "
            f"{len(fk):>11}"
        )

    # Detailed for selected percentile
    pct = args.percentile
    print(f"\n{'='*62}")
    print(f"DETAILED: Bottom {pct}% flagged as trash")
    print(f"{'='*62}")

    flagged, kept, thresh = flag_trash(test, scores, percentile=pct)
    res, false_kills = evaluate_trash_detection(flagged, kept, id_to_cit, id_to_title)

    print(f"Flagged: {res['n_flagged']} papers (novelty <= {thresh:.4f})")
    print(f"Precision (<=19 cit): {res['precision_<=19']:.1%}")
    print(f"Precision (<=9 cit):  {res['precision_<=9']:.1%}")
    print(f"Precision (<=4 cit):  {res['precision_<=4']:.1%}")

    flagged_cit = sorted([id_to_cit[p] for p in flagged])
    print(f"\nCitation distribution of flagged papers:")
    for label, lo, hi in [("0", 0, 0), ("1-4", 1, 4), ("5-19", 5, 19),
                           ("20-49", 20, 49), ("50-99", 50, 99), ("100+", 100, 999999)]:
        n = sum(1 for c in flagged_cit if lo <= c <= hi)
        print(f"  {label:>8}: {n:>5} ({n/len(flagged)*100:>5.1f}%)")

    if false_kills:
        print(f"\nFalse kills (flagged but >19 citations):")
        for pid, cit in false_kills[:10]:
            print(f"  cit={cit:>5} | {id_to_title[pid][:70]}")
    else:
        print("\nNo false kills!")


if __name__ == "__main__":
    main()
