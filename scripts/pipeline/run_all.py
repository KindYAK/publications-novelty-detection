"""Run the full pipeline end-to-end."""
import sys
import time
from pathlib import Path

# Ensure scripts/pipeline is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step1_subset
import step2_extract_ideas
import step3_cluster
import step4_metrics
import step5_validate


def main():
    t0 = time.time()

    print("=" * 60)
    print("STEP 1: Subset Selection")
    print("=" * 60)
    step1_subset.main()

    print("\n" + "=" * 60)
    print("STEP 2: Idea Extraction")
    print("=" * 60)
    step2_extract_ideas.main()

    print("\n" + "=" * 60)
    print("STEP 3: Clustering")
    print("=" * 60)
    step3_cluster.main()

    print("\n" + "=" * 60)
    print("STEP 4: Metrics")
    print("=" * 60)
    step4_metrics.main()

    print("\n" + "=" * 60)
    print("STEP 5: Validation")
    print("=" * 60)
    step5_validate.main()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE — Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
