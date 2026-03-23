"""Step 3: Embed ideas and cluster them into base ideas.

Strategy for 62K ideas:
- PCA to 256 dims for fast clustering
- K-Means (primary) with various K
- HDBSCAN as alternative
- Centroids computed in original embedding space for spectrum quality
"""
import asyncio
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from config import (
    EMBEDDING_BATCH,
    EMBEDDING_CONCURRENCY,
    EMBEDDING_MODEL,
    EXP_DATA,
    EXP_REPORTS,
    HDBSCAN_MIN_SIZES,
    KMEANS_KS,
)

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


# === Async embedding ===
async def embed_all(texts: list[str]) -> np.ndarray:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    embeddings = [None] * len(texts)

    async def process_batch(start: int, batch: list[str]):
        async with semaphore:
            resp = await client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            return start, [d.embedding for d in resp.data]

    tasks = []
    for i in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[i : i + EMBEDDING_BATCH]
        batch = [t[:8000] for t in batch]
        tasks.append(process_batch(i, batch))

    print(f"Embedding {len(texts)} texts in {len(tasks)} batches...")
    done = 0
    for coro in asyncio.as_completed(tasks):
        start, embs = await coro
        for j, emb in enumerate(embs):
            embeddings[start + j] = emb
        done += 1
        if done % 10 == 0:
            print(f"  [{done}/{len(tasks)}] batches done")

    return np.array(embeddings, dtype=np.float32)


# === Clustering ===
def try_kmeans(emb_reduced: np.ndarray, emb_original: np.ndarray, ks: list[int]) -> list[dict]:
    """Try K-Means with various K on PCA-reduced embeddings."""
    results = []
    # Sample indices for silhouette (full is too slow)
    sil_n = min(8000, len(emb_reduced))
    sil_idx = np.random.choice(len(emb_reduced), sil_n, replace=False)

    for k in ks:
        if k >= len(emb_reduced):
            continue
        t0 = time.time()
        if len(emb_reduced) > 50000:
            clust = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, batch_size=4096)
        else:
            clust = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = clust.fit_predict(emb_reduced)
        elapsed = time.time() - t0

        sil = silhouette_score(emb_reduced[sil_idx], labels[sil_idx], metric="euclidean", sample_size=min(3000, sil_n))

        sizes = np.bincount(labels)
        results.append({
            "method": "kmeans",
            "param": f"k={k}",
            "n_clusters": k,
            "silhouette": sil,
            "max_cluster": int(sizes.max()),
            "median_cluster": int(np.median(sizes)),
            "min_cluster": int(sizes.min()),
            "singletons": int((sizes == 1).sum()),
            "elapsed": elapsed,
            "labels": labels,
        })
        print(f"  KMeans(k={k}): sil={sil:.4f}, max={int(sizes.max())}, min={int(sizes.min())}, median={int(np.median(sizes))}, {elapsed:.1f}s")
    return results


def try_hdbscan(emb_reduced: np.ndarray, min_sizes: list[int]) -> list[dict]:
    """Try HDBSCAN on L2-normalized PCA embeddings (euclidean ≈ cosine)."""
    from sklearn.cluster import HDBSCAN

    results = []
    sil_n = min(8000, len(emb_reduced))

    for mcs in min_sizes:
        t0 = time.time()
        clust = HDBSCAN(min_cluster_size=mcs, metric="euclidean", n_jobs=-1)
        labels = clust.fit_predict(emb_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        elapsed = time.time() - t0

        sil = -1
        valid = labels >= 0
        if n_clusters >= 2 and valid.sum() > 100:
            valid_idx = np.where(valid)[0]
            sample = np.random.choice(valid_idx, min(sil_n, len(valid_idx)), replace=False)
            sil = silhouette_score(emb_reduced[sample], labels[sample], metric="euclidean", sample_size=min(3000, len(sample)))

        sizes = np.bincount(labels[labels >= 0]) if n_clusters > 0 else np.array([])
        results.append({
            "method": "hdbscan",
            "param": f"min_size={mcs}",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil,
            "max_cluster": int(sizes.max()) if len(sizes) > 0 else 0,
            "median_cluster": int(np.median(sizes)) if len(sizes) > 0 else 0,
            "min_cluster": 0,
            "singletons": int((sizes == 1).sum()) if len(sizes) > 0 else 0,
            "elapsed": elapsed,
            "labels": labels,
        })
        print(f"  HDBSCAN(min={mcs}): K={n_clusters}, noise={n_noise} ({n_noise/len(labels)*100:.0f}%), sil={sil:.4f}, {elapsed:.1f}s")
    return results


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute L2-normalized cluster centroids in original embedding space."""
    unique = sorted(set(labels[labels >= 0]))
    centroids = []
    for cl in unique:
        mask = labels == cl
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids, dtype=np.float32)
    return normalize(centroids)  # L2 normalize


def get_cluster_descriptions(idea_texts: list[str], labels: np.ndarray, embeddings: np.ndarray) -> dict[int, str]:
    """For each cluster, find the idea closest to centroid as description."""
    centroids = compute_centroids(embeddings, labels)
    unique = sorted(set(labels[labels >= 0]))
    descriptions = {}
    for i, cl in enumerate(unique):
        mask = np.where(labels == cl)[0]
        cluster_embs = embeddings[mask]
        # Find closest to centroid
        sims = cluster_embs @ centroids[i]
        best_idx = mask[np.argmax(sims)]
        descriptions[int(cl)] = idea_texts[best_idx]
    return descriptions


def main():
    # Load ideas
    ideas_path = EXP_DATA / "ideas.jsonl"
    records = []
    with open(ideas_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Flatten to unique ideas with metadata
    idea_list = []
    seen = set()
    for rec in records:
        for idea in rec["ideas"]:
            idea_clean = idea.strip().lower()
            if idea_clean not in seen:
                seen.add(idea_clean)
                idea_list.append((idea.strip(), rec["acl_id"], rec["chunk_idx"]))

    idea_texts = [t[0] for t in idea_list]
    print(f"Unique ideas to embed: {len(idea_texts)}")

    # Check for cached embeddings
    emb_path = EXP_DATA / "idea_embeddings.npy"
    meta_path = EXP_DATA / "idea_metadata.json"
    if emb_path.exists() and meta_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(emb_path)
        with open(meta_path, encoding="utf-8") as f:
            cached_meta = json.load(f)
        if len(cached_meta) == len(idea_texts):
            print(f"  Using cached ({embeddings.shape})")
        else:
            print(f"  Cache mismatch ({len(cached_meta)} vs {len(idea_texts)}), re-embedding...")
            embeddings = asyncio.run(embed_all(idea_texts))
            np.save(emb_path, embeddings)
    else:
        embeddings = asyncio.run(embed_all(idea_texts))
        np.save(emb_path, embeddings)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"text": t, "paper_id": p} for t, p, _ in idea_list], f, ensure_ascii=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # === Preprocessing: L2-normalize + PCA ===
    print("\nPreprocessing: L2-normalize + PCA(256)...")
    emb_normed = normalize(embeddings)
    pca = PCA(n_components=256, random_state=42)
    emb_reduced = pca.fit_transform(emb_normed)
    emb_reduced = normalize(emb_reduced)  # re-normalize after PCA
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {embeddings.shape[1]} -> 256 dims, {explained:.1%} variance explained")

    # === K-Means (primary) ===
    print("\n=== K-Means Clustering ===")
    kmeans_results = try_kmeans(emb_reduced, embeddings, KMEANS_KS)

    # === HDBSCAN (subsample for speed) ===
    hdbscan_results = []
    if len(emb_reduced) <= 20000:
        print("\n=== HDBSCAN ===")
        hdbscan_results = try_hdbscan(emb_reduced, HDBSCAN_MIN_SIZES)
    else:
        print(f"\n=== HDBSCAN skipped (N={len(emb_reduced)} > 20K, too slow) ===")

    all_results = kmeans_results + hdbscan_results

    # Selection: prefer K in 100-500 range (enough granularity for spectra)
    # Silhouette naturally favors low K, but we need discrimination ability
    sweet_spot = [r for r in all_results if 100 <= r["n_clusters"] <= 500 and r["silhouette"] > -0.05]
    if sweet_spot:
        best = max(sweet_spot, key=lambda r: r["silhouette"])
    else:
        viable = [r for r in all_results if r["silhouette"] > 0]
        best = max(viable, key=lambda r: r["silhouette"]) if viable else max(all_results, key=lambda r: r["silhouette"])
    print(f"\n=== Best: {best['method']}({best['param']}) K={best['n_clusters']}, sil={best['silhouette']:.4f} ===")

    # === Save best clustering ===
    labels = best["labels"]
    centroids = compute_centroids(embeddings, labels)  # Original space centroids!
    descriptions = get_cluster_descriptions(idea_texts, labels, embeddings)

    np.save(EXP_DATA / "base_idea_embeddings.npy", centroids)
    with open(EXP_DATA / "cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels.tolist(), f)
    with open(EXP_DATA / "cluster_descriptions.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in descriptions.items()}, f, ensure_ascii=False, indent=2)
    with open(EXP_DATA / "cluster_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": best["method"],
            "param": best["param"],
            "n_clusters": best["n_clusters"],
            "silhouette": best["silhouette"],
            "pca_dims": 256,
            "pca_variance": float(explained),
        }, f, indent=2)

    print(f"Saved {best['n_clusters']} base idea centroids ({centroids.shape})")

    # === Write report ===
    report = "# Clustering Comparison Report\n\n"
    report += f"- Total unique ideas: {len(idea_texts)}\n"
    report += f"- Embedding dims: {embeddings.shape[1]}\n"
    report += f"- PCA: 256 dims ({explained:.1%} variance)\n\n"
    report += "| Method | Param | K | Silhouette | Max | Median | Min | Time |\n"
    report += "|--------|-------|---|-----------|-----|--------|-----|------|\n"
    for r in all_results:
        noise_str = f" noise={r.get('n_noise', 0)}" if r.get("n_noise") else ""
        report += (
            f"| {r['method']} | {r['param']} | {r['n_clusters']} | "
            f"{r['silhouette']:.4f} | {r['max_cluster']} | {r['median_cluster']} | "
            f"{r['min_cluster']}{noise_str} | {r['elapsed']:.1f}s |\n"
        )

    report += f"\n## Selected: **{best['method']}({best['param']})**\n"
    report += f"- Clusters: {best['n_clusters']}\n"
    report += f"- Silhouette: {best['silhouette']:.4f}\n\n"

    report += "## Sample Cluster Descriptions (top 30 by size)\n\n"
    sizes = np.bincount(labels[labels >= 0]) if labels.max() >= 0 else np.array([])
    sorted_by_size = sorted(descriptions.items(), key=lambda x: sizes[int(x[0])] if int(x[0]) < len(sizes) else 0, reverse=True)
    for cl_id, desc in sorted_by_size[:30]:
        cl_size = sizes[int(cl_id)] if int(cl_id) < len(sizes) else 0
        report += f"- **[{cl_id}]** (n={cl_size}): {desc}\n"

    (EXP_REPORTS / "03_clustering_report.md").write_text(report, encoding="utf-8")
    print("Report: reports/03_clustering_report.md")


if __name__ == "__main__":
    main()
