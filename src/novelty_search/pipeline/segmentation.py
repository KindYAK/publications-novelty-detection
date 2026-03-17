"""Semantic text segmentation using embedding-based boundary detection.

Implements the local maximum gap method from the paper:
1. Compute sentence embeddings
2. Compute pairwise distances between consecutive embeddings
3. Find local maxima exceeding adaptive threshold τ = μ + α·σ
4. Split text at detected boundaries
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_gap_distances(embeddings: NDArray[np.float32], metric: str = "cosine") -> NDArray[np.float32]:
    """Compute distances between consecutive sentence embeddings.

    Args:
        embeddings: (T, D) array of sentence embeddings.
        metric: distance metric — "cosine" or "euclidean".

    Returns:
        (T-1,) array of inter-sentence distances.
    """
    if metric == "cosine":
        # cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-10)
        sims = np.sum(normed[:-1] * normed[1:], axis=1)
        return 1.0 - sims
    elif metric == "euclidean":
        return np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_segment_boundaries(distances: NDArray[np.float32], alpha: float = 1.0) -> list[int]:
    """Detect segment boundaries via local maximum gap method.

    A position t is a boundary if:
      - d_t > d_{t-1} AND d_t > d_{t+1}  (local maximum)
      - d_t > μ + α·σ  (exceeds adaptive threshold)

    Args:
        distances: (T-1,) array of inter-sentence distances.
        alpha: sensitivity parameter for the adaptive threshold.

    Returns:
        List of boundary positions (0-indexed into the distances array).
    """
    if len(distances) < 3:
        return []

    mu = np.mean(distances)
    sigma = np.std(distances)
    tau = mu + alpha * sigma

    boundaries = []
    for t in range(1, len(distances) - 1):
        if distances[t] > distances[t - 1] and distances[t] > distances[t + 1] and distances[t] > tau:
            boundaries.append(t)

    return boundaries


def segment_sentences(sentences: list[str], boundaries: list[int]) -> list[list[str]]:
    """Split sentences into segments at the given boundary positions.

    Args:
        sentences: List of sentences.
        boundaries: Boundary positions (indices after which to split).

    Returns:
        List of segments, where each segment is a list of sentences.
    """
    if not boundaries:
        return [sentences]

    segments = []
    prev = 0
    for b in sorted(boundaries):
        split_at = b + 1  # boundary is between sentence b and b+1
        segments.append(sentences[prev:split_at])
        prev = split_at
    if prev < len(sentences):
        segments.append(sentences[prev:])

    return segments
