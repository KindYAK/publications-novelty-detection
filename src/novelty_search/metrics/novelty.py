"""Quantitative novelty metrics for publications in the idea space.

Implements:
- D_M(w): Structural novelty (Mahalanobis distance)
- R(w):   Information rarity of idea configuration
- C(w):   Coherence (spectral concentration)
- ρ(w):   Local density (k-NN based)
- B_pot:  Breakthrough Potential Index
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors


def idea_spectrum(
    segment_embeddings: NDArray[np.float32],
    base_idea_embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute the idea spectrum s(w) for a document.

    s_i(w) = max_j sim(u_i, v_j) where u_i are base idea embeddings
    and v_j are the document's segment idea embeddings.

    Args:
        segment_embeddings: (M, D) embeddings of the document's segment ideas.
        base_idea_embeddings: (K, D) embeddings of base ideas.

    Returns:
        (K,) idea spectrum vector.
    """
    # Normalize
    seg_norm = segment_embeddings / (np.linalg.norm(segment_embeddings, axis=1, keepdims=True) + 1e-10)
    base_norm = base_idea_embeddings / (np.linalg.norm(base_idea_embeddings, axis=1, keepdims=True) + 1e-10)

    # (K, M) similarity matrix
    sim_matrix = base_norm @ seg_norm.T

    # s_i = max over segments
    return np.max(sim_matrix, axis=1)


def structural_novelty(spectrum: NDArray[np.float32], mean: NDArray[np.float32], cov_inv: NDArray[np.float32]) -> float:
    """Mahalanobis distance D_M(w) from the idea field center."""
    return float(mahalanobis(spectrum, mean, cov_inv))


def rarity(spectrum: NDArray[np.float32], mean_prevalence: NDArray[np.float32]) -> float:
    """Information rarity R(w) = -Σ s_i(w) · log(p_i)."""
    log_p = np.log(np.maximum(mean_prevalence, 1e-10))
    return float(-np.sum(spectrum * log_p))


def coherence(spectrum: NDArray[np.float32]) -> float:
    """Spectral concentration C(w) = Σ s_i(w)²."""
    return float(np.sum(spectrum**2))


def local_density(
    spectrum: NDArray[np.float32],
    all_spectra: NDArray[np.float32],
    k: int = 10,
) -> float:
    """Local density ρ(w) based on k-nearest neighbors.

    ρ(w) = (1/k · Σ ||s(w) - s(w')||)^{-1}
    """
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(all_spectra)), metric="euclidean")
    nn.fit(all_spectra)
    distances, _ = nn.kneighbors(spectrum.reshape(1, -1))
    # Skip first neighbor (self) if present
    neighbor_dists = distances[0, 1:]
    mean_dist = np.mean(neighbor_dists)
    return 1.0 / max(mean_dist, 1e-10)


def breakthrough_potential_index(
    spectrum: NDArray[np.float32],
    mean: NDArray[np.float32],
    cov_inv: NDArray[np.float32],
    mean_prevalence: NDArray[np.float32],
    all_spectra: NDArray[np.float32],
    k_neighbors: int = 10,
    mode: str = "full",
) -> dict[str, float]:
    """Compute the Breakthrough Potential Index and all sub-metrics.

    Args:
        spectrum: (K,) idea spectrum of the publication.
        mean: (K,) mean idea spectrum of the corpus.
        cov_inv: (K, K) inverse covariance matrix.
        mean_prevalence: (K,) mean prevalence p_i of each idea.
        all_spectra: (N, K) all spectra in the corpus.
        k_neighbors: number of neighbors for density estimation.
        mode: "simple" for D_M·C, "full" for R·C·(1/ρ).

    Returns:
        Dictionary with all metric values.
    """
    d_m = structural_novelty(spectrum, mean, cov_inv)
    r = rarity(spectrum, mean_prevalence)
    c = coherence(spectrum)
    rho = local_density(spectrum, all_spectra, k=k_neighbors)

    if mode == "simple":
        bpi = d_m * c
    else:
        bpi = r * c * (1.0 / max(rho, 1e-10))

    return {
        "D_M": d_m,
        "R": r,
        "C": c,
        "rho": rho,
        "BPI": bpi,
    }
