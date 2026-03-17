"""Tests for novelty metrics."""

import numpy as np

from novelty_search.metrics.novelty import coherence, idea_spectrum, rarity


def test_idea_spectrum_shape():
    seg_emb = np.random.randn(5, 64).astype(np.float32)
    base_emb = np.random.randn(10, 64).astype(np.float32)
    spectrum = idea_spectrum(seg_emb, base_emb)
    assert spectrum.shape == (10,)


def test_coherence():
    # Concentrated spectrum -> high coherence
    concentrated = np.array([1.0, 0, 0, 0], dtype=np.float32)
    uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    assert coherence(concentrated) > coherence(uniform)


def test_rarity():
    spectrum = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    prevalence = np.array([0.9, 0.5, 0.01], dtype=np.float32)
    r = rarity(spectrum, prevalence)
    # Rare ideas (low prevalence) contribute more to rarity
    assert r > 0
