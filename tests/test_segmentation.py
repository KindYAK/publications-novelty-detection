"""Tests for semantic text segmentation."""

import numpy as np

from novelty_search.pipeline.segmentation import compute_gap_distances, find_segment_boundaries, segment_sentences


def test_compute_gap_distances_cosine():
    embeddings = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    distances = compute_gap_distances(embeddings, metric="cosine")
    assert len(distances) == 2
    assert distances[0] < 0.01  # identical vectors -> ~0
    assert distances[1] > 0.9  # orthogonal vectors -> ~1


def test_find_segment_boundaries():
    # Create a sequence with a clear spike in the middle
    distances = np.array([0.1, 0.2, 0.9, 0.2, 0.1], dtype=np.float32)
    boundaries = find_segment_boundaries(distances, alpha=0.5)
    assert 2 in boundaries


def test_segment_sentences():
    sentences = ["a", "b", "c", "d", "e"]
    segments = segment_sentences(sentences, boundaries=[1, 3])
    assert len(segments) == 3
    assert segments[0] == ["a", "b"]
    assert segments[1] == ["c", "d"]
    assert segments[2] == ["e"]


def test_no_boundaries():
    sentences = ["a", "b", "c"]
    segments = segment_sentences(sentences, boundaries=[])
    assert len(segments) == 1
    assert segments[0] == sentences
