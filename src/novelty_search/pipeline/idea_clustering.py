"""Incremental clustering of ideas into base ideas.

Builds the set I = {I_1, ..., I_K} by processing idea formulations one at a time,
either merging into an existing cluster or creating a new base idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BaseIdea:
    """A base idea (cluster representative)."""

    id: int
    description: str
    embedding: NDArray[np.float32]
    member_descriptions: list[str] = field(default_factory=list)


class IdeaSpace:
    """Incrementally-built space of base ideas."""

    def __init__(self, distance_threshold: float = 0.3):
        self.threshold = distance_threshold
        self.ideas: list[BaseIdea] = []
        self._next_id = 0

    def _cosine_distance(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return 1.0 - float(sim)

    def add_idea(self, description: str, embedding: NDArray[np.float32]) -> tuple[int, bool]:
        """Add a new idea formulation to the space.

        Args:
            description: Textual idea formulation.
            embedding: Vector embedding of the idea.

        Returns:
            (idea_id, is_new) — the ID of the matched/created base idea and whether it's new.
        """
        if not self.ideas:
            idea = BaseIdea(
                id=self._next_id,
                description=description,
                embedding=embedding.copy(),
                member_descriptions=[description],
            )
            self.ideas.append(idea)
            self._next_id += 1
            return idea.id, True

        distances = [self._cosine_distance(embedding, idea.embedding) for idea in self.ideas]
        min_dist = min(distances)
        nearest_idx = int(np.argmin(distances))

        if min_dist < self.threshold:
            # Merge into existing cluster
            self.ideas[nearest_idx].member_descriptions.append(description)
            return self.ideas[nearest_idx].id, False
        else:
            # Create new base idea
            idea = BaseIdea(
                id=self._next_id,
                description=description,
                embedding=embedding.copy(),
                member_descriptions=[description],
            )
            self.ideas.append(idea)
            self._next_id += 1
            return idea.id, True

    def update_representative(self, idea_id: int, new_description: str, new_embedding: NDArray[np.float32]) -> None:
        """Update the representative description and embedding of a base idea after summarization."""
        for idea in self.ideas:
            if idea.id == idea_id:
                idea.description = new_description
                idea.embedding = new_embedding.copy()
                return
        raise ValueError(f"Idea {idea_id} not found")

    def get_embeddings_matrix(self) -> NDArray[np.float32]:
        """Return (K, D) matrix of all base idea embeddings."""
        return np.array([idea.embedding for idea in self.ideas])

    @property
    def k(self) -> int:
        return len(self.ideas)
