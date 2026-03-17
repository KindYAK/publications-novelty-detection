"""Embedding utilities — wraps sentence-transformers and OpenAI embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SentenceEmbedder:
    """Wrapper around sentence-transformers for computing text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.

        Returns:
            (N, D) array of embeddings.
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


class OpenAIEmbedder:
    """Wrapper around OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def encode(self, texts: list[str], batch_size: int = 100) -> NDArray[np.float32]:
        """Encode texts using OpenAI embeddings API."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)
        return np.array(all_embeddings, dtype=np.float32)
