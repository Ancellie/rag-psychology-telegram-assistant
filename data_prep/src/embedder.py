"""
Embedder: wraps a Hugging Face sentence-transformers model and produces
embeddings for a list of chunk texts.

Responsibility: model loading + batched encoding only.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str, passage_prefix: str = "", batch_size: int = 16):
        self.model = SentenceTransformer(model_name)
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a list of chunk texts (as passages, not queries)."""
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity ready
        )
        return np.asarray(embeddings, dtype="float32")
