"""
Embedder: wraps a Hugging Face sentence-transformers model and produces
embeddings for chunk texts.

Responsibility: model loading + batched encoding only.

Public API (unchanged): embed_passages(texts: list[str]) -> np.ndarray
New: embed_passages_batched(texts, batch_size) -> Iterator[np.ndarray], a
streaming generator that yields one batch of embeddings at a time. This is
what lets a future caller avoid holding the full corpus + full embedding
matrix in memory simultaneously (see run_ingestion.py, not changed yet).
"""

from collections.abc import Iterable, Iterator
from itertools import islice

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(
        self,
        model_name: str,
        passage_prefix: str = "",
        batch_size: int = 16,
        device: str | None = None,
        use_fp16: bool = True,
    ):
        self.device = device or _auto_device()
        self.model = SentenceTransformer(model_name, device=self.device)

        # fp16 halves activation/weight memory and speeds up inference on
        # GPU/MPS. Skipped on CPU, where fp16 matmul isn't well supported
        # and can be *slower* than fp32 — so this is a pure win, no downside.
        if use_fp16 and self.device in ("cuda", "mps"):
            self.model = self.model.half()

        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_passages_batched(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream embeddings one batch at a time. Accepts any iterable
        (list or generator) — does not require the caller to have the
        full text corpus materialized in memory.
        """
        batch_size = batch_size or self.batch_size
        iterator = iter(texts)

        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return

            # Prefix lazily, per batch — no full-corpus duplicate list.
            prefixed_batch = [f"{self.passage_prefix}{t}" for t in batch]

            with torch.inference_mode():
                batch_embeddings = self.model.encode(
                    prefixed_batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # cosine similarity ready
                    convert_to_numpy=True,
                )

            yield np.asarray(batch_embeddings, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single user query (as a query, not a passage).

        Uses config.E5_QUERY_PREFIX instead of self.passage_prefix, per
        e5's asymmetric query/passage convention. Reuses the already-loaded
        model — no separate model instance is created.

        Returns a (1, embedding_dim) float32 array, ready to pass directly
        to a FAISS index's .search().
        """
        prefixed = f"{config.E5_QUERY_PREFIX}{text}"

        with torch.inference_mode():
            embedding = self.model.encode(
                [prefixed],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        return np.asarray(embedding, dtype="float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of chunk texts (as passages, not queries).

        Same signature and return type as before — existing callers are
        unaffected. Internally now streams through embed_passages_batched()
        and writes into a preallocated array instead of accumulating a
        list of per-batch arrays before a final concatenation, avoiding
        that extra peak-memory moment.
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype="float32")

        output = np.empty((n, self.embedding_dim), dtype="float32")
        offset = 0
        for batch_embeddings in self.embed_passages_batched(texts):
            batch_len = batch_embeddings.shape[0]
            output[offset : offset + batch_len] = batch_embeddings
            offset += batch_len

        return output
