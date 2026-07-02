"""
Storage: persists chunk metadata + text to JSON, and vectors to a FAISS index.

This is a local staging format, not the production vector DB. Keeping it
decoupled from Qdrant means re-indexing into Qdrant later doesn't require
re-embedding — you just load this JSON + index and upsert.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


def save_chunks_json(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_id": c.chunk_id,
            "lesson_id": c.lesson_id,
            "lesson_title": c.lesson_title,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "boundary_reason": c.boundary_reason,
            "text": c.text,
        }
        for c in chunks
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_faiss_index(embeddings: np.ndarray, path: Path) -> None:
    """Flat, exact-search index — fine for staging; Qdrant handles production search."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine, since vectors are normalized
    index.add(embeddings)
    faiss.write_index(index, str(path))


def load_chunks_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
