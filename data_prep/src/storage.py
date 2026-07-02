"""
Storage: persists chunk metadata + text to JSON, and vectors to a FAISS index.

This is a local staging format, not the production vector DB. Keeping it
decoupled from Qdrant means re-indexing into Qdrant later doesn't require
re-embedding — you just load this JSON + index and upsert.

Two persistence styles are available:
- save_chunks_json() / save_faiss_index(): unchanged, whole-corpus-at-once.
  Kept exactly as before for backward compatibility.
- ChunkJsonWriter / FaissIndexBuilder: new, incremental. Let a streaming
  caller (e.g. a per-lesson ingestion loop) append/add data one lesson at
  a time without ever holding the full corpus in memory, while producing
  byte-for-byte the same output formats as the whole-corpus functions.
"""

import json
from pathlib import Path
from types import TracebackType

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


def _chunk_to_record(c: Chunk) -> dict:
    """Same field mapping as save_chunks_json(), factored out so both paths agree."""
    return {
        "chunk_id": c.chunk_id,
        "lesson_id": c.lesson_id,
        "lesson_title": c.lesson_title,
        "source_file": c.source_file,
        "chunk_index": c.chunk_index,
        "token_count": c.token_count,
        "boundary_reason": c.boundary_reason,
        "text": c.text,
    }


class ChunkJsonWriter:
    """
    Streaming writer that produces the exact same output as save_chunks_json()
    — a single valid JSON array, fully readable by load_chunks_json() — but
    writes it incrementally, one lesson's chunks at a time, instead of
    requiring the full chunk list in memory.

    Usage:
        with ChunkJsonWriter(path) as writer:
            for lesson in lessons:
                chunks = build_chunks(...)
                writer.write_chunks(chunks)
        # file now contains a valid JSON array on disk, e.g. []  or  [ {...}, {...} ]

    Guarantees:
    - The array brackets are always balanced: '[' is written on open, ']' on
      close, even if zero chunks were ever written (result: valid "[]").
    - Commas are placed correctly regardless of how many chunks are written
      per call to write_chunks(), including zero.
    - The file handle is always closed, including when an exception occurs
      mid-ingestion — __exit__ runs unconditionally as part of the `with`
      protocol, so no file handle is ever leaked. The exception itself is
      NOT suppressed; it still propagates after cleanup, which is what lets
      run_ingestion.py's per-lesson logging report exactly which lesson failed.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._wrote_any = False

    def __enter__(self) -> "ChunkJsonWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._file.write("[")
        return self

    def write_chunks(self, chunks: list[Chunk]) -> None:
        """Append this lesson's chunks to the array. Safe to call with an empty list."""
        for chunk in chunks:
            if self._wrote_any:
                self._file.write(",\n")
            else:
                self._file.write("\n")
            json.dump(_chunk_to_record(chunk), self._file, ensure_ascii=False, indent=2)
            self._wrote_any = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            if self._file is not None:
                # Close the array so the file is valid JSON even after a
                # mid-ingestion failure — better a truncated-but-valid
                # array of whatever succeeded than a corrupt file.
                self._file.write("\n]" if self._wrote_any else "]")
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
        return False  # never suppress the original exception


class FaissIndexBuilder:
    """
    Incremental counterpart to save_faiss_index(): create the index once,
    call add() per lesson as embeddings become available, write to disk
    once at the very end. IndexFlatIP natively supports repeated .add()
    calls, so no approximation or restructuring of the index type itself
    is needed — this only changes *when* embeddings are added.

    Usage:
        builder = FaissIndexBuilder(dim=embedder.embedding_dim)
        for lesson in lessons:
            embeddings = embedder.embed_passages([...])
            builder.add(embeddings)
        builder.save(path)
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine, vectors are normalized

    def add(self, embeddings: np.ndarray) -> None:
        """Add this lesson's embeddings. Safe to call with a (0, dim) empty array."""
        if embeddings.shape[0] == 0:
            return
        self.index.add(embeddings)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

