"""
Retriever: embeds a user query, searches the FAISS index, and returns the
matching chunks as typed RetrievedChunk objects.

Independent of Telegram, any LLM provider, prompt construction, and
conversation history — its only job is query -> ranked chunks.
"""

from pathlib import Path

import faiss

from ..embedder import Embedder
from ..storage import load_chunks_json
from .models import RetrievedChunk


class Retriever:
    """
    Loads a FAISS index and its aligned chunks.json once at construction,
    then serves repeated search() calls against them.

    Dependencies (embedder, paths) are passed in explicitly rather than
    constructed internally, so this class can be tested with a mock
    embedder and reused by any caller (CLI, bot, LLM layer) unchanged.
    """

    def __init__(
        self,
        embedder: Embedder,
        chunks_json_path: Path,
        faiss_index_path: Path,
        default_top_k: int = 5,
    ):
        self.embedder = embedder
        self.default_top_k = default_top_k

        self.index = faiss.read_index(str(faiss_index_path))
        self.chunks = load_chunks_json(chunks_json_path)

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"FAISS index and chunks.json are out of sync: "
                f"index has {self.index.ntotal} vectors, "
                f"chunks.json has {len(self.chunks)} records "
                f"({faiss_index_path} vs {chunks_json_path})."
            )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed the query, search the FAISS index, and return the matching
        chunks ordered by descending similarity.

        top_k defaults to self.default_top_k. FAISS pads results with
        index -1 when fewer than top_k vectors exist in the index; those
        padding rows are dropped rather than mapped back to chunks.json.
        """
        k = top_k if top_k is not None else self.default_top_k
        query_vector = self.embedder.embed_query(query)

        similarities, indices = self.index.search(query_vector, k)

        results: list[RetrievedChunk] = []
        for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0]), start=1):
            if idx == -1:
                continue
            record = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=record["chunk_id"],
                    lesson_id=record["lesson_id"],
                    lesson_title=record["lesson_title"],
                    source_file=record["source_file"],
                    chunk_index=record["chunk_index"],
                    token_count=record["token_count"],
                    boundary_reason=record["boundary_reason"],
                    text=record["text"],
                    similarity=float(similarity),
                    rank=rank,
                )
            )

        return results
