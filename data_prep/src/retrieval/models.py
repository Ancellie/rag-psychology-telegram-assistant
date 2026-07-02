"""
Domain model for retrieval results.

RetrievedChunk is the contract between the retrieval layer and any future
consumer (LLM prompt builder, Telegram bot, CLI). Consumers depend only on
this dataclass — never on FAISS or chunks.json directly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by Retriever.search(), with its retrieval rank and score."""

    chunk_id: str
    lesson_id: str
    lesson_title: str
    source_file: str
    chunk_index: int
    token_count: int
    boundary_reason: str
    text: str
    similarity: float
    rank: int
