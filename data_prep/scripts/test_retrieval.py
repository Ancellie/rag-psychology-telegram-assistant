"""
Standalone smoke test for the retrieval layer.

Loads the embedder and retriever, runs a handful of hardcoded psychology
questions, and prints the ranked results. No LLM, no Telegram — this only
verifies that query -> chunks works end to end.

Usage:
    python scripts/test_retrieval.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.embedder import Embedder
from src.retrieval import Retriever

TEST_QUESTIONS = [
    "How to overcome anxiety?",
    "How to deal with difficult people?",
    "How to improve self-esteem?",
    "How to stop procrastinating?",
    "How to become a better leader?",
    "How to control anger?",
    "How to change negative beliefs?",
    "How to overcome fear?",
    "How to reduce stress?",
    "What is the ego?",
    "What is the unconscious mind?",
    "What is attachment?",
]


def print_results(query: str, results: list) -> None:
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)
    if not results:
        print("  (no results)")
        return
    for r in results:
        preview = r.text[:160].replace("\n", " ")
        print(f"  Rank {r.rank} | similarity={r.similarity:.4f}")
        print(f"    Lesson:  {r.lesson_title}")
        print(f"    Chunk:   {r.chunk_id}")
        print(f"    Preview: {preview}...")
        print()


def main() -> None:
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embedder = Embedder(
        model_name=config.EMBEDDING_MODEL_NAME,
        passage_prefix=config.E5_PASSAGE_PREFIX,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )

    retriever = Retriever(
        embedder=embedder,
        chunks_json_path=config.CHUNKS_JSON_PATH,
        faiss_index_path=config.FAISS_INDEX_PATH,
        default_top_k=5,
    )

    for question in TEST_QUESTIONS:
        results = retriever.search(question)
        print_results(question, results)


if __name__ == "__main__":
    main()
