"""
Standalone smoke test for the full RAG pipeline.

Builds the default pipeline (real Retriever + PromptBuilder + BaseLLM,
wired from config.py) and runs a handful of hardcoded psychology
questions end to end: query -> retrieved chunks -> built prompt ->
generated Answer. This is the first test that exercises all three layers
together — test_retrieval.py and test_llm.py each cover one layer with
the others faked or hand-built.

Usage:
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import build_default_pipeline

TEST_QUESTIONS = [
    "What is cognitive dissonance?",
    "How do defense mechanisms work?",
    "What is the difference between classical and operant conditioning?",
    "What are the stages of grief?",
]


def print_answer(question: str, answer) -> None:
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    print(f"ANSWER:\n{answer.text}\n")
    print(f"model_name:        {answer.model_name}")
    print(f"latency_seconds:   {answer.latency_seconds:.2f}")
    print(f"prompt_tokens:     {answer.prompt_tokens}")
    print(f"completion_tokens: {answer.completion_tokens}")
    print(f"chunks_used:       {answer.chunks_used}")
    print(f"truncated:         {answer.truncated}")
    print()


def main() -> None:
    print("Building default RAG pipeline (embedder, retriever, LLM)...")
    pipeline = build_default_pipeline()
    print("Pipeline ready.\n")

    for question in TEST_QUESTIONS:
        answer = pipeline.answer(question)
        print_answer(question, answer)


if __name__ == "__main__":
    main()
