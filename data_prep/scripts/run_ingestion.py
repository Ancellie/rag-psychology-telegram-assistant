"""
Entrypoint: runs the full data preparation pipeline end to end.

Usage:
    python scripts/run_ingestion.py

Reads all .md files from data/raw, cleans + chunks + embeds them,
and writes data/processed/chunks.json + data/processed/index.faiss.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.loader import load_markdown_files
from src.cleaner import clean_text
from src.chunker import build_chunks
from src.embedder import Embedder
from src.storage import save_chunks_json, save_faiss_index


def main() -> None:
    print(f"Loading lessons from {config.RAW_DIR} ...")
    lessons = load_markdown_files(config.RAW_DIR)
    print(f"Loaded {len(lessons)} lesson files.")

    all_chunks = []
    for index, lesson in enumerate(lessons, 1):
        print(f"Processing {index}/{len(lessons)}: {lesson.lesson_title}")
        cleaned = clean_text(lesson.raw_text)
        chunks = build_chunks(
            lesson_id=lesson.lesson_id,
            lesson_title=lesson.lesson_title,
            source_file=lesson.source_file,
            cleaned_text=cleaned,
            target_tokens=config.TARGET_CHUNK_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
            min_chunk_tokens=config.MIN_CHUNK_TOKENS,
            topic_markers=config.TOPIC_SHIFT_MARKERS,
        )
        all_chunks.extend(chunks)

    print(f"Produced {len(all_chunks)} chunks from {len(lessons)} lessons.")

    if not all_chunks:
        print("No chunks produced — check raw files. Exiting.")
        return

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embedder = Embedder(
        model_name=config.EMBEDDING_MODEL_NAME,
        passage_prefix=config.E5_PASSAGE_PREFIX,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_passages(texts)
    print(f"Generated embeddings: shape={embeddings.shape}")

    save_chunks_json(all_chunks, config.CHUNKS_JSON_PATH)
    save_faiss_index(embeddings, config.FAISS_INDEX_PATH)

    print(f"Saved chunk metadata to {config.CHUNKS_JSON_PATH}")
    print(f"Saved FAISS index to {config.FAISS_INDEX_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
