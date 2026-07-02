"""
Entrypoint: runs the full data preparation pipeline end to end, streaming.

For each lesson, one at a time:
    read -> clean -> chunk -> embed -> persist -> release

No list ever holds more than one lesson's worth of data. Persistence is
incremental (ChunkJsonWriter, FaissIndexBuilder) so nothing is buffered
across lessons. A failure on one lesson is logged and skipped; ingestion
continues with the rest.

Usage:
    python scripts/run_ingestion.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from src.loader import iter_markdown_files, RawLesson
from src.cleaner import clean_text
from src.chunker import build_chunks
from src.embedder import Embedder
from src.storage import ChunkJsonWriter, FaissIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingestion")

STAGES = ("loading", "cleaning", "chunking", "embedding", "storage")


class StageTimer:
    """Lightweight accumulator for per-stage timing across the whole run."""

    def __init__(self):
        self.totals = {stage: 0.0 for stage in STAGES}

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] += seconds

    def report(self) -> str:
        return ", ".join(f"{k}={v:.2f}s" for k, v in self.totals.items())


def count_lesson_files(raw_dir: Path) -> int:
    """
    Cheap upfront count for progress display ('3/243'). Only counts Path
    objects from glob — does not read file contents, so this does not
    reintroduce full-corpus memory usage.
    """
    return sum(1 for _ in raw_dir.glob("*.md"))


def process_lesson(
    lesson: RawLesson,
    embedder: Embedder,
    chunk_writer: ChunkJsonWriter,
    faiss_builder: FaissIndexBuilder,
    timer: StageTimer,
) -> int:
    """
    Run one lesson through clean -> chunk -> embed -> persist.
    Returns the number of chunks produced. Raises on failure — the caller
    decides how to handle/log/continue, keeping this function a pure
    single-lesson unit (easy to test in isolation, e.g. with a fake embedder).
    """
    t0 = time.perf_counter()
    cleaned = clean_text(lesson.raw_text)
    timer.add("cleaning", time.perf_counter() - t0)

    t0 = time.perf_counter()
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
    timer.add("chunking", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if chunks:
        embeddings = embedder.embed_passages([c.text for c in chunks])
    else:
        embeddings = np.empty((0, embedder.embedding_dim), dtype="float32")
    timer.add("embedding", time.perf_counter() - t0)

    t0 = time.perf_counter()
    chunk_writer.write_chunks(chunks)
    faiss_builder.add(embeddings)
    timer.add("storage", time.perf_counter() - t0)

    return len(chunks)


def run_pipeline(
    raw_dir: Path,
    embedder: Embedder,
    chunks_json_path: Path,
    faiss_index_path: Path,
) -> dict:
    """
    Streaming ingestion loop: one lesson resident in memory at a time.
    Returns a summary dict. Never raises on a single bad lesson — logs
    and continues. Returns/writes files even if some lessons failed.
    """
    run_start = time.perf_counter()
    timer = StageTimer()

    total_lessons = count_lesson_files(raw_dir)
    if total_lessons == 0:
        raise FileNotFoundError(
            f"No .md files found in {raw_dir}. "
            f"Place your lesson files there before running ingestion."
        )

    logger.info(f"Found {total_lessons} lesson files in {raw_dir}")
    logger.info("=" * 60)
    logger.info("INGESTION STARTED")
    logger.info(f"Embedding model : {config.EMBEDDING_MODEL_NAME}")
    logger.info(f"Embedding dim   : {embedder.embedding_dim}")
    logger.info(f"Batch size      : {embedder.batch_size}")
    logger.info(f"Raw directory   : {raw_dir}")
    logger.info(f"Output chunks   : {chunks_json_path}")
    logger.info(f"Output FAISS    : {faiss_index_path}")
    logger.info("=" * 60)

    processed_ok = 0
    failed = 0
    total_chunks = 0
    failed_lessons: list[tuple[str, str, str]] = []  # (filename, exc_type, exc_msg)

    lesson_iterator = iter_markdown_files(raw_dir)
    lesson_index = 0

    with ChunkJsonWriter(chunks_json_path) as chunk_writer:
        faiss_builder = FaissIndexBuilder(dim=embedder.embedding_dim)

        while True:
            t_load_start = time.perf_counter()
            try:
                lesson = next(lesson_iterator)
            except StopIteration:
                break
            except Exception as e:
                # A file-read failure inside the generator itself (rare: e.g.
                # permissions, file removed mid-run). Logged and skipped —
                # note the underlying generator may be exhausted after this,
                # in which case the loop simply ends on the next iteration.
                failed += 1
                failed_lessons.append(("<unreadable file>", type(e).__name__, str(e)))
                logger.error(f"FAILED to read next lesson file: {type(e).__name__}: {e}")
                continue
            timer.add("loading", time.perf_counter() - t_load_start)

            lesson_index += 1
            elapsed = time.perf_counter() - run_start
            logger.info(
                f"[{lesson_index}/{total_lessons}] "
                f"Lesson {lesson.lesson_id} | "
                f"{lesson.source_file} | "
                f"Elapsed: {elapsed:.1f}s"
            )

            try:
                chunk_count = process_lesson(
                    lesson, embedder, chunk_writer, faiss_builder, timer
                )
                logger.info(
                    f"✓ Created {chunk_count} chunks "
                    f"(running total: {total_chunks + chunk_count})"
                )
                total_chunks += chunk_count
                processed_ok += 1
            except Exception as e:
                failed += 1
                failed_lessons.append((lesson.source_file, type(e).__name__, str(e)))
                logger.error(
                    f"FAILED lesson '{lesson.source_file}': {type(e).__name__}: {e}"
                )
            # lesson/cleaned/chunks/embeddings all fall out of scope at the
            # top of the next loop iteration — nothing is retained.

        logger.info("Writing FAISS index to disk...")
        faiss_builder.save(faiss_index_path)

    total_elapsed = time.perf_counter() - run_start

    summary = {
        "total_lessons": total_lessons,
        "processed_ok": processed_ok,
        "failed": failed,
        "total_chunks": total_chunks,
        "total_elapsed_seconds": total_elapsed,
        "stage_totals": dict(timer.totals),
        "failed_lessons": failed_lessons,
    }
    return summary


def print_summary(summary: dict, chunks_json_path: Path, faiss_index_path: Path) -> None:
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info(f"  Total lessons found:     {summary['total_lessons']}")
    logger.info(f"  Processed successfully:  {summary['processed_ok']}")
    logger.info(f"  Failed:                  {summary['failed']}")
    logger.info(f"  Total chunks produced:   {summary['total_chunks']}")
    logger.info(f"  Total elapsed time:      {summary['total_elapsed_seconds']:.1f}s")
    stage_str = ", ".join(f"{k}={v:.2f}s" for k, v in summary["stage_totals"].items())
    logger.info(f"  Stage time breakdown:    {stage_str}")
    if summary["failed_lessons"]:
        logger.info("  Failed lessons:")
        for fname, exc_type, exc_msg in summary["failed_lessons"]:
            logger.info(f"    - {fname}: {exc_type}: {exc_msg}")
    logger.info("=" * 60)
    logger.info(f"Chunks saved to {chunks_json_path}")
    logger.info(f"FAISS index saved to {faiss_index_path}")
    avg = summary["total_chunks"] / max(summary["processed_ok"], 1)

    logger.info(f"  Average chunks/lesson:   {avg:.1f}")


def main() -> None:
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embedder = Embedder(
        model_name=config.EMBEDDING_MODEL_NAME,
        passage_prefix=config.E5_PASSAGE_PREFIX,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    logger.info(
        f"Embedding model loaded (device={embedder.device}, dim={embedder.embedding_dim})"
    )

    summary = run_pipeline(
        raw_dir=config.RAW_DIR,
        embedder=embedder,
        chunks_json_path=config.CHUNKS_JSON_PATH,
        faiss_index_path=config.FAISS_INDEX_PATH,
    )
    print_summary(summary, config.CHUNKS_JSON_PATH, config.FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()
