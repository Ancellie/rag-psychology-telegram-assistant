"""
Loader: reads raw markdown lesson files from disk and extracts
basic per-lesson metadata (lesson_id, lesson_title, raw text).

Responsibility: I/O only. No cleaning, no chunking here.
"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class RawLesson:
    lesson_id: str       # derived from filename, e.g. "042"
    lesson_title: str    # derived from filename (no markdown headers exist in source files)
    source_file: str     # filename for traceability
    raw_text: str


def _extract_lesson_id(filename: str) -> str:
    """Pull a lesson id from the filename (e.g. '042_defense-mechanisms.md' -> '042')."""
    match = re.match(r"^(\d+)", filename)
    return match.group(1) if match else filename.rsplit(".", 1)[0]


def _derive_title_from_filename(stem: str) -> str:
    """
    Files are plain transcripts with no markdown headers, so there is no
    in-text title to extract. Derive a human-readable title from the
    filename instead: '042_defense-mechanisms' -> 'Defense Mechanisms'.
    """
    without_id = re.sub(r"^\d+[_\-\s]*", "", stem)
    words = re.split(r"[_\-]+", without_id.strip())
    words = [w for w in words if w]
    return " ".join(w.capitalize() for w in words) if words else stem


def load_markdown_files(raw_dir: Path) -> list[RawLesson]:
    """
    Read every .md file in raw_dir and return a list of RawLesson objects.
    Raises a clear error if the directory is empty so failures are obvious early.
    """
    files = sorted(raw_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No .md files found in {raw_dir}. "
            f"Place your 243 lesson files there before running ingestion."
        )

    lessons = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lesson_id = _extract_lesson_id(file_path.stem)
        title = _derive_title_from_filename(file_path.stem)
        lessons.append(
            RawLesson(
                lesson_id=lesson_id,
                lesson_title=title,
                source_file=file_path.name,
                raw_text=text,
            )
        )
    return lessons
