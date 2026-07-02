"""
Chunker: splits cleaned lesson text into semantic, sentence-safe chunks.

The source files have NO markdown structure (plain subtitle/transcript text),
so there is nothing to split on structurally. Chunking works purely off the
sentence stream:

  1. Split the full lesson text into sentences.
  2. Slide a token-window over sentences, never cutting mid-sentence.
  3. Prefer breaking at a topic-shift sentence (one starting with a marker
     phrase like "first of all" / "во-первых" / "another important point")
     once the chunk has reached MIN_CHUNK_TOKENS, rather than always
     breaking exactly at TARGET_CHUNK_TOKENS — this is what makes chunking
     topic-aware rather than purely size-based.
  4. Apply sliding-window overlap between consecutive chunks.

Token counts are approximated by whitespace word count — a deliberate,
dependency-free simplification. Swap in a real tokenizer later if exact
sizing becomes important.
"""

from dataclasses import dataclass, field
import re
import uuid

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Chunk:
    chunk_id: str
    lesson_id: str
    lesson_title: str
    source_file: str
    chunk_index: int
    token_count: int
    text: str
    boundary_reason: str  # "topic_shift", "token_limit", or "end_of_lesson"
    metadata: dict = field(default_factory=dict)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    # Paragraph breaks (from cleaner.py) still count as sentence separators
    text = text.replace("\n\n", " ")
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _starts_with_topic_marker(sentence: str, markers: list[str]) -> bool:
    lowered = sentence.strip().lower()
    return any(lowered.startswith(marker) for marker in markers)


def _chunk_sentences(
    sentences: list[str],
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    topic_markers: list[str],
) -> list[tuple[str, str]]:
    """
    Returns a list of (chunk_text, boundary_reason) tuples.
    """
    if not sentences:
        return []

    results: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    # FIX: track whether at least one new sentence has been appended since the
    # last chunk boundary.  A boundary may only be emitted when this is True,
    # which guarantees that every iteration of the outer loop either (a) emits
    # a chunk that contains at least one sentence that was not in the previous
    # chunk's overlap window, or (b) appends a sentence and advances i.
    # Without this guard, a sentence whose word-count exceeds
    # (target_tokens − overlap_tokens) causes an infinite loop: the overlap
    # window is re-emitted as a chunk, the identical overlap is rebuilt, and
    # the oversized sentence is re-evaluated forever without i ever advancing.
    new_sentence_added = False

    i = 0
    n = len(sentences)
    while i < n:
        sentence = sentences[i]
        sentence_len = _word_count(sentence)
        is_topic_shift = _starts_with_topic_marker(sentence, topic_markers)

        should_break_for_topic = (
            is_topic_shift and current and current_len >= min_chunk_tokens
        )
        should_break_for_size = (
            current and (current_len + sentence_len > target_tokens)
        )

        if (should_break_for_topic or should_break_for_size) and new_sentence_added:
            reason = "topic_shift" if should_break_for_topic else "token_limit"
            results.append((" ".join(current), reason))

            # Build overlap window from the tail of the chunk just closed
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                s_len = _word_count(s)
                if overlap_len + s_len > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += s_len

            current = overlap_sentences
            current_len = overlap_len
            new_sentence_added = False  # reset: need a fresh sentence before next break
            continue  # re-evaluate the same sentence against the fresh window

        current.append(sentence)
        current_len += sentence_len
        new_sentence_added = True
        i += 1

    if current:
        results.append((" ".join(current), "end_of_lesson"))

    return results


def build_chunks(
    lesson_id: str,
    lesson_title: str,
    source_file: str,
    cleaned_text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    topic_markers: list[str],
) -> list[Chunk]:
    sentences = _split_sentences(cleaned_text)
    raw_chunks = _chunk_sentences(
        sentences, target_tokens, overlap_tokens, min_chunk_tokens, topic_markers
    )

    chunks: list[Chunk] = []
    for idx, (text, reason) in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                lesson_id=lesson_id,
                lesson_title=lesson_title,
                source_file=source_file,
                chunk_index=idx,
                token_count=_word_count(text),
                text=text,
                boundary_reason=reason,
            )
        )
    return chunks