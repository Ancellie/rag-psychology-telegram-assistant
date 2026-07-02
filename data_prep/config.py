"""
Central configuration for the data preparation pipeline.
Keep every tunable in one place so nothing is hardcoded in the logic files.
"""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNKS_JSON_PATH = PROCESSED_DIR / "chunks.json"
FAISS_INDEX_PATH = PROCESSED_DIR / "index.faiss"

# --- Chunking parameters ---
# Token counts here are approximated by word count (simple, dependency-free,
# good enough at this stage). Swap for a real tokenizer later if needed.
#
# No markdown structure exists in the source files (plain subtitle-style
# transcripts) — chunking is purely sentence-based. See src/chunker.py.
TARGET_CHUNK_TOKENS = 380
CHUNK_OVERLAP_TOKENS = 60          # sliding-window overlap: keep within 40-80 tokens
MIN_CHUNK_TOKENS = 120             # don't emit/topic-break into chunks smaller than this

# Phrases that typically signal a topic shift in spoken/lecture-style text.
# A sentence starting with one of these is treated as a preferred chunk
# boundary once the current chunk has reached MIN_CHUNK_TOKENS — i.e. we
# prefer breaking on a topic shift over breaking mid-topic at the token limit.
TOPIC_SHIFT_MARKERS = [
    # English
    "first of all", "firstly", "secondly", "thirdly", "another important point",
    "another point", "next,", "next let's", "now let's", "let's move on",
    "let's talk about", "moving on", "on the other hand", "in contrast",
    "to summarize", "to sum up", "in summary", "let's turn to", "now, let's discuss",
    "the next topic", "let's now look at",
    # Russian
    "во-первых", "во-вторых", "в-третьих", "теперь поговорим", "теперь давайте",
    "перейдём к", "перейдем к", "следующий вопрос", "еще один важный момент",
    "ещё один важный момент", "стоит отметить", "с другой стороны", "подведём итог",
    "подведем итог", "в заключение", "далее рассмотрим",
    # Ukrainian
    "по-перше", "по-друге", "по-третє", "тепер поговоримо", "перейдімо до",
    "перейдемо до", "наступне питання", "ще один важливий момент",
    "варто зазначити", "з іншого боку", "підсумуємо", "на завершення",
]

# --- Embedding model ---
# Primary recommendation: multilingual, strong semantic search performance.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# e5 models require these prefixes on input text — baked into embedder.py
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

EMBEDDING_BATCH_SIZE = 16
