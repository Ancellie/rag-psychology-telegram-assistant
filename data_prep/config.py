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


# --- Prompt building ---
# Token budget (word-count approximation, same convention as chunker.py) for
# how much retrieved chunk text can be included in a single prompt's context.
MAX_CONTEXT_TOKENS = 3000


# --- LLM (Hugging Face backend, first implementation) ---
LLM_PROVIDER = "huggingface"  # only "huggingface" implemented so far

HF_LLM_MODEL_NAME = "Qwen/Qwen3-8B"
HF_LLM_DEVICE = None                  # None -> auto-detect via src.utils.device.auto_device()
HF_LLM_TORCH_DTYPE = "auto"
HF_LLM_MAX_NEW_TOKENS = 512
HF_LLM_TEMPERATURE = 0.3
HF_LLM_DO_SAMPLE = True
HF_LLM_ENABLE_THINKING = False        # RAG wants a direct grounded answer, not a <think> block

# Quantization mode: "4bit" | "8bit" | "none" (full precision).
# Default is 4-bit NF4 — sized for a 12GB RTX 4070; full FP16 (~16.4GB
# weights alone) does not fit. Built into a BitsAndBytesConfig by
# src/llm/quantization.py — never hardcoded in LocalHFModel itself, so
# switching precision is a change to this file only.
HF_LLM_QUANTIZATION = "4bit"
HF_LLM_BNB_4BIT_QUANT_TYPE = "nf4"
HF_LLM_BNB_4BIT_USE_DOUBLE_QUANT = True
# compute dtype is not set here: bfloat16 if the GPU supports it (Ampere/Ada+,
# which covers the RTX 4070), float16 otherwise — decided at load time in
# quantization.py since it depends on the actual GPU in use, not a static setting. # RAG wants a direct grounded answer, not a <think> block


# --- Debugging ---
# Master switch plus one flag per pipeline stage, so a developer can enable
# exactly the layer they're diagnosing (e.g. only DEBUG_RETRIEVAL while
# tuning chunk relevance) without being flooded by the others. All debug
# output is formatted and printed by src/debug.py — nothing outside that
# module ever calls print() for debugging purposes. When every flag below
# is False, each call site does one boolean check and nothing else: no
# extra allocation, no extra I/O, no behavior change.
#
# DEBUG is a convenience master switch: if True, it forces every
# per-stage flag on regardless of their individual values (see
# src/debug.py:is_enabled()). Leave DEBUG=False and toggle the specific
# stage flags for targeted debugging instead.
DEBUG = True
DEBUG_RETRIEVAL = True   # query + retrieved chunks (id, lesson, score, preview)
DEBUG_PROMPT = True      # exact system/context/user text sent to the LLM
DEBUG_GENERATION = True  # raw model output + final Answer + token counts
DEBUG_TIMINGS = True     # per-stage wall-clock time (retrieval/prompt/generation/total)