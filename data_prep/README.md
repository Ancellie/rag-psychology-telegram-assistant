# Psychology Course – Data Preparation Pipeline

Stage 1 of the RAG system: turns 243 raw markdown lesson files into
sentence-safe, metadata-tagged chunks with embeddings, ready to be
loaded into a production vector DB (Qdrant) in a later stage.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Put all 243 `.md` lesson files into `data/raw/`.
2. Run:

```bash
python scripts/run_ingestion.py
```

3. Output:
   - `data/processed/chunks.json` — chunk text + metadata (human-readable, portable)
   - `data/processed/index.faiss` — embeddings as a FAISS flat index (staging only)

## Design notes

- **Source files have no markdown structure** (plain subtitle/transcript text) —
  there is no header-based splitting stage. Chunking is purely sentence-based:
  1. `cleaner.py` strips timestamps/noise and collapses subtitle line-wraps
     into continuous prose (single `\n` = line wrap, not a real break).
  2. `chunker.py` splits into sentences, then slides a token window over them.
     A chunk boundary is placed either (a) at a **topic-shift sentence** —
     one starting with a marker phrase like "first of all" / "во-первых" /
     "another important point" (see `TOPIC_SHIFT_MARKERS` in `config.py`) —
     once the chunk has reached `MIN_CHUNK_TOKENS`, or (b) at the token
     limit if no topic shift occurs first. This is what makes chunking
     topic-aware rather than purely size-based.
  3. Sliding-window overlap (40–80 tokens, configurable) is kept between
     consecutive chunks so context isn't lost at a boundary.
  4. No LLM is used anywhere in this stage — it's rule-based end to end.
- Each chunk records a `boundary_reason` (`topic_shift`, `token_limit`, or
  `end_of_lesson`) so you can audit boundary quality in `chunks.json` before
  trusting it at scale.
- **Known trade-off:** topic-shift detection here is marker-phrase-based, not
  embedding-similarity-based (e.g. TextTiling). That avoids loading a second
  model during chunking. If you inspect real output and find topic drift
  within chunks that marker phrases don't catch, upgrading `_chunk_sentences`
  in `chunker.py` to use sentence-embedding similarity is a contained change —
  it doesn't touch the loader, storage, or embedding stage.
- Embedding model: `intfloat/multilingual-e5-large` (see `config.py` to swap for
  `BAAI/bge-m3` if you need longer-context chunks or hybrid search later).
- FAISS here is a **local staging index**, not the production vector store.
  Qdrant will be introduced in the retrieval stage — this pipeline just needs
  to produce clean, embedded, well-tagged chunks that Qdrant can be loaded from.

## Explicitly out of scope for this stage

- Retrieval logic / query embedding
- Qdrant integration
- LLM answer generation
- Telegram bot

These are separate stages, built next.
