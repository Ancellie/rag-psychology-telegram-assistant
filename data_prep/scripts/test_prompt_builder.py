import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

from src.embedder import Embedder
from src.retrieval import Retriever
from src.prompt.builder import PromptBuilder

embedder = Embedder(
    model_name=config.EMBEDDING_MODEL_NAME,
    passage_prefix=config.E5_PASSAGE_PREFIX,
    batch_size=config.EMBEDDING_BATCH_SIZE,
)

retriever = Retriever(
    embedder=embedder,
    chunks_json_path=config.CHUNKS_JSON_PATH,
    faiss_index_path=config.FAISS_INDEX_PATH,
)

builder = PromptBuilder(max_context_tokens=config.MAX_CONTEXT_TOKENS)

query = "How to overcome anxiety?"

chunks = retriever.search(query)

prompt = builder.build(query, chunks)

print("=" * 80)
print("SYSTEM")
print("=" * 80)
print(prompt.system)

print("\n" + "=" * 80)
print("CONTEXT")
print("=" * 80)
print(prompt.context)

print("\n" + "=" * 80)
print("USER")
print("=" * 80)
print(prompt.user)

print("\nChunks used:", prompt.chunks_used)
print("Truncated:", prompt.truncated)