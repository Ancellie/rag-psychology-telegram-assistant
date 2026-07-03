"""
RAGPipeline: composes Retriever, PromptBuilder, and BaseLLM into a single
end-to-end service: question in, Answer out.

Responsibility: orchestration only. This class does not retrieve, does
not build prompts, does not generate text — it delegates each step to
the layer that already owns it, in order:

    query -> Retriever.search() -> RetrievedChunk[]
          -> PromptBuilder.build() -> BuiltPrompt
          -> BaseLLM.generate() -> Answer

No provider-specific code, no Telegram-specific code, no retrieval or
prompt-assembly logic lives here. RAGPipeline only ever depends on the
public contracts of the three layers it composes (Retriever, PromptBuilder,
BaseLLM) — it never imports a concrete LLM backend, FAISS, or embedder
internals directly. Swapping the LLM backend (Hugging Face -> Groq, or
turning quantization on/off) requires no change to this file, since it
only ever holds something typed as BaseLLM.

Dependencies are injected via the constructor, same DI discipline as
Retriever (embedder passed in) and PromptBuilder (max_context_tokens
passed in). build_default_pipeline() is a convenience constructor that
wires the real components from config.py, so callers (CLI, bot) don't
have to repeat that wiring — it does not change the class's DI contract,
it's just one way of satisfying it.
"""

import config
from ..embedder import Embedder
from ..llm.client import BaseLLM
from ..llm.factory import get_llm
from ..prompt.builder import PromptBuilder
from ..retrieval.retriever import Retriever
from ..llm.models import Answer


class RAGPipeline:
    """
    Usage:
        pipeline = build_default_pipeline()
        answer = pipeline.answer("What is cognitive dissonance?")

    Or with explicit dependencies (e.g. for testing with fakes):
        pipeline = RAGPipeline(retriever=fake_retriever,
                                prompt_builder=fake_builder,
                                llm=fake_llm)
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm

    def answer(self, query: str, top_k: int | None = None) -> Answer:
        """
        Run the full retrieve -> build prompt -> generate chain for one
        question. top_k is passed straight through to Retriever.search()
        (None uses the retriever's own default_top_k).

        Raises whatever the underlying layers raise — this method adds
        no error handling of its own. Callers (CLI, bot) decide how to
        surface a failure to the user, same as with BaseLLM.generate().
        """
        retrieved_chunks = self.retriever.search(query, top_k=top_k)
        prompt = self.prompt_builder.build(query, retrieved_chunks)
        return self.llm.generate(prompt)


def build_default_pipeline(cfg=config) -> RAGPipeline:
    """
    Convenience constructor: wires the real Retriever, PromptBuilder, and
    BaseLLM (via get_llm(), which reads cfg.LLM_PROVIDER) from config.py.

    This is the only place outside tests that should construct these
    three components directly — CLI, bot, and other future callers should
    call this function rather than repeating the wiring.
    """
    embedder = Embedder(
        model_name=cfg.EMBEDDING_MODEL_NAME,
        passage_prefix=cfg.E5_PASSAGE_PREFIX,
        batch_size=cfg.EMBEDDING_BATCH_SIZE,
    )
    retriever = Retriever(
        embedder=embedder,
        chunks_json_path=cfg.CHUNKS_JSON_PATH,
        faiss_index_path=cfg.FAISS_INDEX_PATH,
    )
    prompt_builder = PromptBuilder(max_context_tokens=cfg.MAX_CONTEXT_TOKENS)
    llm = get_llm(cfg)

    return RAGPipeline(retriever=retriever, prompt_builder=prompt_builder, llm=llm)
