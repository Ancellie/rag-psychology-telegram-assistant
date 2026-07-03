"""
PromptBuilder: assembles a user query and retrieved chunks into a
provider-agnostic BuiltPrompt.

Responsibility: assembly and budget logic only. No prompt wording lives
here — see templates.py. No LLM calls, no Telegram, no retrieval logic.
Single-turn only: this milestone takes (query, chunks) and nothing else.
Conversation history is explicitly out of scope, to be added as a
separate layer later without reshaping this one.

Dependencies: src.retrieval.models.RetrievedChunk (read-only) and config
for MAX_CONTEXT_TOKENS. Nothing else.
"""

import config
from ..retrieval.models import RetrievedChunk
from .models import BuiltPrompt
from . import templates


class PromptBuilder:
    """
    Turns (query, ranked chunks) into a BuiltPrompt ready for an LLM
    adapter to format for a specific provider.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(query, retriever.search(query))
    """

    def __init__(self, max_context_tokens: int | None = None):
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else config.MAX_CONTEXT_TOKENS
        )

    def build(self, query: str, chunks: list[RetrievedChunk]) -> BuiltPrompt:
        """
        Build the prompt. chunks is assumed to already be ranked by
        descending similarity (as returned by Retriever.search) — this
        method does not re-sort.
        """
        if not chunks:
            return BuiltPrompt(
                system=templates.SYSTEM_PROMPT_TEMPLATE,
                context=templates.NO_CONTEXT_NOTICE,
                user=query,
                chunks_used=[],
                truncated=False,
            )

        included, truncated = self._select_within_budget(chunks)
        context = templates.CONTEXT_CHUNK_SEPARATOR.join(
            templates.CONTEXT_CHUNK_TEMPLATE.format(
                lesson_title=c.lesson_title, text=c.text
            )
            for c in included
        )

        return BuiltPrompt(
            system=templates.SYSTEM_PROMPT_TEMPLATE,
            context=context,
            user=query,
            chunks_used=[c.chunk_id for c in included],
            truncated=truncated,
        )

    def _select_within_budget(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        """
        Walk chunks in the given (rank) order, accumulating token_count.
        The top-ranked chunk is always included, even if it alone exceeds
        the budget — an ungrounded empty context is worse than one
        slightly-over-budget grounded chunk. After that, include each
        subsequent chunk only while it still fits; stop at the first one
        that doesn't (tail-drop), since chunks are already ranked by
        relevance and re-ordering to fit smaller ones later would silently
        override that ranking.
        """
        included: list[RetrievedChunk] = [chunks[0]]
        running_total = chunks[0].token_count

        cutoff = len(chunks)
        for i, chunk in enumerate(chunks[1:], start=1):
            if running_total + chunk.token_count > self.max_context_tokens:
                cutoff = i
                break
            included.append(chunk)
            running_total += chunk.token_count

        truncated = cutoff < len(chunks)
        return included, truncated
