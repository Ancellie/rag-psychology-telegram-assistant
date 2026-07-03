"""
Domain model for the prompt-building layer.

BuiltPrompt is the contract between PromptBuilder and any future consumer
(an LLM adapter, first and foremost). Consumers depend only on this
dataclass — never on templates.py or the truncation logic directly. This
mirrors how RetrievedChunk (src/retrieval/models.py) is the contract
between Retriever and PromptBuilder.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltPrompt:
    """
    A fully assembled, provider-agnostic prompt.

    system, context, and user are kept as separate fields rather than one
    pre-joined string. This is deliberate: a future LLM adapter needs to
    map these onto whatever a specific provider's API expects (e.g. a
    single system message vs. a system message plus a user message with
    context prepended vs. a tool-call-style context injection). Pre-joining
    here would force that decision prematurely and leak provider-specific
    formatting into a provider-independent layer.
    """

    system: str            # instructor persona + grounding rules (from templates.py)
    context: str            # formatted retrieved chunks, or the no-context notice
    user: str                # the raw user query, unmodified
    chunks_used: list[str]  # chunk_ids actually included in `context`
    truncated: bool          # True if lower-ranked chunks were dropped for budget
