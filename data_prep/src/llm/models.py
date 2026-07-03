"""
Domain model for the LLM layer.

Answer is the contract between LLMClient implementations and any future
consumer (CLI, Telegram bot, web layer). Consumers depend only on this
dataclass — never on a specific backend's response format. Mirrors how
RetrievedChunk (src/retrieval/models.py) and BuiltPrompt (src/prompt/models.py)
are the contracts for the layers before this one.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    """
    The normalized output of any LLMClient.generate() call, regardless of
    backend.

    prompt_tokens / completion_tokens are optional because not every
    backend can report them cheaply or at all (e.g. some remote APIs
    don't return usage data unless explicitly requested). They default
    to None rather than 0, so "unknown" is never confused with "zero
    tokens used" — callers that display/log these must handle None.
    """

    text: str                          # the model's answer
    chunks_used: list[str]             # passthrough from BuiltPrompt, for citation/debugging
    truncated: bool                    # passthrough from BuiltPrompt
    model_name: str                    # e.g. "huggingface:Qwen/Qwen3-8B" — backend + model
    latency_seconds: float             # wall-clock time for the generate() call
    prompt_tokens: int | None = None       # input token count, if the backend reports it
    completion_tokens: int | None = None   # output token count, if the backend reports it
