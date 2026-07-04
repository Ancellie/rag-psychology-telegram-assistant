"""
Debug utility: the single place in the project that formats, gates, and
prints debug output for the RAG pipeline.

Responsibility: presentation AND the enable/disable decision. Every
debug_* function here checks its own config.DEBUG_* flag (or the master
config.DEBUG switch) internally and returns immediately if it's off.
Callers (Retriever, PromptBuilder, LocalHFModel, RAGPipeline) never
inspect config themselves for this purpose — they just call
`debug_retrieval(...)`, `debug_prompt(...)`, etc. unconditionally. This
keeps every enable/disable rule in exactly one file: turning a flag on or
off, or changing what "enabled" means (e.g. the DEBUG master switch),
never requires touching retriever.py, builder.py, huggingface_llm.py, or
pipeline.py.

This module never fetches data itself; it only formats objects that
already exist at the caller's call site (RetrievedChunk, BuiltPrompt,
Answer, a plain timings dict). This mirrors templates.py owning prompt
*wording* while builder.py owns prompt *assembly* — here, debug.py owns
both the debug *decision* and the debug *formatting*.

Usage (identical shape at every call site, no config import needed):

    debug_retrieval(query, results)

When the corresponding flag is False, this call still happens but is a
single attribute lookup plus an early return — no data is copied, no
string is formatted, nothing is printed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    # Import only for type hints — avoids debug.py creating a hard runtime
    # dependency between otherwise-independent layers (retrieval/prompt/llm).
    from .retrieval.models import RetrievedChunk
    from .prompt.models import BuiltPrompt
    from .llm.models import Answer


_WIDTH = 24
_SECTION_RULE = "=" * _WIDTH
_SUBSECTION_RULE = "-" * _WIDTH
_PREVIEW_CHARS = 300
_CONTEXT_PREVIEW_CHARS = 500


def is_enabled(cfg, *flag_names: str) -> bool:
    """
    True if cfg.DEBUG (master switch) or any of the named per-stage flags
    is set. A missing attribute is treated as False, not an error, so
    debug.py never breaks a config that predates a given flag.
    """
    if getattr(cfg, "DEBUG", False):
        return True
    return any(getattr(cfg, name, False) for name in flag_names)


def _section(title: str) -> None:
    print(_SECTION_RULE)
    print(title)
    print(_SECTION_RULE)


def _subrule() -> None:
    print(_SUBSECTION_RULE)


def debug_retrieval(query: str, results: list["RetrievedChunk"]) -> None:
    """
    Print the query and every retrieved chunk: similarity score, chunk
    id, lesson, and a short text preview. Score is printed first since
    it's the primary signal used when tuning retrieval quality. Answers
    "did the retriever find the right material, and how confidently?"
    without needing to inspect RetrievedChunk objects by hand.
    """
    if not is_enabled(config, "DEBUG_RETRIEVAL"):
        return

    print()
    print("Query:")
    print(query)
    print()
    _section("Retrieved chunks")
    print()

    if not results:
        print("(none)")
        print()
        return

    for i, chunk in enumerate(results, start=1):
        preview = chunk.text[:_PREVIEW_CHARS].replace("\n", " ").strip()
        if len(chunk.text) > _PREVIEW_CHARS:
            preview += "..."

        print(f"{i}.")
        print(f"score: {chunk.similarity:.4f}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"lesson: {chunk.lesson_title}")
        print(f"token_count: {chunk.token_count}")
        print()
        print("Preview:")
        print(preview)
        print()
        _subrule()
    print()


def debug_prompt(prompt: "BuiltPrompt") -> None:
    """
    Print what will be sent to the LLM, kept readable rather than
    exhaustive: the full system prompt (it's a fixed, short template —
    see templates.py), only the first ~500 characters of the retrieved
    context (context is the part that can legitimately run to thousands
    of tokens), and the full user question. An approximate token count
    (word-based, same convention as the rest of the project) and the
    truncation flag are shown so debugging doesn't require flooding the
    terminal with the entire context block.
    """
    if not is_enabled(config, "DEBUG_PROMPT"):
        return

    _section("PROMPT")

    print("System Prompt:")
    print(prompt.system)
    _subrule()

    print("Context (first {} chars):".format(_CONTEXT_PREVIEW_CHARS))
    context_preview = prompt.context[:_CONTEXT_PREVIEW_CHARS]
    if len(prompt.context) > _CONTEXT_PREVIEW_CHARS:
        context_preview += "..."
    print(context_preview)
    _subrule()

    print("User Question:")
    print(prompt.user)
    _subrule()

    approx_tokens = (
        len(prompt.system.split()) + len(prompt.context.split()) + len(prompt.user.split())
    )
    print(f"Prompt token count (approx, word-based): {approx_tokens}")
    _subrule()

    print(f"Context truncated: {prompt.truncated}")
    print()


def debug_generation(answer: "Answer", raw_text: str | None = None) -> None:
    """
    Print what the model actually produced. raw_text is the decoded
    completion before any post-processing; if the caller has no
    post-processing step (as is currently the case in LocalHFModel),
    it's identical to answer.text and can be omitted.
    """
    if not is_enabled(config, "DEBUG_GENERATION"):
        return

    _section("GENERATION")

    print(f"Model:             {answer.model_name}")
    print(f"Generation time:   {answer.latency_seconds:.2f} s")
    print(f"Prompt tokens:     {answer.prompt_tokens}")
    print(f"Completion tokens: {answer.completion_tokens}")
    print()

    print("Raw generated text:")
    print(raw_text if raw_text is not None else answer.text)
    _subrule()

    print("Final Answer:")
    print(answer.text)
    print()


def debug_timings(timings: dict[str, float]) -> None:
    """
    Print stage timings in milliseconds for fast stages and seconds for
    slow ones (e.g. "Query embedding + FAISS search: 12 ms" vs
    "Generation: 4.81 s"). `timings` maps a human-readable stage label to
    elapsed seconds; a "Total" key, if present, is printed last.
    """
    if not is_enabled(config, "DEBUG_TIMINGS"):
        return

    _section("TIMINGS")

    ordered = [(k, v) for k, v in timings.items() if k != "Total"]
    if "Total" in timings:
        ordered.append(("Total", timings["Total"]))

    for label, seconds in ordered:
        if seconds < 1.0:
            print(f"{label}: {seconds * 1000:.0f} ms")
        else:
            print(f"{label}: {seconds:.2f} s")
    print()


_BOT_PREVIEW_CHARS = 80


def debug_bot_startup(bot_username: str | None = None) -> None:
    """
    Printed once the bot finishes initializing and starts polling.
    Gated by DEBUG_BOT like every other debug_* function here — bot
    lifecycle is treated as just another stage, not a special always-on
    print outside this module's normal gating contract.
    """
    if not is_enabled(config, "DEBUG_BOT"):
        return

    _section("BOT STARTUP")
    print(f"Bot username: @{bot_username}" if bot_username else "Bot username: (unknown)")
    print("Polling started.")
    print()


def debug_bot_shutdown() -> None:
    """Printed once the bot's polling loop stops."""
    if not is_enabled(config, "DEBUG_BOT"):
        return

    _section("BOT SHUTDOWN")
    print("Polling stopped.")
    print()


def debug_bot_message(chat_id: int, message_length: int, preview: str) -> None:
    """
    Printed for every incoming text message the bot processes.

    Deliberately never receives or prints the full message text — only
    chat_id, message_length, and a preview. Callers (telegram_bot.py) are
    expected to pass an already-short preview, but this function
    truncates again defensively to _BOT_PREVIEW_CHARS so a caller mistake
    can never leak a full message into the logs.
    """
    if not is_enabled(config, "DEBUG_BOT"):
        return

    trimmed = preview[:_BOT_PREVIEW_CHARS]
    if len(preview) > _BOT_PREVIEW_CHARS:
        trimmed += "..."

    print(f"[bot] chat_id={chat_id} message_length={message_length} preview={trimmed!r}")


def debug_bot_error(chat_id: int, exc: Exception) -> None:
    """Printed when pipeline.answer() raises inside the message handler."""
    if not is_enabled(config, "DEBUG_BOT"):
        return

    print(f"[bot] chat_id={chat_id} ERROR: {type(exc).__name__}: {exc}")

def debug_bot_response(chat_id: int, response_length: int, latency_seconds: float) -> None:
    """
    Printed after pipeline.answer() succeeds inside the message handler.
    Mirrors debug_bot_message()'s discipline: metadata only, never the
    actual response text. Latency is logged here (not just inside
    Answer.latency_seconds) because that field is generation-only time —
    this captures the full pipeline.answer() wall-clock time as observed
    by the bot layer, which is what actually matters for typing-indicator
    and responsiveness tuning.
    """
    if not is_enabled(config, "DEBUG_BOT"):
        return

    print(
        f"[bot] chat_id={chat_id} response_length={response_length} "
        f"latency_seconds={latency_seconds:.2f}"
    )