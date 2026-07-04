"""
Application bootstrap: the one place that composes this project's shared
services for any frontend (Telegram bot today; CLI, FastAPI, or Discord
later).

Responsibility: composition only. This module wires already-existing
components (RAGPipeline via build_default_pipeline()) together — it
contains no business logic of its own, no retrieval code, no prompt
code, no LLM code. It is intentionally as thin as factory.py (LLM
backend selection) or pipeline.py's build_default_pipeline() — one more
layer of "don't repeat this wiring," not a new architectural concept.

Every frontend (run_bot.py today; a future run_cli.py, a FastAPI app, a
Discord bot) should call build_application() instead of constructing
RAGPipeline itself. This is what lets multiple frontends share identical
initialization without duplicating it, and what keeps a frontend-specific
change (e.g. adding a REST API) from ever touching how the pipeline is
built.
"""

import config
from .rag.pipeline import RAGPipeline, build_default_pipeline


def build_application(cfg=config) -> RAGPipeline:
    """
    Construct and return the shared RAGPipeline. Currently a direct
    passthrough to build_default_pipeline() — kept as its own function
    (rather than every frontend importing build_default_pipeline
    directly) so that if shared services grow beyond a single pipeline
    (e.g. a shared cache, a metrics client) later, every frontend picks
    that up by calling this one function again, with no changes to
    run_bot.py or any other caller.
    """
    return build_default_pipeline(cfg)