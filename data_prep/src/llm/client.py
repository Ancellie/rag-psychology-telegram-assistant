"""
BaseLLM: the abstract interface every LLM backend implements.

Responsibility: define the contract only. No backend logic lives here —
see huggingface_llm.py for the first implementation, and future
groq_llm.py / openai_llm.py / ollama_llm.py for the rest.

This interface represents "an LLM I can generate text from," not "an API
client." A locally loaded Hugging Face model and a future remote
provider's HTTP API satisfy the exact same contract from the outside —
nothing about network calls, auth, or request/response shapes appears
here. Retriever, PromptBuilder, and the Telegram bot depend only on this
abstract type, never on a concrete backend. Nothing outside src/llm/
should ever import a concrete class directly.
"""

from abc import ABC, abstractmethod

from ..prompt.models import BuiltPrompt
from .models import Answer


class BaseLLM(ABC):
    """
    One method, deliberately narrow — same philosophy as Retriever.search()
    and PromptBuilder.build() each having a single main entrypoint.
    """

    @abstractmethod
    def generate(self, prompt: BuiltPrompt) -> Answer:
        """
        Turn a BuiltPrompt into an Answer.

        Raises on unrecoverable failure. Standard exceptions (RuntimeError,
        ValueError) for now — a typed error hierarchy is deferred until a
        second backend actually needs normalized error handling across
        providers; introducing one for a single backend would be
        speculative abstraction. Callers (CLI, bot) decide how to surface
        a failure to the user.
        """
        raise NotImplementedError
