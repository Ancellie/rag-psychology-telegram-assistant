"""
LLM backend selection.

get_llm(config) -> BaseLLM is the one place that knows how to turn config
into a concrete backend instance. Everything else in the project —
retriever, prompt builder, Telegram bot — imports only BaseLLM and calls
get_llm(config); it never imports LocalHFModel (or, later, GroqLLM /
OpenAILLM / OllamaLLM) directly.

Deliberately a single if/return, not a registry or plugin system: with
one backend there is nothing to generalize yet. When a second backend
(Groq/OpenAI/Ollama) is added, this function grows one more branch keyed
off config.LLM_PROVIDER — that's the natural point to introduce a dict-based
dispatch if the branching gets unwieldy, not before.
"""

import config
from .client import BaseLLM
from .huggingface_llm import LocalHFModel
from .quantization import build_quantization_config


def get_llm(cfg=config) -> BaseLLM:
    """Construct the BaseLLM implementation selected by config.LLM_PROVIDER."""
    provider = getattr(cfg, "LLM_PROVIDER", "huggingface")

    if provider == "huggingface":
        return LocalHFModel(
            model_name=cfg.HF_LLM_MODEL_NAME,
            device=cfg.HF_LLM_DEVICE,
            max_new_tokens=cfg.HF_LLM_MAX_NEW_TOKENS,
            temperature=cfg.HF_LLM_TEMPERATURE,
            do_sample=cfg.HF_LLM_DO_SAMPLE,
            torch_dtype=cfg.HF_LLM_TORCH_DTYPE,
            enable_thinking=cfg.HF_LLM_ENABLE_THINKING,
            quantization_config=build_quantization_config(cfg),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        f"Only 'huggingface' is implemented so far."
    )
