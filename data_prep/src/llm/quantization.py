"""
Builds a BitsAndBytesConfig (or None) from config.py settings.

Responsibility: translate config.HF_LLM_QUANTIZATION and related settings
into a concrete transformers/bitsandbytes quantization config object.
This is the one place that decides *how* the Hugging Face model is loaded
(4-bit / 8-bit / full precision) — LocalHFModel itself never hardcodes a
BitsAndBytesConfig; it only ever receives one as a constructor argument
(or None for full precision) via get_llm() in factory.py. Switching
precision is therefore a config.py change only.

Kept as a separate module rather than folded into huggingface_llm.py so
the compute-dtype fallback logic (bfloat16 if the GPU supports it,
float16 otherwise) and the mode-to-config mapping have one place to live,
independent of model loading/generation code.
"""

import torch
from transformers import BitsAndBytesConfig


def _select_compute_dtype() -> torch.dtype:
    """bfloat16 on GPUs that support it (Ampere/Ada and newer), float16 otherwise."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(cfg) -> BitsAndBytesConfig | None:
    """
    Returns a BitsAndBytesConfig matching cfg.HF_LLM_QUANTIZATION, or None
    for full precision (no quantization).

    Raises ValueError on an unrecognized mode so a typo in config.py fails
    loudly at startup rather than silently falling back to full precision
    on a 12GB card.
    """
    mode = getattr(cfg, "HF_LLM_QUANTIZATION", "none")

    if mode == "none":
        return None

    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=getattr(cfg, "HF_LLM_BNB_4BIT_QUANT_TYPE", "nf4"),
            bnb_4bit_compute_dtype=_select_compute_dtype(),
            bnb_4bit_use_double_quant=getattr(
                cfg, "HF_LLM_BNB_4BIT_USE_DOUBLE_QUANT", True
            ),
        )

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(
        f"Unknown HF_LLM_QUANTIZATION: {mode!r}. Expected '4bit', '8bit', or 'none'."
    )
