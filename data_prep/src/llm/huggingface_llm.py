"""
LocalHFModel: local Hugging Face Transformers backend for the LLM layer.

Responsibility: turn a BuiltPrompt into an Answer using a Hugging Face
causal LM running in-process on this machine. No network calls, no API
keys — model + tokenizer are loaded once at construction time, same
lifecycle as Embedder in embedder.py.

Built directly on AutoTokenizer + AutoModelForCausalLM rather than
transformers.pipeline(): apply_chat_template() + model.generate() gives
direct access to generation kwargs (stop strings, sampling params),
keeps the door open for streaming (TextIteratorStreamer) and batching
later, and avoids the pipeline's extra abstraction layer for a case
where we already need format control (chat template, thinking-mode
toggle, token accounting).

Device selection reuses src.utils.device.auto_device() — the same
function embedder.py uses — so CUDA/MPS/CPU detection has one
implementation project-wide.

Quantization (BitsAndBytes 4-bit/8-bit) is a constructor passthrough —
LocalHFModel applies whatever BitsAndBytesConfig it's given (or None for
full precision) but never builds one itself; see quantization.py for
where that decision is made from config.py.

Attention implementation is auto-detected, not hardcoded: if the
flash-attn package is importable and a CUDA device is in use, Flash
Attention 2 is requested; otherwise transformers falls back to its own
default (typically "sdpa"). This only ever loosens to a slower-but-always-
available implementation — it never hard-fails a run because flash-attn
isn't installed.
"""

import importlib.util
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompt.models import BuiltPrompt
from ..utils.device import auto_device
from .client import BaseLLM
from .models import Answer


def _select_attn_implementation(device: str) -> str | None:
    """
    Return "flash_attention_2" if usable, otherwise None (let transformers
    pick its own default, e.g. "sdpa"). Detection only, no installation —
    flash-attn has native-build requirements outside pip's normal path,
    so we probe for it rather than declaring it a hard dependency.
    """
    if device != "cuda":
        return None
    if importlib.util.find_spec("flash_attn") is None:
        return None
    return "flash_attention_2"


class LocalHFModel(BaseLLM):
    """
    Wraps a locally loaded Hugging Face causal LM as a BaseLLM.

    Usage:
        llm = LocalHFModel(model_name=config.HF_LLM_MODEL_NAME)
        answer = llm.generate(built_prompt)
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = True,
        torch_dtype: str = "auto",
        enable_thinking: bool = False,
        quantization_config=None,  # e.g. a BitsAndBytesConfig instance, when needed
    ):
        self.device = device or auto_device()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = dict(torch_dtype=torch_dtype)

        attn_implementation = _select_attn_implementation(self.device)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.attn_implementation = attn_implementation or "default (sdpa/eager)"

        if quantization_config is not None:
            # BitsAndBytes 4-bit/8-bit config, when introduced, passes
            # straight through here — no other change required anywhere
            # in this class. When quantized, bitsandbytes handles device
            # placement itself, so device_map="auto" is used instead of a
            # manual .to(device) call below.
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if quantization_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: BuiltPrompt) -> Answer:
        """
        Map BuiltPrompt onto a chat-formatted input, run generation, and
        normalize the output into an Answer.

        Raises RuntimeError on generation failure (e.g. OOM, decoding
        error) — standard exceptions for now, per BaseLLM's docstring.
        """
        messages = [
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": f"Context:\n{prompt.context}\n\nQuestion: {prompt.user}",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        prompt_token_count = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        except Exception as e:
            raise RuntimeError(
                f"LocalHFModel generation failed (model={self.model_name}): {e}"
            ) from e
        latency = time.perf_counter() - start

        # Slice off the prompt tokens — generate() returns prompt + completion.
        completion_ids = output_ids[0][prompt_token_count:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        return Answer(
            text=text,
            chunks_used=prompt.chunks_used,
            truncated=prompt.truncated,
            model_name=f"huggingface:{self.model_name}",
            latency_seconds=latency,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_ids.shape[0],
        )
