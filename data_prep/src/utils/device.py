"""
Shared device-detection utility.

Single source of truth for CUDA/MPS/CPU auto-detection. Both the retrieval
embedder (Embedder in src/embedder.py) and the LLM layer (LocalHFModel in
src/llm/huggingface_llm.py) call this instead of each probing torch's
backends independently — one place to update if device-selection logic
ever needs to change (e.g. adding a new backend, changing MPS fallback
rules).
"""

import torch


def auto_device() -> str:
    """Return the best available device: 'cuda' > 'mps' > 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
