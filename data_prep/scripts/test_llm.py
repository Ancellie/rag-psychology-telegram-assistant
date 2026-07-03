"""
Standalone smoke test for the LLM layer.

Builds a BuiltPrompt by hand (no retriever/prompt-builder wiring yet —
that integration is the next milestone) and runs it through get_llm(),
i.e. through whatever backend config.LLM_PROVIDER selects. Prints the
Answer. No Telegram, no retrieval — this only verifies that
BuiltPrompt -> Answer works end to end for the current backend.

Usage:
    python scripts/test_llm.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm.factory import get_llm
from src.prompt.models import BuiltPrompt
from src.prompt import templates

TEST_PROMPTS = [
    BuiltPrompt(
        system=templates.SYSTEM_PROMPT_TEMPLATE,
        context=(
            "[Lesson: Defense Mechanisms]\n"
            "Defense mechanisms are unconscious psychological strategies "
            "the mind uses to protect itself from anxiety, unacceptable "
            "thoughts, or feelings. Sigmund Freud first described them; "
            "his daughter Anna Freud expanded the concept. Common examples "
            "include repression, denial, projection, and rationalization."
        ),
        user="What is repression as a defense mechanism?",
        chunks_used=["chunk-001"],
        truncated=False,
    ),
    BuiltPrompt(
        system=templates.SYSTEM_PROMPT_TEMPLATE,
        context=templates.NO_CONTEXT_NOTICE,
        user="What is the airspeed velocity of an unladen swallow?",
        chunks_used=[],
        truncated=False,
    ),
]


def main() -> None:
    print("Loading LLM backend...")
    llm = get_llm()
    print(f"Backend ready.\n")

    for prompt in TEST_PROMPTS:
        print("=" * 60)
        print(f"QUESTION: {prompt.user}")
        print("=" * 60)

        answer = llm.generate(prompt)

        print(f"ANSWER:\n{answer.text}\n")
        print(f"model_name:        {answer.model_name}")
        print(f"latency_seconds:   {answer.latency_seconds:.2f}")
        print(f"prompt_tokens:     {answer.prompt_tokens}")
        print(f"completion_tokens: {answer.completion_tokens}")
        print(f"chunks_used:       {answer.chunks_used}")
        print(f"truncated:         {answer.truncated}")
        print()


if __name__ == "__main__":
    main()
