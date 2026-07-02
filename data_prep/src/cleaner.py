"""
Cleaner: strips subtitle-style noise from raw lesson text before chunking.
Responsibility: text normalization only. No structural splitting here.
"""

import re

# Common subtitle artifacts: timestamps, cue markers, bracketed sound tags
_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?([.,]\d+)?\s*(-->|→)?\s*\d{0,2}:?\d{0,2}:?\d{0,2}")
_BRACKET_NOISE_RE = re.compile(r"\[(music|applause|noise|silence|пауза|музыка)\]", re.IGNORECASE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")   # true paragraph break -> keep as a break
_SINGLE_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")  # single newline -> subtitle line-wrap, not a real break


def clean_text(text: str) -> str:
    """
    Remove timestamps and bracketed noise tags, then normalize line breaks.
    Source files are plain subtitle-style text with no markdown structure,
    so single newlines are subtitle line-wraps (mid-sentence) rather than
    paragraph boundaries — they're collapsed to spaces so sentence splitting
    in the chunker works on continuous prose.
    """
    text = _TIMESTAMP_RE.sub("", text)
    text = _BRACKET_NOISE_RE.sub("", text)

    # Preserve genuine paragraph breaks with a placeholder, collapse the rest
    text = _PARAGRAPH_BREAK_RE.sub("\u2029", text)   # temp marker for real breaks
    text = _SINGLE_NEWLINE_RE.sub(" ", text)
    text = text.replace("\u2029", "\n\n")

    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
