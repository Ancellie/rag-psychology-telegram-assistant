"""
Prompt templates: all natural-language prompt text lives here, and nowhere
else in the prompt package.

Responsibility: prompt engineering only. No logic, no formatting decisions,
no imports of RetrievedChunk or config. builder.py owns the *assembly*
(truncation, iteration, budget); this module owns the *wording*. Keeping
them apart means a prompt-engineering change (tone, instructions, wording)
never requires touching builder.py, and a logic change never requires
touching prompt text.
"""

SYSTEM_PROMPT_TEMPLATE = """You are an experienced instructor for a psychology course. You answer \
student questions in the same clear, structured, and approachable style the \
course itself uses.

Rules you must always follow:
1. Base your answer only on the course material provided in the context below. \
Do not use outside knowledge, even if you know the answer from elsewhere.
2. If the provided context does not contain enough information to answer the \
question, say so explicitly and clearly (e.g. "This isn't covered in the \
course material provided."). Do not guess or fill gaps from general knowledge.
3. Answer in the same language the student's question is written in.
4. Keep the instructor-like tone: clear, patient, and precise, as if teaching \
the concept in a lecture."""

# Used to format each individual retrieved chunk inside the context block.
# lesson_title is included now so citation-style answers are possible, and so
# that adding further metadata (e.g. source_file) later is a template edit,
# not a logic change in builder.py.
CONTEXT_CHUNK_TEMPLATE = """[Lesson: {lesson_title}]
{text}"""

# Joins multiple formatted chunks together inside the context block.
CONTEXT_CHUNK_SEPARATOR = "\n\n---\n\n"

# Used verbatim as the `context` field when no chunks were retrieved at all,
# or none could be included. Explicit rather than an empty string so the
# LLM is told plainly rather than left to infer from an empty context.
NO_CONTEXT_NOTICE = (
    "No relevant course material was found for this question. Tell the "
    "student explicitly that this topic does not appear to be covered in "
    "the course, and do not attempt to answer from outside knowledge."
)
