# Psychology Course RAG Telegram Bot

This project is a Retrieval-Augmented Generation (RAG) system built over a 243-lesson psychology course.

## Features
- Ask questions via Telegram bot
- Answers grounded in course materials (RAG)
- Falls back to LLM when context is missing
- Instructor-style responses based on course author tone

## Tech Stack
- Python
- FastAPI (optional backend)
- Telegram Bot API
- Vector Database (Qdrant / FAISS)
- Embeddings (OpenAI / bge-m3)
- LLM (GPT / Grok / local Llama)

## Architecture
- Document ingestion pipeline
- Chunking + embeddings
- Vector search retrieval
- LLM response generation with grounding

## Status
Initial development stage (MVP)