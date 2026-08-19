# Psychology Course RAG Telegram Bot

A Retrieval-Augmented Generation (RAG) system built over a paid 243-lesson psychology course.
The bot answers questions grounded in course materials, mimicking the tone and style of the course author.

> ⚠️ **Note:** The source lesson files (`chunks.json` and `index.faiss`) are **not included** in this repository.
> They are derived from a **paid course** and cannot be redistributed. You must prepare your own data using the pipeline described below.

---

## Features

- Ask questions via Telegram bot
- Answers grounded in course materials (RAG)
- Falls back to LLM reasoning when relevant context is not found
- Responses written in the instructor's tone and style

---

## Tech Stack

- **Language:** Python 3.13
- **Bot:** Telegram Bot API (via `aiogram` / `python-telegram-bot`)
- **Vector Index:** FAISS
- **Embeddings:** `bge-m3` or OpenAI embeddings
- **LLM:** GPT / Grok / local Llama
- **Data Preparation:** custom chunking + embedding pipeline

## Architecture

1. **Ingestion** — raw lesson files are loaded and cleaned
2. **Chunking** — text is split into semantic chunks with metadata
3. **Embedding** — each chunk is embedded and stored in a FAISS index
4. **Retrieval** — user query is embedded and matched against the index
5. **Generation** — top chunks are passed to the LLM as context to produce a grounded answer

---

## Why `chunks.json` and `index.faiss` Are Hidden

These files contain processed content derived from a **commercial psychology course**.
Distributing them would violate the author's intellectual property rights.
If you want to use this system with your own content, prepare your data using the pipeline in `data_prep/`.

---

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials
4. Prepare your own data using the `data_prep` pipeline
5. Run the bot:
   ```bash
   python main.py
   ```

---

## Status

🚧 Active development — core RAG pipeline and bot handlers implemented