"""
Entrypoint: obtains the shared RAGPipeline from the application
bootstrap and starts the Telegram bot.

This script no longer constructs RAGPipeline itself — src/app.py owns
that composition so any future frontend (CLI, FastAPI, Discord) can
reuse the identical initialization without duplicating it here.

Usage:
    python scripts/run_bot.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.app import build_application
from src.bot.telegram_bot import TelegramBot


def main() -> None:
    print("Building application (RAG pipeline)...")
    pipeline = build_application()
    print("Pipeline ready.")

    # TelegramBot's constructor validates TELEGRAM_BOT_TOKEN itself and
    # raises a clear ValueError if it's missing — no separate check here.
    bot = TelegramBot(pipeline=pipeline, token=config.TELEGRAM_BOT_TOKEN)

    print("Starting Telegram bot (polling)...")
    bot.run()


if __name__ == "__main__":
    main()