"""
TelegramBot: thin Telegram transport layer over RAGPipeline.

Responsibility: Telegram wiring only — receive an update, extract and
validate the text, hand it to RAGPipeline.answer(), send back
Answer.text. No retrieval, no prompt construction, no LLM-specific code
lives here; this module never imports Retriever, PromptBuilder, BaseLLM,
FAISS, or an embedder. Its only dependency into the RAG system is the
RAGPipeline contract (query in, Answer out), injected via the
constructor rather than built internally — same DI discipline as
RAGPipeline (Retriever/PromptBuilder/BaseLLM injected) and Retriever
(Embedder injected).

pipeline.answer() is synchronous and can take several seconds on a local
8B model. Every call to it runs through asyncio.to_thread() so the bot's
event loop stays free to keep handling other Telegram updates —
including sending typing actions — while one user's generation runs.

All logging goes through src/debug.py (debug_bot_*), never plain
print(): startup, shutdown, and per-message logs never include the full
message text, only chat_id, message length, and a short preview.
"""

import asyncio
import time

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .. import debug
from ..rag.pipeline import RAGPipeline

START_MESSAGE = (
    "Hi! I'm an assistant for this psychology course. Ask me anything "
    "covered in the lessons and I'll answer based only on the course "
    "material — I'll tell you plainly if something isn't covered."
)

HELP_MESSAGE = (
    "Just send me a question about the course material, in any language "
    "you like, and I'll look it up in the lessons and answer.\n\n"
    "Commands:\n"
    "/start - short introduction\n"
    "/help - this message"
)

GENERIC_ERROR_MESSAGE = (
    "Sorry, something went wrong while processing your question. "
    "Please try again in a moment."
)

_TYPING_REFRESH_SECONDS = 4  # Telegram's typing indicator expires after ~5s
_MESSAGE_PREVIEW_CHARS = 80


class TelegramBot:
    """
    Usage:
        pipeline = build_default_pipeline()
        bot = TelegramBot(pipeline=pipeline, token=config.TELEGRAM_BOT_TOKEN)
        bot.run()
    """

    def __init__(self, pipeline: RAGPipeline, token: str):
        if not token or not token.strip():
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is missing or empty. Set it in config.py "
                "(or whatever environment/secret store config.py reads it "
                "from) before starting the bot."
            )

        self.pipeline = pipeline
        self.token = token
        self.app: Application = (
            ApplicationBuilder()
            .token(self.token)
            .post_init(self._on_startup)
            .post_shutdown(self._on_shutdown)
            .build()
        )
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("help", self._handle_help))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        # No handler is registered for photos, voice, stickers, documents,
        # etc. — python-telegram-bot simply drops updates with no matching
        # handler, which is exactly the desired "ignore non-text updates"
        # behavior without extra code.

    async def _on_startup(self, application: Application) -> None:
        debug.debug_bot_startup(application.bot.username)

    async def _on_shutdown(self, application: Application) -> None:
        debug.debug_bot_shutdown()

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(START_MESSAGE)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP_MESSAGE)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        The only handler that ever touches self.pipeline. Everything else
        in this class is Telegram plumbing (commands, startup/shutdown).

        Flat control flow: start typing, run the pipeline call, always
        clean up the typing task in `finally`. No nested try/except —
        the single try/except/finally below preserves the exact same
        behavior (reply with the generic error message and log on
        failure; reply with the answer on success) with one fewer level
        of nesting.
        """
        message = update.message
        if message is None or not message.text:
            return  # non-text update slipped through, or no message body

        text = message.text.strip()
        if not text:
            return

        chat_id = update.effective_chat.id
        preview = text[:_MESSAGE_PREVIEW_CHARS]
        debug.debug_bot_message(chat_id, len(text), preview)

        typing_task = asyncio.create_task(self._keep_typing(context, chat_id))
        start = time.perf_counter()
        try:
            answer = await asyncio.to_thread(self.pipeline.answer, text)
        except Exception as exc:
            debug.debug_bot_error(chat_id, exc)
            await message.reply_text(GENERIC_ERROR_MESSAGE)
            return
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        latency = time.perf_counter() - start
        debug.debug_bot_response(chat_id, len(answer.text), latency)
        await message.reply_text(answer.text)

    async def _keep_typing(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        """
        Resend the typing action every _TYPING_REFRESH_SECONDS while
        generation runs. Telegram's own typing indicator expires after
        ~5s; a local 8B model routinely takes longer than that, so without
        this refresh the indicator would vanish partway through — which
        defeats the "immediate feedback while generating" goal for
        anything but the shortest answers.
        """
        try:
            while True:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

    def run(self) -> None:
        """Start polling. Blocks until interrupted (e.g. Ctrl+C)."""
        self.app.run_polling()