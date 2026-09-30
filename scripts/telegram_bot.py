"""
Задание 4 (тариф "Про"): интерфейс RAG-бота в Telegram.

Запуск:
    python scripts/telegram_bot.py

Требует переменную окружения TELEGRAM_BOT_TOKEN в .env (получена у @BotFather).
Требует сетевого доступа к Telegram Bot API и к OpenRouter.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag_pipeline import answer_query

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


HELP_TEXT = (
    "Я корпоративный RAG-бот QuantumForge Software (учебный проект, Спринт 7).\n\n"
    "Что я умею:\n"
    "- Отвечаю на вопросы по внутренней базе знаний компании (вселенная "
    "\"Сага о Нексарионе\") - ищу релевантные документы и формирую ответ "
    "с пошаговым рассуждением.\n"
    "- Если ответа нет в базе знаний, честно говорю \"Я не знаю\" вместо того, "
    "чтобы придумывать факты.\n"
    "- Отфильтровываю документы с признаками вредоносных инструкций "
    "(prompt-injection) и не раскрываю секреты, даже если они встречаются "
    "в базе знаний.\n\n"
    "Команды:\n"
    "/start - приветствие\n"
    "/help - эта справка\n\n"
    "Просто напишите вопрос обычным сообщением, например:\n"
    "\"Кто такой Каэл Виндраннер?\""
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я корпоративный RAG-бот QuantumForge Software (учебный проект). "
        "Задайте вопрос по базе знаний или напишите /help для справки."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text

    try:
        result = answer_query(query)
        reply = result["answer"]
        if result["sources"]:
            reply += "\n\nИсточники: " + ", ".join(result["sources"])
    except Exception as exc:  # noqa: BLE001 - бот не должен молча падать на сообщении
        reply = (
            "Извините, произошла внутренняя ошибка при обработке вопроса. "
            "Попробуйте, пожалуйста, переформулировать запрос или повторить "
            "попытку позже."
        )
        print(f"[handle_message] ошибка: {exc}")

    await update.message.reply_text(reply)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram-бот запущен. Ожидание сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
