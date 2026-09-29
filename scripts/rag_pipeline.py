"""
Задание 4: ядро RAG-бота.

Пайплайн: запрос пользователя -> эмбеддинг -> поиск в FAISS ->
фильтрация чанков (safety_in) -> сборка промпта (few-shot + CoT) ->
вызов LLM через OpenRouter (OpenAI-совместимый API) -> проверка
ответа (safety_out) -> логирование (для Задания 7) -> ответ пользователю.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from retriever import search
from safety import (
    SAFE_SYSTEM_PROMPT,
    filter_chunks,
    is_decline,
    is_output_safe,
    redact_secrets,
    sanitize_user_query,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
QUERY_LOG_PATH = LOGS_DIR / "query_logs.jsonl"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

TOP_K = 5
I_DONT_KNOW = "Я не знаю. В базе знаний не нашлось информации, чтобы ответить на этот вопрос."

# FAISS IndexFlatIP всегда возвращает top_k ближайших соседей, даже если ни
# один из них реально не релевантен запросу - порога релевантности "из коробки"
# нет. Поэтому нужен собственный порог по косинусной близости: ниже него чанк
# считается "случайным совпадением", а не найденным ответом.
#
# ВАЖНОЕ ОГРАНИЧЕНИЕ (проверено экспериментально): порог отсекает большинство
# реалистичных нерелевантных вопросов на связные темы вне базы знаний
# (например, "Какая сегодня погода в Лондоне?" - score 0.16, "Сколько стоит
# билет на поезд?" - 0.22), но НЕ является надежной защитой против случайного
# набора символов/бессмысленных слов: короткие OOV-строки у этой модели иногда
# дают неожиданно высокий score (пример: "абырвалг щзч плюмбум трокодыль" -
# 0.72, выше типичного релевантного попадания 0.55-0.65). Простое пороговое
# отсечение по косинусной близости в принципе не может полностью решить эту
# проблему - по-хорошему нужен дополнительный шаг ранжирования cross-encoder'ом
# поверх bi-encoder'а (см. урок 13 курса, раздел "Bi-encoder vs Cross-encoder"),
# который здесь не реализован ради простоты учебного проекта.
MIN_RELEVANCE_SCORE = 0.45

# Few-shot примеры, реально извлеченные из базы знаний (см. knowledge_base/).
#
# ВАЖНО: примеры намеренно не содержат фактов о сущностях, которые могут
# исключаться из поиска в Задании 7 (evaluate.py, GAP_SOURCES) - иначе
# few-shot попросту "обучает" модель этим фактам напрямую через промпт,
# в обход векторного поиска, и тест на честное "не знаю" перестает быть
# показательным (это и была часть находки код-ревью: пример с Войд-Ядром
# делал бессмысленным любой gap-тест по этой сущности).
FEW_SHOT_EXAMPLES = [
    {
        "q": "Кто является напарником Даска Оррина на корабле \"Блуждающая Комета\"?",
        "a": (
            "Шаг 1: найду в контексте упоминание напарника Даска Оррина.\n"
            "Шаг 2: в документе о Врошане указано, что он служит верным напарником "
            "и вторым пилотом Даска Оррина на корабле \"Блуждающая Комета\".\n"
            "Шаг 3: значит, напарником Даска Оррина является Врошан.\n"
            "Ответ: Врошан, представитель расы трешкин."
        ),
    },
    {
        "q": "На какой планете Каэл Виндраннер обучался у мастера Виндрала?",
        "a": (
            "Шаг 1: найду документ про обучение Каэла Виндраннера.\n"
            "Шаг 2: в документе о Виндрале и о планете Мурквейл указано, что обучение "
            "проходило на болотистой планете Мурквейл.\n"
            "Ответ: на планете Мурквейл."
        ),
    },
]


def build_messages(query: str, context_chunks: list[dict]) -> list[dict]:
    context_text = "\n\n".join(
        f"[Источник {i + 1}: {c['title']}]\n{c['text']}" for i, c in enumerate(context_chunks)
    )

    messages = [{"role": "system", "content": SAFE_SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["q"]})
        messages.append({"role": "assistant", "content": ex["a"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"[CONTEXT]\n<<<\n{context_text}\n>>>\n\n"
                f"[Вопрос]\n{query}\n\n"
                "Ответь строго на основе CONTEXT, следуя формату из примеров выше "
                "(шаги рассуждения, затем строка 'Ответ: ...')."
            ),
        }
    )
    return messages


def call_llm(messages: list[dict], max_retries: int = 3) -> str:
    """
    Вызывает LLM через OpenRouter. Бесплатные модели (общий пул провайдеров)
    иногда временно недоступны или возвращают ответ без choices - делаем
    несколько попыток с паузой, прежде чем поднять ошибку.
    """
    from openai import OpenAI

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не задан в .env")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0.2,
            )
            if response.choices:
                return response.choices[0].message.content or ""
            last_error = RuntimeError(f"Пустой choices в ответе провайдера: {response!r}")
        except Exception as exc:  # noqa: BLE001 - временные сбои общего бесплатного пула
            last_error = exc

        if attempt < max_retries:
            print(f"  [call_llm] попытка {attempt} неудачна ({last_error}), повтор...")
            time.sleep(3 * attempt)

    raise RuntimeError(f"LLM недоступна после {max_retries} попыток: {last_error}")


def log_query(entry: dict) -> None:
    with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def answer_query(
    query: str,
    top_k: int = TOP_K,
    filter_enabled: bool = True,
    exclude_sources: set[str] | None = None,
) -> dict:
    """
    Полный цикл RAG-ответа. Возвращает словарь с ответом, источниками
    и флагами безопасности - используется REPL, Telegram-ботом и evaluate.py.

    exclude_sources пробрасывается в retriever.search() и используется
    только в evaluate.py для имитации пробелов в базе знаний (Задание 7).
    """
    t0 = time.time()
    clean_query = sanitize_user_query(query)

    raw_chunks = search(clean_query, top_k=top_k, exclude_sources=exclude_sources)
    relevant_chunks = [c for c in raw_chunks if c["score"] >= MIN_RELEVANCE_SCORE]
    clean_chunks, blocked_chunks = filter_chunks(relevant_chunks, enabled=filter_enabled)

    found_chunks = bool(clean_chunks)

    llm_error: str | None = None

    if not found_chunks:
        answer = I_DONT_KNOW
        safe_output = True
        source_titles: list[str] = []
    else:
        messages = build_messages(clean_query, clean_chunks)
        source_titles = [c["title"] for c in clean_chunks]
        try:
            raw_answer = call_llm(messages)
            safe_output = is_output_safe(raw_answer)
            answer = raw_answer if safe_output else redact_secrets(raw_answer)
        except Exception as exc:  # noqa: BLE001 - LLM недоступна (лимиты, сеть, провайдер)
            llm_error = str(exc)
            safe_output = True
            answer = (
                "Извините, сервис генерации ответов сейчас недоступен "
                "(перегружен провайдер или исчерпан лимит запросов). "
                "Я нашел релевантные источники в базе знаний, но не смог "
                "сформировать по ним ответ. Попробуйте, пожалуйста, позже.\n\n"
                f"Найденные источники: {', '.join(source_titles)}"
            )

    elapsed = time.time() - t0

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "chunks_found": found_chunks,
        "chunks_blocked_by_safety": len(blocked_chunks),
        "answer_length": len(answer),
        "successful_answer": found_chunks and llm_error is None and not is_decline(answer),
        "safe_output": safe_output,
        "sources": source_titles,
        "elapsed_seconds": round(elapsed, 2),
        "filter_enabled": filter_enabled,
        "llm_error": llm_error,
    }
    log_query(log_entry)

    return {
        "answer": answer,
        "sources": source_titles,
        "blocked_chunks": [c["title"] for c in blocked_chunks],
        "safe_output": safe_output,
        "log_entry": log_entry,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Кто такой Каэл Виндраннер?"
    result = answer_query(q)
    print(result["answer"])
    print("\nИсточники:", ", ".join(result["sources"]) or "-")
