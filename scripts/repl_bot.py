"""
Задание 4: простой консольный интерфейс (REPL) к RAG-пайплайну.
Резервный вариант интерфейса, если Telegram-бот недоступен.

Запуск:
    python scripts/repl_bot.py
"""

from rag_pipeline import answer_query


def main() -> None:
    print("RAG-бот QuantumForge Software. Введите вопрос или 'exit' для выхода.\n")
    while True:
        try:
            query = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "выход"}:
            print("До свидания.")
            break

        result = answer_query(query)
        print(f"\nБот: {result['answer']}")
        if result["sources"]:
            print(f"Источники: {', '.join(result['sources'])}")
        if result["blocked_chunks"]:
            print(f"[safety] отфильтровано подозрительных чанков: {len(result['blocked_chunks'])}")
        print()


if __name__ == "__main__":
    main()
