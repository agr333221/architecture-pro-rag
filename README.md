# RAG-бот для QuantumForge Software (учебный проект, Спринт 7)

Учебный проект: бот на основе Retrieval-Augmented Generation (RAG) поверх собственной
(вымышленной) базы знаний. Подробные ответы по каждому заданию - в [Project_template.md](Project_template.md).

## Стек

- LLM: OpenRouter (OpenAI-совместимый API), модель настраивается в .env (OPENROUTER_MODEL).
- Эмбеддинги: локальная модель Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2), без облачных ключей.
- Векторная БД: FAISS (локально, без сервера).
- Интерфейс: Telegram-бот (тариф "Про") + консольный REPL как резервный вариант.
- Оркестрация промптинга: few-shot + Chain-of-Thought, слои защиты от prompt-injection.

## Структура репозитория

```
knowledge_base/     - база знаний (30+ документов, вымышленный мир на основе Star Wars)
terms_map.json       - словарь замен терминов (исходное -> вымышленное)
docs/                 - "источник" новых документов для автообновления индекса (Задание 6)
scripts/
  build_index.py      - чанкинг + эмбеддинги + построение FAISS-индекса (Задание 3)
  safety.py            - safety-классификаторы и фильтры prompt-injection (Задание 5)
  rag_pipeline.py      - ядро RAG: поиск + few-shot + CoT + генерация ответа (Задание 4)
  repl_bot.py           - консольный интерфейс к RAG-пайплайну
  telegram_bot.py       - Telegram-интерфейс к RAG-пайплайну (Задание 4, тариф "Про")
  update_index.py       - автоматическое обновление индекса (Задание 6)
  evaluate.py            - прогон golden-вопросов и подсчет метрик (Задание 7)
data/index/            - сгенерированный FAISS-индекс (хранится в git - это
                          сдаваемый результат Задания 3, см. .gitignore)
logs/                   - логи запросов и обновлений индекса (хранятся в git -
                          сдаваемый результат Заданий 6 и 7)
diagrams/               - диаграммы PlantUML (архитектура обновления, sequence-диаграмма оценки)
golden_questions.txt     - золотой набор вопросов для автотестирования (Задание 7)
screenshots/             - скриншоты диалогов бота (Задание 4, 5)
```

## Быстрый старт

1. Создать виртуальное окружение и установить зависимости:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Скопировать `.env.example` в `.env` и заполнить реальными значениями
   (в этом проекте `.env` уже создан и заполнен - не публикуйте его).

3. Построить индекс базы знаний:

   ```
   python scripts/build_index.py
   ```

4. Запустить консольного бота:

   ```
   python scripts/repl_bot.py
   ```

   Либо Telegram-бота:

   ```
   python scripts/telegram_bot.py
   ```

5. Обновить индекс новыми документами из `docs/`:

   ```
   python scripts/update_index.py
   ```

6. Прогнать автотесты по golden-вопросам:

   ```
   python scripts/evaluate.py
   ```

## Docker

```
docker compose up --build
```

Поднимает контейнер с ботом. FAISS используется как встроенная библиотека внутри
контейнера бота (отдельный сервис для FAISS не нужен - в отличие от Qdrant/Chroma,
у FAISS нет клиент-серверного режима).

**Проверено реальной сборкой:** образ `repo-result-bot:latest` (3.73 ГБ, в основном
за счет torch) успешно собран на `python:3.11-slim`. Smoke-тест
(`docker compose run --rm bot python -c "import faiss, sentence_transformers, telegram, openai"`)
подтвердил, что все зависимости импортируются, а смонтированные тома
`data/`, `knowledge_base/`, `logs/`, `docs/` видны внутри контейнера
(индекс и 35 документов базы знаний доступны).

## Безопасность

Все реальные ключи и токены хранятся только в `.env`, который исключен из git
через `.gitignore`. Перед тем как сделать репозиторий публичным, убедитесь,
что `.env` не закоммичен: `git ls-files | grep env`.

## Примечание о диаграммах

Диаграммы в [diagrams/](diagrams/) и в `Project_template.md` сделаны в
PlantUML (`.puml`), как указано в списке инструментов задания. Чтобы увидеть
отрисованную схему, вставьте содержимое `.puml`-файла в любой онлайн-редактор
PlantUML, например [plantuml.com/plantuml](https://www.plantuml.com/plantuml).
