"""
Задание 6: автоматическое обновление базы знаний.

Источник новых документов - локальная папка docs/ (без сети, см. 00_NEEDED_INFO.md).
Скрипт сканирует docs/, находит новые или измененные файлы (по времени
изменения относительно предыдущего запуска), разбивает их на чанки,
строит эмбеддинги и добавляет их в существующий индекс FAISS, не
перестраивая его с нуля. Каждый запуск логируется в logs/update_log.jsonl.

Запуск вручную:
    python scripts/update_index.py

Периодический запуск на Windows - через Планировщик заданий (пример команды -
см. README.md и Project_template.md, раздел "Задание 6", там путь приведен
как строка PowerShell без конфликтов экранирования).

При ошибке скрипт возвращает ненулевой код выхода и пишет причину в лог -
Планировщик заданий Windows покажет это в истории запуска задачи, что
эквивалентно "повтору при следующем расписании" (без отдельного retry
внутри скрипта, чтобы не усложнять учебный проект).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from indexing_common import (
    ROOT,
    chunk_documents,
    embed_texts,
    load_documents,
    load_faiss_index,
    save_faiss_index,
)

load_dotenv()

DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", str(INDEX_DIR / "faiss.index")))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = ROOT / INDEX_PATH
METADATA_PATH = INDEX_DIR / "metadata.json"
STATE_PATH = INDEX_DIR / "source_state.json"
UPDATE_LOG_PATH = ROOT / "logs" / "update_log.jsonl"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def find_new_or_changed_docs(state: dict) -> list[dict]:
    all_docs = load_documents(DOCS_DIR)
    changed = []
    for doc in all_docs:
        mtime = os.path.getmtime(ROOT / doc["source"])
        if state.get(doc["source"]) != mtime:
            changed.append(doc)
    return changed, all_docs


def log_run(entry: dict) -> None:
    UPDATE_LOG_PATH.parent.mkdir(exist_ok=True)
    with UPDATE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    try:
        import faiss  # noqa: F401 - нужен для faiss.IndexFlatIP ниже

        DOCS_DIR.mkdir(exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        state = load_state()
        changed_docs, all_docs = find_new_or_changed_docs(state)

        if not changed_docs:
            print("Новых или измененных документов в docs/ не найдено.")
            log_run(
                {
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "new_chunks": 0,
                    "index_size_after": _current_index_size(),
                    "status": "no_changes",
                    "error": None,
                    "duration_seconds": round(time.time() - t0, 2),
                }
            )
            return 0

        new_chunks = chunk_documents(changed_docs)
        print(f"Найдено измененных документов: {len(changed_docs)}, новых чанков: {len(new_chunks)}")

        embeddings = embed_texts([c["text"] for c in new_chunks])

        changed_sources = {d["source"] for d in changed_docs}

        if INDEX_PATH.exists():
            old_index = load_faiss_index(INDEX_PATH)
            old_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

            # Находка код-ревью: раньше старые чанки измененного документа не
            # удалялись перед добавлением новых - при каждом изменении файла
            # в индексе копились дубликаты. IndexFlatIP не поддерживает
            # точечное удаление векторов, поэтому пересобираем индекс:
            # оставляем векторы всех чанков, ЧЬИ документы не менялись, и
            # добавляем к ним свежие эмбеддинги измененных документов.
            keep_positions = [
                i for i, m in enumerate(old_metadata) if m["source"] not in changed_sources
            ]
            kept_metadata = [old_metadata[i] for i in keep_positions]

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            if keep_positions:
                old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
                index.add(old_vectors[keep_positions])
            metadata = kept_metadata
        else:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            metadata = []

        index.add(embeddings)
        metadata.extend(new_chunks)

        save_faiss_index(index, INDEX_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        for doc in all_docs:
            state[doc["source"]] = os.path.getmtime(ROOT / doc["source"])
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        log_run(
            {
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "new_chunks": len(new_chunks),
                "changed_documents": [d["source"] for d in changed_docs],
                "index_size_after": index.ntotal,
                "status": "ok",
                "error": None,
                "duration_seconds": round(time.time() - t0, 2),
            }
        )
        print(f"Индекс обновлен. Итоговый размер индекса: {index.ntotal} чанков.")
        return 0

    except Exception as exc:  # noqa: BLE001 - логируем любую ошибку обновления
        log_run(
            {
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "new_chunks": 0,
                "index_size_after": _current_index_size(),
                "status": "error",
                "error": f"{exc}\n{traceback.format_exc()}",
                "duration_seconds": round(time.time() - t0, 2),
            }
        )
        print(f"Ошибка обновления индекса: {exc}")
        return 1


def _current_index_size() -> int | None:
    try:
        if INDEX_PATH.exists():
            return load_faiss_index(INDEX_PATH).ntotal
    except Exception:  # noqa: BLE001
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
