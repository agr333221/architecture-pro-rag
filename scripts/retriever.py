"""
Общий модуль поиска по векторному индексу FAISS.
Используется build_index.py (для демо-запроса), rag_pipeline.py и evaluate.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from indexing_common import load_faiss_index

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", str(INDEX_DIR / "faiss.index")))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = ROOT / INDEX_PATH
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

_model = None
_index = None
_chunks = None


def _lazy_load():
    global _model, _index, _chunks
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {INDEX_PATH}. Сначала запустите scripts/build_index.py"
            )
        _index = load_faiss_index(INDEX_PATH)
        _chunks = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model, _index, _chunks


def search(query: str, top_k: int = 5, exclude_sources: set[str] | None = None) -> list[dict]:
    """
    Возвращает top_k чанков, наиболее близких к запросу по косинусной близости.

    exclude_sources - опциональный набор путей источников (chunk["source"]),
    которые нужно исключить из результатов. Используется в evaluate.py (Задание 7)
    для имитации "удаленных" из базы знаний сущностей без физического удаления
    файлов knowledge_base/ (чтобы не ломать демонстрации в Заданиях 4-5).
    """
    import faiss

    model, index, chunks = _lazy_load()

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    # Нормализуем разделители пути (Windows дает "knowledge_base\\file.md",
    # а exclude_sources в evaluate.py задан с "/") - иначе сравнение никогда
    # не совпадет на Windows и исключение молча не сработает.
    normalized_exclude = (
        {s.replace("\\", "/") for s in exclude_sources} if exclude_sources else None
    )

    # Если есть исключения - берем с запасом, чтобы после фильтрации
    # все равно осталось top_k результатов (по возможности).
    search_k = top_k * 4 if exclude_sources else top_k
    scores, ids = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        if normalized_exclude and chunk["source"].replace("\\", "/") in normalized_exclude:
            continue
        chunk["score"] = float(score)
        results.append(chunk)
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Кто уничтожил Войд-Ядро в Битве у Нексус-Рубежа?"
    print(f"Запрос: {query}\n")
    for i, chunk in enumerate(search(query, top_k=3), start=1):
        print(f"[{i}] score={chunk['score']:.3f} источник={chunk['source']} ({chunk['title']})")
        print(chunk["text"][:300].replace("\n", " "))
        print("-" * 60)
