"""
Задание 3: построение векторного индекса базы знаний.

Читает документы из knowledge_base/, разбивает их на чанки,
строит эмбеддинги локальной моделью Sentence-Transformers
и сохраняет индекс FAISS вместе с метаданными чанков.

Запуск:
    python scripts/build_index.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from indexing_common import (
    EMBEDDING_MODEL_NAME,
    ROOT,
    chunk_documents,
    embed_texts,
    load_documents,
    save_faiss_index,
)

load_dotenv()

KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "data" / "index"
INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", str(INDEX_DIR / "faiss.index")))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = ROOT / INDEX_PATH
METADATA_PATH = INDEX_DIR / "metadata.json"
STATE_PATH = INDEX_DIR / "source_state.json"


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents(KB_DIR)
    if not docs:
        raise SystemExit(f"В папке {KB_DIR} не найдено ни одного документа .md")

    chunks = chunk_documents(docs)
    print(f"Документов: {len(docs)}, чанков: {len(chunks)}")

    print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}")
    t0 = time.time()
    embeddings = embed_texts([c["text"] for c in chunks])
    elapsed = time.time() - t0

    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_faiss_index(index, INDEX_PATH)
    METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    state = {doc["source"]: os.path.getmtime(ROOT / doc["source"]) for doc in docs}
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "model": EMBEDDING_MODEL_NAME,
        "knowledge_base": str(KB_DIR.relative_to(ROOT)),
        "documents": len(docs),
        "chunks": len(chunks),
        "embedding_dim": dim,
        "generation_seconds": round(elapsed, 2),
        "index_path": str(INDEX_PATH.relative_to(ROOT)),
    }
    (INDEX_DIR / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Индекс построен и сохранен:")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
