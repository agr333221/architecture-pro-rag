"""Общая логика чанкинга и эмбеддинга, переиспользуемая build_index.py и update_index.py."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 40

_model = None


def load_documents(folder: Path) -> list[dict]:
    """Читает все .md/.txt файлы из папки как отдельные документы.

    Путь "source" нормализуется к прямому слэшу ("/") независимо от ОС.
    Находка код-ревью: на Windows pathlib.Path.relative_to() дает путь с
    обратным слэшем ("knowledge_base\\file.md"). Если индекс строится на
    Windows, а обновляется/читается позже в Linux-контейнере (или наоборот),
    строковое сравнение путей в source_state.json перестает совпадать -
    все документы ошибочно считаются "измененными" и добавляются в индекс
    повторно, создавая дубликаты чанков."""
    docs = []
    for pattern in ("*.md", "*.txt"):
        for path in sorted(folder.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
            source = str(path.relative_to(ROOT)).replace(os.sep, "/")
            docs.append({"source": source, "title": title, "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Разбивает документы на чанки с overlap через LangChain splitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size_chars = CHUNK_SIZE_TOKENS * 4
    chunk_overlap_chars = CHUNK_OVERLAP_TOKENS * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{Path(doc['source']).stem}-{idx}",
                    "text": piece,
                    "source": doc["source"],
                    "title": doc["title"],
                    "chunk_index": idx,
                }
            )
    return chunks


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def save_faiss_index(index, path: Path) -> None:
    """
    Сохраняет индекс через сериализацию в байты и обычную запись файла.

    faiss.write_index() использует низкоуровневый fopen() и падает с
    "No such file or directory" на путях, содержащих не-ASCII символы
    (в частности кириллицу) на Windows, даже если директория существует
    и путь корректен. Обходим это через faiss.serialize_index() + запись
    байтов штатным Python open(), который путь с кириллицей поддерживает.
    """
    import faiss

    data = faiss.serialize_index(index)
    path.write_bytes(bytes(data))


def load_faiss_index(path: Path):
    """Обратная операция к save_faiss_index()."""
    import faiss
    import numpy as np

    raw = path.read_bytes()
    return faiss.deserialize_index(np.frombuffer(raw, dtype="uint8"))


def embed_texts(texts: list[str]):
    import faiss

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings
