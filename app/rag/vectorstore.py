# app/rag/vectorstore.py
from pathlib import Path

from langchain_chroma import Chroma

from app.config.settings import get_settings
from app.rag.embeddings import get_embeddings

settings = get_settings()


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(settings.VECTORSTORE_PATH),
        embedding_function=embeddings,
        collection_name="banco_rag",
        collection_metadata={"hnsw:space": "cosine"}
    )


def verify_vectorstore() -> bool:
    if not settings.VECTORSTORE_PATH.exists():
        return False
    try:
        store = get_vectorstore()
        return store._collection.count() > 0
    except:
        return False
