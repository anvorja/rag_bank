# app/rag/retriever.py
from functools import lru_cache

from app.config.settings import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import get_vectorstore

settings = get_settings()


@lru_cache()
def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_K,
            "fetch_k": settings.RETRIEVER_FETCH_K,
            "lambda_mult": settings.LAMBDA_MULT
        }
    )
