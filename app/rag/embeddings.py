# app/rag/embeddings.py
from functools import lru_cache
from typing import Literal

from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config.settings import get_settings

settings = get_settings()


@lru_cache()
def get_embeddings() -> Embeddings:
    if settings.is_local:
        return HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required for cloud mode")

    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
