from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

def get_embeddings() -> Embeddings:
    """Factory que devuelve embeddings según el modo"""
    if settings.MODE == "local":
        # Embeddings locales: rápidos, privados, sin costo
        return HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Cambia a 'cuda' si tienes GPU
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # Embeddings cloud: mejores para español complejo
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY requerida en modo cloud")
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )