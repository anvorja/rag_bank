import os

from langchain_chroma import Chroma
from rag.embeddings import get_embeddings
from config.settings import settings


def get_vectorstore(recreate: bool = False) -> Chroma:
    """Carga o crea el vectorstore de Chroma"""
    embeddings = get_embeddings()

    if recreate or not os.path.exists(settings.VECTORSTORE_PATH):
        # Crear nuevo vectorstore
        return Chroma(
            persist_directory=settings.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name="banco_rag"
        )
    else:
        # Cargar existente
        return Chroma(
            persist_directory=settings.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name="banco_rag"
        )