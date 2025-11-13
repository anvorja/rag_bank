from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Modo de operación: local (sin costos) - cloud (cuando escale)
    MODE: Literal["local", "cloud"] = "local"

    # Paths
    DOCS_FOLDER: str = "./docs"
    VECTORSTORE_PATH: str = "./vectorstore"
    LOGS_PATH: str = "./logs/conversations.jsonl"

    # Embeddings
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LLM
    LOCAL_LLM_MODEL: str = "llama3.2:3b"
    CLOUD_LLM_MODEL: str = "gpt-4o-mini"

    # Retrieval
    RETRIEVER_K: int = 5
    RETRIEVER_FETCH_K: int = 20
    LAMBDA_MULT: float = 0.7

    # API Keys (solo para cloud)
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()