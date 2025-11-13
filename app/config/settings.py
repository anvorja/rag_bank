# app/config/settings.py
from functools import lru_cache
from typing import Literal
from pathlib import Path

from pydantic import Field, ValidationError, validator
from pydantic_settings import BaseSettings

# Al inicio del archivo, después de imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    MODE: Literal["local", "cloud"] = "local"

    # Paths (resueltos desde PROJECT_ROOT)
    DOCS_FOLDER: Path = Field(default=PROJECT_ROOT / "docs")
    VECTORSTORE_PATH: Path = Field(default=PROJECT_ROOT / "vectorstore")
    LOGS_PATH: Path = Field(default=PROJECT_ROOT / "logs" / "conversations.jsonl")

    # Embeddings
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LLM
    LOCAL_LLM_MODEL: str = "llama3.2:3b"
    CLOUD_LLM_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Agregado

    # Retrieval
    RETRIEVER_K: int = Field(default=5, ge=1, le=20)
    RETRIEVER_FETCH_K: int = Field(default=20, ge=5, le=100)
    LAMBDA_MULT: float = Field(default=0.7, ge=0.0, le=1.0)

    # Document processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Bank BorjaM RAG"
    VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    ENABLE_DOCS: bool = True

    # Cloud credentials
    OPENAI_API_KEY: str = Field(default="")

    @validator("VECTORSTORE_PATH", "DOCS_FOLDER")
    def ensure_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("LOGS_PATH")
    def ensure_log_path(cls, v: Path) -> Path:
        # Solo crea el directorio padre, no el archivo
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @validator("OPENAI_API_KEY")
    def validate_api_key(cls, v, values):
        if values.get("MODE") == "cloud" and not v:
            raise ValidationError("OPENAI_API_KEY required in cloud mode")
        return v

    @property
    def is_local(self) -> bool:
        return self.MODE == "local"

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        case_sensitive = True
        extra = "ignore"  # Ignora variables extra del .env


@lru_cache()
def get_settings() -> Settings:
    return Settings()