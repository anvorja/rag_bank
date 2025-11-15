# app/core/config.py
"""
Application Configuration with Pydantic Settings
Implements Environment-based configuration with validation
"""
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Literal, Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with validation and type safety
    Follows 12-factor app configuration principles
    """

    # === PROJECT INFO ===
    PROJECT_NAME: str = "Borgian Bank RAG API"
    VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"

    # === API CONFIG ===
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ENABLE_DOCS: bool = True

    # === CORS ===
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8080",
        "http://127.0.0.1:5173"
        "http://127.0.0.1:5173"
        "http://127.0.0.1:8080"
    ]

    # === RAG MODE ===
    MODE: Literal["local", "cloud"] = "local"

    # === PATHS ===
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DOCS_PATH: Path = Field(default_factory=lambda: Path("./docs"))
    VECTORSTORE_PATH: Path = Field(default_factory=lambda: Path("./vectorstore"))
    LOGS_PATH: Path = Field(default_factory=lambda: Path("./logs/conversations.jsonl"))

    # === EMBEDDINGS ===
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # === LLM MODELS ===
    LOCAL_LLM_MODEL: str = "llama3.2:3b"
    CLOUD_LLM_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # === RETRIEVAL CONFIG ===
    RETRIEVER_K: int = Field(default=5, ge=1, le=20)
    RETRIEVER_FETCH_K: int = Field(default=20, ge=10, le=100)
    LAMBDA_MULT: float = Field(default=0.7, ge=0.0, le=1.0)

    # === CHUNKING CONFIG ===
    CHUNK_SIZE: int = Field(default=800, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=120, ge=0, le=500)

    # === API KEYS ===
    OPENAI_API_KEY: Optional[str] = Field(default=None)

    # === LOGGING ===
    LOG_LEVEL: str = "INFO"

    # === CONFIG (pydantic v2) ===
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_prefix="",
        extra="ignore"
    )

    # ------------------------------------------------------------
    # VALIDADORES
    # ------------------------------------------------------------

    @field_validator("DOCS_PATH", "VECTORSTORE_PATH", "LOGS_PATH", mode="before")
    def resolve_paths(cls, v):
        """Convert string or Path to absolute Path."""
        path = Path(v)
        if not path.is_absolute():
            return Path.cwd() / path
        return path

    @field_validator("OPENAI_API_KEY", mode="before")
    def normalize_api_key(cls, v: Optional[str]):
        """Convert empty strings to None."""
        return v or None

    @field_validator("CHUNK_OVERLAP")
    def validate_chunk_overlap(cls, v: int, info: FieldValidationInfo):
        """Ensure overlap < chunk_size."""
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v

    @field_validator("OPENAI_API_KEY")
    def ensure_api_key_if_cloud(cls, v: Optional[str], info: FieldValidationInfo):
        """Require API key when MODE=cloud."""
        if info.data.get("MODE") == "cloud" and not v:
            raise ValueError("OPENAI_API_KEY is required when MODE=cloud")
        return v

    # ------------------------------------------------------------

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for path_attr in ["DOCS_PATH", "VECTORSTORE_PATH"]:
            getattr(self, path_attr).mkdir(parents=True, exist_ok=True)

        # Logs directory
        self.LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    @property
    def database_url(self) -> str:
        """Generate database URL for future extensions"""
        return f"sqlite:///{self.PROJECT_ROOT}/data.db"


# Global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()