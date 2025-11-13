# app/core/config.py
"""
Application Configuration with Pydantic Settings
Implements Environment-based configuration with validation
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from typing import List, Literal
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with validation and type safety
    Follows 12-factor app configuration principles
    """

    # === PROJECT INFO ===
    PROJECT_NAME: str = "Bank BorjaM RAG API"
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
        "http://127.0.0.1:5173"
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
    RETRIEVER_K: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    RETRIEVER_FETCH_K: int = Field(20, ge=10, le=100, description="Number of documents to fetch before MMR")
    LAMBDA_MULT: float = Field(0.7, ge=0.0, le=1.0, description="MMR lambda parameter for diversity")

    # === CHUNKING CONFIG ===
    CHUNK_SIZE: int = Field(800, ge=100, le=2000, description="Maximum chunk size in tokens")
    CHUNK_OVERLAP: int = Field(120, ge=0, le=500, description="Overlap between chunks")

    # === API KEYS ===
    OPENAI_API_KEY: str = Field("", description="OpenAI API key for cloud mode")

    # === LOGGING ===
    LOG_LEVEL: str = "INFO"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        env_prefix=""
    )

    @field_validator("DOCS_PATH", "VECTORSTORE_PATH", "LOGS_PATH", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Convert string paths to absolute Path objects"""
        path = Path(v)
        if not path.is_absolute():
            return Path.cwd() / path
        return path

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v, info):
        """Validate OpenAI API key when in cloud mode"""
        mode = info.data.get("MODE")
        if mode == "cloud" and not v:
            raise ValueError("OPENAI_API_KEY required when MODE=cloud")
        return v

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Ensure overlap is less than chunk size"""
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for path_attr in ["DOCS_PATH", "VECTORSTORE_PATH"]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        self.LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    @property
    def database_url(self) -> str:
        """Generate database URL for future extensions"""
        return f"sqlite:///{self.PROJECT_ROOT}/data.db"


# Global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()