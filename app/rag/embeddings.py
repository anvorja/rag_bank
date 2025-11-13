# app/rag/embeddings.py
"""
Embeddings Factory and Abstraction Layer
Implements Strategy Pattern for different embedding providers
"""
from abc import ABC, abstractmethod
from functools import lru_cache

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return configured embeddings instance"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local HuggingFace embeddings provider"""

    def __init__(self):
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self._dimension = None

    def get_embeddings(self) -> Embeddings:
        """Get local HuggingFace embeddings"""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {e}")
            raise RuntimeError(f"Local embeddings initialization failed: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension for local model"""
        if self._dimension is None:
            # Common dimensions for popular models
            dimension_map = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-distilroberta-v1": 768,
            }
            self._dimension = dimension_map.get(self.model_name, 384)
        return self._dimension


class CloudEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider"""

    def __init__(self):
        self.model_name = settings.OPENAI_EMBEDDING_MODEL
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for cloud embeddings")

    def get_embeddings(self) -> Embeddings:
        """Get OpenAI embeddings"""
        try:
            return OpenAIEmbeddings(
                model=self.model_name,
                api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise RuntimeError(f"OpenAI embeddings initialization failed: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI model"""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model_name, 1536)


class EmbeddingFactory:
    """Factory class for creating embedding providers"""

    @staticmethod
    def get_provider() -> EmbeddingProvider:
        """Get embedding provider based on configuration"""
        if settings.MODE == "local":
            return LocalEmbeddingProvider()
        elif settings.MODE == "cloud":
            return CloudEmbeddingProvider()
        else:
            raise ValueError(f"Unknown embedding mode: {settings.MODE}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Get configured embeddings instance with caching
    Uses LRU cache to ensure single instance per process
    """
    provider = EmbeddingFactory.get_provider()
    embeddings = provider.get_embeddings()

    logger.info(
        f"Initialized embeddings provider",
        mode=settings.MODE,
        model=settings.LOCAL_EMBEDDING_MODEL if settings.MODE == "local" else settings.OPENAI_EMBEDDING_MODEL,
        dimension=provider.get_dimension()
    )

    return embeddings


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Get embedding dimension with caching"""
    provider = EmbeddingFactory.get_provider()
    return provider.get_dimension()


# En app/rag/embeddings.py - método test_embeddings
def test_embeddings() -> bool:
    """Test embeddings functionality"""
    try:
        embeddings = get_embeddings()
        test_text = "This is a test sentence for embedding validation."
        result = embeddings.embed_query(test_text)

        expected_dim = get_embedding_dimension()

        if hasattr(result, '__len__'):
            actual_dim = len(result)
        else:
            actual_dim = 1

        if actual_dim != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch",
                expected=expected_dim,
                actual=actual_dim
            )
            return False

        logger.info(f"Embeddings test successful", dimension=actual_dim)
        return True

    except Exception as e:
        logger.error(f"Embeddings test failed: {e}")
        return False