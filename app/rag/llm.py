# app/rag/llm.py
"""
Language Model Factory and Abstraction Layer
Implements Strategy Pattern for different LLM providers
"""
from abc import ABC, abstractmethod
from functools import lru_cache
import httpx

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return configured LLM instance"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate provider connection"""
        pass


class LocalLLMProvider(LLMProvider):
    """Ollama local LLM provider"""

    def __init__(self):
        self.model_name = settings.LOCAL_LLM_MODEL
        self.base_url = settings.OLLAMA_BASE_URL

    def get_llm(self) -> BaseChatModel:
        """Get Ollama LLM instance"""
        if not self.validate_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running and model '{self.model_name}' is available"
            )

        try:
            return ChatOllama(
                model=self.model_name,
                temperature=0.3,
                num_predict=800,  # Equivalent to max_tokens
                base_url=self.base_url,
                # Additional Ollama-specific parameters
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise RuntimeError(f"Ollama LLM initialization failed: {e}")

    def validate_connection(self) -> bool:
        """Validate Ollama connection and model availability"""
        try:
            # Check if Ollama is running
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    logger.error(f"Ollama server not responding: {response.status_code}")
                    return False

                # Check if required model is available
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]

                if self.model_name not in available_models:
                    logger.error(
                        f"Model '{self.model_name}' not found in Ollama",
                        available_models=available_models,
                        suggestion=f"Run: ollama pull {self.model_name}"
                    )
                    return False

                logger.info(f"Ollama connection validated", model=self.model_name)
                return True

        except httpx.TimeoutException:
            logger.error("Ollama connection timeout")
            return False
        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return False


class CloudLLMProvider(LLMProvider):
    """OpenAI cloud LLM provider"""

    def __init__(self):
        self.model_name = settings.CLOUD_LLM_MODEL
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for cloud LLM")

    def get_llm(self) -> BaseChatModel:
        """Get OpenAI LLM instance"""
        try:
            return ChatOpenAI(
                model=self.model_name,
                temperature=0.3,
                max_tokens=800,
                api_key=settings.OPENAI_API_KEY,
                timeout=30.0,
                max_retries=2
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise RuntimeError(f"OpenAI LLM initialization failed: {e}")

    def validate_connection(self) -> bool:
        """Validate OpenAI API connection"""
        try:
            llm = self.get_llm()
            # Simple test to validate API key and model
            test_response = llm.invoke("Test message")
            logger.info(f"OpenAI connection validated", model=self.model_name)
            return True
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            return False


class LLMFactory:
    """Factory class for creating LLM providers"""

    @staticmethod
    def get_provider() -> LLMProvider:
        """Get LLM provider based on configuration"""
        if settings.MODE == "local":
            return LocalLLMProvider()
        elif settings.MODE == "cloud":
            return CloudLLMProvider()
        else:
            raise ValueError(f"Unknown LLM mode: {settings.MODE}")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Get configured LLM instance with caching
    Uses LRU cache to ensure single instance per process
    """
    provider = LLMFactory.get_provider()
    llm = provider.get_llm()

    logger.info(
        f"Initialized LLM provider",
        mode=settings.MODE,
        model=settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL
    )

    return llm


def validate_llm_connection() -> bool:
    """Validate LLM connection without caching"""
    try:
        provider = LLMFactory.get_provider()
        return provider.validate_connection()
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        return False


def test_llm_generation() -> bool:
    """Test LLM generation capability"""
    try:
        llm = get_llm()
        test_prompt = "Responde brevemente: ¿Qué servicios bancarios conoces?"
        response = llm.invoke(test_prompt)

        if response and response.content:
            logger.info(f"LLM generation test successful", response_length=len(response.content))
            return True
        else:
            logger.error("LLM returned empty response")
            return False

    except Exception as e:
        logger.error(f"LLM generation test failed: {e}")
        return False