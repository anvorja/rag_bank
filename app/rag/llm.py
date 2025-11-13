# app/rag/llm.py
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings

settings = get_settings()


@lru_cache()
def get_llm() -> BaseChatModel:
    if settings.is_local:
        return ChatOllama(
            model=settings.LOCAL_LLM_MODEL,
            temperature=0.3,
            num_predict=800,
            base_url="http://localhost:11434"
        )

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY required for cloud mode")

    return ChatOpenAI(
        model=settings.CLOUD_LLM_MODEL,
        temperature=0.3,
        max_tokens=800,
        api_key=settings.OPENAI_API_KEY
    )


def verify_llm_connection() -> bool:
    try:
        llm = get_llm()
        llm.invoke("test")
        return True
    except Exception as e:
        raise RuntimeError(f"LLM connection failed: {e}")
