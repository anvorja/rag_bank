from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config.settings import settings

def get_llm() -> BaseChatModel:
    """Factory que devuelve LLM según el modo"""
    if settings.MODE == "local":
        # LLM local con Ollama
        try:
            return ChatOllama(
                model=settings.LOCAL_LLM_MODEL,
                temperature=0.3,
                num_predict=800,  # Equivalente a max_tokens
                base_url="http://localhost:11434"  # Ollama default
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo conectar a Ollama. "
                f"Asegúrate de ejecutar: ollama pull {settings.LOCAL_LLM_MODEL}\n"
                f"Error: {str(e)}"
            )
    else:
        # LLM cloud
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY requerida en modo cloud")
        return ChatOpenAI(
            model=settings.CLOUD_LLM_MODEL,
            temperature=0.3,
            max_tokens=800,
            api_key=settings.OPENAI_API_KEY
        )