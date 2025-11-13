# app/api/v1/endpoints/stats.py
from fastapi import APIRouter, HTTPException

from app.config.settings import get_settings
from app.rag.vectorstore import get_vectorstore
from app.rag.embeddings import get_embeddings

router = APIRouter()
settings = get_settings()


@router.get("/", summary="Vectorstore statistics")
async def get_stats():
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        
        embeddings = get_embeddings()
        sample = embeddings.embed_query("test")
        
        return {
            "total_chunks": count,
            "embedding_dim": len(sample),
            "config": {
                "mode": settings.MODE,
                "models": {
                    "llm": settings.LOCAL_LLM_MODEL if settings.is_local else settings.CLOUD_LLM_MODEL,
                    "embeddings": settings.LOCAL_EMBEDDING_MODEL if settings.is_local else settings.OPENAI_EMBEDDING_MODEL
                },
                "retriever_k": settings.RETRIEVER_K
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
