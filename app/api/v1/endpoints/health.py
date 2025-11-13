# app/api/v1/endpoints/health.py
from fastapi import APIRouter
from datetime import datetime

from app.config.settings import get_settings
from app.rag.llm import verify_llm_connection
from app.rag.vectorstore import verify_vectorstore

router = APIRouter()
settings = get_settings()


@router.get("/", summary="System health check")
async def health_check():
    response = {
        "status": "healthy",
        "mode": settings.MODE,
        "timestamp": datetime.now().isoformat()
    }

    if not verify_vectorstore():
        response["vectorstore"] = {"status": "error"}
        response["status"] = "degraded"

    if settings.is_local:
        try:
            verify_llm_connection()
            response["llm"] = {"status": "healthy"}
        except:
            response["llm"] = {"status": "error"}
            response["status"] = "degraded"

    return response