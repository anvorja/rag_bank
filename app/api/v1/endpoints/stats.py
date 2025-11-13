# app/api/v1/endpoints/stats.py
"""
Statistics Endpoint
Provides comprehensive system metrics and analytics
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import json
from pathlib import Path

from app.core.config import settings
from app.utils.logger import get_logger
from app.schemas.chat import StatsResponse, VectorstoreStats, ConfigStats, ModelInfo
from app.rag.vectorstore import get_vectorstore_stats
from app.rag.embeddings import get_embedding_dimension

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_model=StatsResponse,
    summary="Estadísticas completas del sistema",
    description="""
    Retorna métricas detalladas del sistema RAG:

    **Incluye:**
    - Estadísticas del vectorstore (documentos, dimensiones)
    - Configuración actual del sistema
    - Información de modelos utilizados
    - Métricas de rendimiento
    """
)
async def get_system_stats():
    """
    Obtiene estadísticas completas del sistema

    Returns:
        StatsResponse con métricas detalladas

    Raises:
        HTTPException: Si hay errores obteniendo las estadísticas
    """
    try:
        logger.info("Generating system statistics")

        # Get vectorstore statistics
        vectorstore_stats = get_vectorstore_stats()

        if vectorstore_stats["status"] != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vectorstore not available: {vectorstore_stats.get('error', 'Unknown error')}"
            )

        # Create vectorstore stats object
        vs_stats = VectorstoreStats(
            total_chunks=vectorstore_stats.get("count", 0),
            path=str(settings.VECTORSTORE_PATH),
            collection_name="banco_rag",
            embedding_dim=get_embedding_dimension()
        )

        # Create model info
        model_info = ModelInfo(
            llm_model=settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL,
            embedding_model=settings.LOCAL_EMBEDDING_MODEL if settings.MODE == "local" else settings.OPENAI_EMBEDDING_MODEL,
            embedding_dimension=get_embedding_dimension(),
            mode=settings.MODE
        )

        # Create configuration stats
        config_stats = ConfigStats(
            mode=settings.MODE,
            retriever_k=settings.RETRIEVER_K,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            models=model_info
        )

        logger.info(
            "System statistics generated successfully",
            total_chunks=vs_stats.total_chunks,
            mode=settings.MODE
        )

        return StatsResponse(
            status="success",
            vectorstore=vs_stats,
            config=config_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )


@router.get(
    "/usage",
    summary="Estadísticas de uso del sistema"
)
async def get_usage_stats():
    """
    Obtiene métricas de uso basadas en logs de conversaciones

    Returns:
        Estadísticas de uso y patrones de consulta
    """
    try:
        log_path = Path(settings.LOGS_PATH)

        if not log_path.exists():
            return {
                "status": "no_data",
                "message": "No usage logs found",
                "total_interactions": 0
            }

        # Parse interaction logs
        interactions = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    interactions.append(interaction)
                except json.JSONDecodeError:
                    continue

        if not interactions:
            return {
                "status": "no_data",
                "message": "No valid interactions found",
                "total_interactions": 0
            }

        # Calculate statistics
        total_interactions = len(interactions)
        unique_sessions = len(set(i.get("session_id") for i in interactions if i.get("session_id")))

        # Average processing time
        processing_times = [
            i.get("processing_time_ms", 0)
            for i in interactions
            if i.get("processing_time_ms")
        ]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        # Question length statistics
        question_lengths = [
            len(i.get("question", ""))
            for i in interactions
            if i.get("question")
        ]
        avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0

        # Recent activity (last 24 hours)
        from datetime import datetime, timedelta
        now = datetime.now()
        last_24h = now - timedelta(hours=24)

        recent_interactions = [
            i for i in interactions
            if i.get("timestamp") and datetime.fromisoformat(i["timestamp"].replace("Z", "+00:00")) > last_24h
        ]

        return {
            "status": "success",
            "total_interactions": total_interactions,
            "unique_sessions": unique_sessions,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "avg_question_length": round(avg_question_length, 1),
            "recent_activity": {
                "last_24h": len(recent_interactions)
            },
            "data_period": {
                "first_interaction": interactions[0].get("timestamp") if interactions else None,
                "last_interaction": interactions[-1].get("timestamp") if interactions else None
            }
        }

    except Exception as e:
        logger.error(f"Error generating usage stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving usage statistics: {str(e)}"
        )


@router.get(
    "/performance",
    summary="Métricas de rendimiento del sistema"
)
async def get_performance_stats():
    """
    Obtiene métricas de rendimiento y optimización

    Returns:
        Métricas de rendimiento del sistema
    """
    try:
        # Get basic system info
        import psutil
        import platform

        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Python process info
        process = psutil.Process()
        process_memory = process.memory_info()

        performance_stats = {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                    "available_gb": round(memory.available / (1024 ** 3), 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024 ** 3), 2),
                    "free_gb": round(disk.free / (1024 ** 3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1)
                }
            },
            "process": {
                "memory_mb": round(process_memory.rss / (1024 ** 2), 2),
                "memory_percent": round(process.memory_percent(), 2)
            },
            "rag_config": {
                "mode": settings.MODE,
                "chunk_size": settings.CHUNK_SIZE,
                "retriever_k": settings.RETRIEVER_K,
                "fetch_k": settings.RETRIEVER_FETCH_K
            }
        }

        return performance_stats

    except ImportError:
        # psutil not available
        return {
            "status": "limited",
            "message": "System monitoring unavailable (psutil not installed)",
            "rag_config": {
                "mode": settings.MODE,
                "chunk_size": settings.CHUNK_SIZE,
                "retriever_k": settings.RETRIEVER_K
            }
        }
    except Exception as e:
        logger.error(f"Error generating performance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving performance statistics: {str(e)}"
        )