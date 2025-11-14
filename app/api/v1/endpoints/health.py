# app/api/v1/endpoints/health.py
"""
Health Check Endpoint
Implements comprehensive system health monitoring
"""
import time
from datetime import datetime
from typing import Dict, Literal

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.utils.logger import get_logger
from app.schemas.chat import HealthResponse, ComponentStatus
from app.rag.vectorstore import get_vectorstore_stats, test_vectorstore
from app.rag.embeddings import test_embeddings, get_embedding_dimension
from app.rag.llm import validate_llm_connection, test_llm_generation

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check completo del sistema",
    description="""
    Verifica el estado de todos los componentes críticos del sistema RAG:

    Componentes verificados:
    - Vectorstore (existencia, integridad, dimensiones)
    - Modelo de embeddings (conectividad, dimensiones)
    - LLM (conectividad, generación)
    - Configuración del sistema
    - Documentos fuente

    Estados posibles:
    - healthy: Todos los componentes funcionando correctamente
    - degraded: Algunos componentes tienen problemas menores
    - unhealthy: Componentes críticos fallan
    """
)
async def health_check():
    """
    Ejecuta verificación completa de salud del sistema

    Returns:
        HealthResponse con estado general y detalle por componente
    """
    start_time = time.time()
    overall_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    components: Dict[str, ComponentStatus] = {}
    action_required = None

    try:
        logger.info("Starting comprehensive health check")

        # 1. Check Vectorstore
        vectorstore_status = await _check_vectorstore()
        components["vectorstore"] = vectorstore_status
        if vectorstore_status.status in ["warning", "error"]:
            overall_status = "degraded" if vectorstore_status.status == "warning" else "unhealthy"

        # 2. Check Embeddings
        embeddings_status = await _check_embeddings()
        components["embeddings"] = embeddings_status
        if embeddings_status.status == "error":
            overall_status = "unhealthy"

        # 3. Check LLM (only in local mode or if API key available)
        llm_status = await _check_llm()
        components["llm"] = llm_status
        if llm_status.status == "error" and settings.MODE == "local":
            overall_status = "degraded"  # Can still work in cloud mode

        # 4. Check Configuration
        config_status = await _check_configuration()
        components["configuration"] = config_status
        if config_status.status == "error":
            overall_status = "unhealthy"

        # 5. Check Documentation
        docs_status = await _check_documentation()
        components["documentation"] = docs_status
        # Documentation warnings don't affect overall status

        # 6. Set action required if needed
        if overall_status == "unhealthy":
            action_required = "Critical components failing. Check logs and configuration."
        elif overall_status == "degraded":
            action_required = "Some components need attention. Review component details."

        check_time = (time.time() - start_time) * 1000

        logger.info(
            "Health check completed",
            overall_status=overall_status,
            check_time_ms=round(check_time, 2),
            components_checked=len(components)
        )

        return HealthResponse(
            status=overall_status,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            mode=settings.MODE,
            timestamp=datetime.now(),
            components=components,
            action_required=action_required
        )

    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            mode=settings.MODE,
            timestamp=datetime.now(),
            components={"error": ComponentStatus(
                status="error",
                message=str(e),
                details={"exception_type": type(e).__name__}
            )},
            action_required="System health check failed. Check logs immediately."
        )


async def _check_vectorstore() -> ComponentStatus:
    """Check vectorstore health"""
    try:
        stats = get_vectorstore_stats()

        if stats["status"] == "not_found":
            return ComponentStatus(
                status="error",
                message="Vectorstore not found",
                details={
                    "path": str(settings.VECTORSTORE_PATH),
                    "action": "Run: python scripts/rebuild_vectorstore.py"
                }
            )

        if stats["status"] == "error":
            return ComponentStatus(
                status="error",
                message=f"Vectorstore error: {stats.get('error')}",
                details=stats
            )

        # Test vectorstore functionality
        if not test_vectorstore():
            return ComponentStatus(
                status="warning",
                message="Vectorstore exists but functionality test failed",
                details=stats
            )

        # Check if we have enough documents
        doc_count = stats.get("count", 0)
        if doc_count < 10:
            return ComponentStatus(
                status="warning",
                message=f"Low document count: {doc_count}",
                details=stats
            )

        return ComponentStatus(
            status="healthy",
            message=f"Vectorstore operational with {doc_count} chunks",
            details=stats
        )

    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Vectorstore check failed: {str(e)}",
            details={"exception_type": type(e).__name__}
        )


async def _check_embeddings() -> ComponentStatus:
    """Check embeddings model health"""
    try:
        # Test embeddings functionality
        if not test_embeddings():
            return ComponentStatus(
                status="error",
                message="Embeddings test failed",
                details={"test_type": "functionality"}
            )

        dimension = get_embedding_dimension()
        model_name = (
            settings.LOCAL_EMBEDDING_MODEL
            if settings.MODE == "local"
            else settings.OPENAI_EMBEDDING_MODEL
        )

        return ComponentStatus(
            status="healthy",
            message=f"Embeddings model operational",
            details={
                "model": model_name,
                "dimension": dimension,
                "mode": settings.MODE
            }
        )

    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Embeddings check failed: {str(e)}",
            details={"exception_type": type(e).__name__}
        )


async def _check_llm() -> ComponentStatus:
    """Check LLM health"""
    try:
        # Validate connection
        if not validate_llm_connection():
            error_msg = "LLM connection failed"
            if settings.MODE == "local":
                error_msg += " - Ensure Ollama is running and model is available"
            return ComponentStatus(
                status="error",
                message=error_msg,
                details={
                    "mode": settings.MODE,
                    "model": (
                        settings.LOCAL_LLM_MODEL
                        if settings.MODE == "local"
                        else settings.CLOUD_LLM_MODEL
                    )
                }
            )

        # Test generation
        if not test_llm_generation():
            return ComponentStatus(
                status="warning",
                message="LLM connected but generation test failed",
                details={"test_type": "generation"}
            )

        return ComponentStatus(
            status="healthy",
            message="LLM operational",
            details={
                "mode": settings.MODE,
                "model": (
                    settings.LOCAL_LLM_MODEL
                    if settings.MODE == "local"
                    else settings.CLOUD_LLM_MODEL
                )
            }
        )

    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"LLM check failed: {str(e)}",
            details={"exception_type": type(e).__name__}
        )


async def _check_configuration() -> ComponentStatus:
    """Check system configuration"""
    try:
        issues = []

        # Check required directories
        if not settings.VECTORSTORE_PATH.exists():
            issues.append("Vectorstore directory missing")

        if not settings.DOCS_PATH.exists():
            issues.append("Documents directory missing")

        # Check API keys for cloud mode
        if settings.MODE == "cloud" and not settings.OPENAI_API_KEY:
            issues.append("OpenAI API key required for cloud mode")

        # Check configuration consistency
        if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
            issues.append("Invalid chunk configuration: overlap >= size")

        if issues:
            return ComponentStatus(
                status="error",
                message=f"Configuration issues: {', '.join(issues)}",
                details={"issues": issues}
            )

        return ComponentStatus(
            status="healthy",
            message="Configuration valid",
            details={
                "mode": settings.MODE,
                "environment": settings.ENVIRONMENT,
                "chunk_size": settings.CHUNK_SIZE,
                "retriever_k": settings.RETRIEVER_K
            }
        )

    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Configuration check failed: {str(e)}",
            details={"exception_type": type(e).__name__}
        )


async def _check_documentation() -> ComponentStatus:
    """Check documentation availability"""
    try:
        if not settings.DOCS_PATH.exists():
            return ComponentStatus(
                status="warning",
                message="Documentation directory not found",
                details={"path": str(settings.DOCS_PATH)}
            )

        md_files = list(settings.DOCS_PATH.glob("*.md"))
        pdf_files = list(settings.DOCS_PATH.glob("*.pdf"))

        total_files = len(md_files) + len(pdf_files)

        if total_files == 0:
            return ComponentStatus(
                status="warning",
                message="No documentation files found",
                details={"path": str(settings.DOCS_PATH)}
            )

        return ComponentStatus(
            status="healthy",
            message=f"Documentation available: {total_files} files",
            details={
                "path": str(settings.DOCS_PATH),
                "markdown_files": len(md_files),
                "pdf_files": len(pdf_files)
            }
        )

    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Documentation check failed: {str(e)}",
            details={"exception_type": type(e).__name__}
        )


@router.get(
    "/quick",
    summary="Health check rápido"
)
async def quick_health_check():
    """
    Verificación rápida de salud sin tests exhaustivos
    Útil para los - Load balancers - y monitoreo básico
    """
    try:
        # Quick checks only
        vectorstore_exists = settings.VECTORSTORE_PATH.exists()

        return {
            "status": "healthy" if vectorstore_exists else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": settings.VERSION,
            "mode": settings.MODE,
            "vectorstore_exists": vectorstore_exists
        }
    except Exception as e:
        logger.error(f"Quick health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quick health check failed"
        )