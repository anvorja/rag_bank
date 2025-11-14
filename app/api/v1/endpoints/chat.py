# app/api/v1/endpoints/chat.py
"""
Chat Endpoint - Main RAG functionality
Implements comprehensive error handling and request validation
"""
import time
from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.chat import QuestionRequest, AnswerResponse, ErrorResponse
from app.services.rag_services import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Dependency injection for RAG service
def get_rag_service() -> RAGService:
    """Dependency to get RAG service instance"""
    return RAGService()


@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    summary="Procesa una pregunta del cliente usando RAG",
    description="""
    Endpoint principal para procesar preguntas bancarias usando Retrieval-Augmented Generation.

    Características:
    - Búsqueda semántica en documentos bancarios
    - Respuestas contextuales basadas en información oficial
    - Soporte para conversaciones multi-turno
    - Validación de seguridad de contenido

    Limitaciones:
    - Texto mínimo: 1 carácter
    - Pregunta máxima: 2000 caracteres
    - Historial máximo: 20 mensajes
    - Tiempo límite: 30 segundos
    """
)
async def ask_question(
        request: QuestionRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Procesa una pregunta del cliente usando el pipeline RAG completo

    Args:
        request: Solicitud con pregunta y contexto opcional
        rag_service: Servicio RAG inyectado

    Returns:
        Respuesta estructurada con answer, sources y metadata

    Raises:
        HTTPException: Para errores de validación o procesamiento
    """
    start_time = time.time()

    try:
        logger.info(
            "Chat request received",
            question_length=len(request.question),
            session_id=request.session_id,
            has_history=bool(request.conversation_history),
            is_first_message=request.is_first_message
        )

        if len(request.question.strip()) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La pregunta debe tener al menos 1 carácter"
            )

        # Check for rate limiting (basic implementation)
        # In production, use Redis or similar for distributed rate limiting

        # Process with RAG service
        result = await rag_service.process_question(
            question=request.question,
            conversation_history=request.conversation_history,
            session_id=request.session_id,
            is_first_message=request.is_first_message,
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Chat request completed successfully",
            processing_time_ms=round(processing_time, 2),
            session_id=request.session_id,
            confidence=result.confidence
        )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error de validación: {str(e)}"
        )

    except RuntimeError as e:
        # Handle service errors
        logger.error(f"Service error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio temporalmente no disponible. Intenta nuevamente."
        )

    except Exception as e:
        # Handle unexpected errors
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"Unexpected error in chat endpoint: {e}",
            processing_time_ms=round(processing_time, 2),
            session_id=request.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del sistema"
        )


@router.post(
    "/stream",
    summary="Streaming de respuestas (implementación futura)",
    description="Endpoint para recibir respuestas en streaming. Actualmente no implementado."
)
async def ask_question_stream():
    """
    Endpoint placeholder para streaming de respuestas
    Implementación futura para mejor experiencia de usuario
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming de respuestas aún no implementado"
    )


@router.get(
    "/service-stats",
    summary="Estadísticas del servicio de chat"
)
async def get_chat_service_stats(rag_service: RAGService = Depends(get_rag_service)):
    """
    Obtiene estadísticas y métricas del servicio de chat

    Returns:
        Estadísticas del servicio RAG
    """
    try:
        stats = rag_service.get_service_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting chat service stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error obteniendo estadísticas del servicio"
        )