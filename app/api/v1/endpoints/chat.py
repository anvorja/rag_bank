# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import QuestionRequest, AnswerResponse
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_rag_service() -> RAGService:
    return RAGService()


@router.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: QuestionRequest,
    service: RAGService = Depends(get_rag_service)
):
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        return await service.process_question(
            question=request.question,
            conversation_history=request.conversation_history,
            session_id=request.session_id,
            is_first_message=request.is_first_message
        )
    except Exception as e:
        logger.exception(f"RAG processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )
