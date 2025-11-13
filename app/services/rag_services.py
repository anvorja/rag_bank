# app/services/rag_service.py
"""
Core RAG Service - Business Logic Layer
Implements comprehensive RAG pipeline with monitoring and optimization
"""
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.utils.logger import get_logger
from app.rag.llm import get_llm
from app.rag.retriever import get_retriever
from app.rag.embeddings import get_embedding_dimension
from app.schemas.chat import (
    AnswerResponse,
    SourceInfo,
    ResponseMetadata,
    ModelInfo,
    MessageHistory
)

logger = get_logger(__name__)


class RAGService:
    """
    Service class encapsulating all RAG business logic
    Follows Single Responsibility Principle and Clean Architecture
    """

    def __init__(self):
        """Initialize RAG service with all components"""
        start_time = time.time()

        try:
            self.llm = get_llm()
            self.retriever = get_retriever()
            self.prompt_template = self._build_prompt_template()

            initialization_time = (time.time() - start_time) * 1000
            logger.info(
                "RAG service initialized successfully",
                initialization_time_ms=round(initialization_time, 2)
            )

        except Exception as e:
            logger.error(f"RAG service initialization failed: {e}")
            raise RuntimeError(f"RAG service initialization failed: {e}")

    async def process_question(
            self,
            question: str,
            conversation_history: Optional[List[MessageHistory]] = None,
            session_id: Optional[str] = None,
            is_first_message: bool = True,
    ) -> AnswerResponse:
        """
        Main method to process a question through the RAG pipeline

        Args:
            question: The user's question
            conversation_history: Optional conversation context
            session_id: Optional session identifier
            is_first_message: Whether this is the first message

        Returns:
            AnswerResponse with answer, sources, and metadata
        """
        start_time = time.time()

        try:
            logger.info(
                "Processing question",
                question_length=len(question),
                session_id=session_id,
                is_first_message=is_first_message
            )

            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            docs = self.retriever.retrieve(question)
            retrieval_time = (time.time() - retrieval_start) * 1000

            if not docs:
                raise ValueError("No relevant documents found for the question")

            logger.info(
                "Document retrieval completed",
                docs_retrieved=len(docs),
                retrieval_time_ms=round(retrieval_time, 2)
            )

            # Step 2: Format context and history
            context = self._format_context(docs)
            history_text = self._format_conversation_history(conversation_history or [])

            # Step 3: Generate answer using LLM
            generation_start = time.time()
            prompt = self.prompt_template.format(
                conversation_history=history_text,
                context=context,
                question=question
            )

            response = self.llm.invoke(prompt)
            generation_time = (time.time() - generation_start) * 1000

            answer = response.content

            logger.info(
                "Answer generation completed",
                answer_length=len(answer),
                generation_time_ms=round(generation_time, 2)
            )

            # Step 4: Post-process answer
            if is_first_message:
                answer = self._add_contextual_greeting(answer)

            # Step 5: Format sources and calculate confidence
            sources = self._format_sources(docs)
            confidence = self._calculate_confidence(docs)

            # Step 6: Create response metadata
            total_time = (time.time() - start_time) * 1000
            metadata = self._create_metadata(
                session_id=session_id,
                processing_time=total_time,
                retrieval_stats={
                    "docs_retrieved": len(docs),
                    "retrieval_time_ms": round(retrieval_time, 2),
                    "generation_time_ms": round(generation_time, 2)
                }
            )

            # Step 7: Log interaction for analytics
            self._log_interaction(
                session_id=session_id or f"anon_{datetime.now().timestamp()}",
                question=question,
                answer=answer,
                sources_count=len(sources),
                metadata=metadata.model_dump()  # ✅ Cambiado a model_dump
            )

            logger.info(
                "Question processing completed",
                total_time_ms=round(total_time, 2),
                confidence=confidence
            )

            return AnswerResponse(
                answer=answer,
                sources=[SourceInfo(**src) for src in sources],
                confidence=confidence,
                metadata=metadata
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"RAG pipeline error: {e}",
                processing_time_ms=round(processing_time, 2),
                question_length=len(question)
            )
            raise

    @staticmethod
    def _build_prompt_template() -> ChatPromptTemplate:
        """Build optimized prompt template for banking domain"""
        template = """Eres Sebastián, el asistente virtual de Bank BorjaM. Tu objetivo es ayudar a clientes con información precisa y segura.

HISTORIAL DE CONVERSACIÓN (contexto):
{conversation_history}

CONTEXTO DOCUMENTAL RELEVANTE:
{context}

PREGUNTA DEL CLIENTE:
{question}

INSTRUCCIONES CRÍTICAS:
1. Responde EXCLUSIVAMENTE con información del contexto documental proporcionado
2. Para datos sensibles (saldos, números de cuenta): "Para tu seguridad, valida esta información en tu portal bancario o llama al 01 8000 515 050"
3. Si la información NO está en el contexto: "No encuentro esa información específica en mis documentos. Te recomiendo contactar a un asesor en 01 8000 515 050"
4. Sé breve pero completo. Usa listas numeradas y tablas cuando sea apropiado
5. NO inventes tasas de interés, números de teléfono, ni procedimientos
6. Mantén un tono profesional y empático
7. Responde en español colombiano
8. Si mencionas productos o servicios, siempre incluye el contexto bancario

RESPUESTA DE SEBASTIÁN:"""

        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def _format_context(docs) -> str:
        """Format retrieved documents for the prompt"""
        if not docs:
            return "No se encontró información relevante en los documentos."

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            section = doc.metadata.get('section') or doc.metadata.get('sección', 'N/A')
            formatted_docs.append(
                f"[Fuente {i} - {section}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(formatted_docs)

    @staticmethod
    def _format_conversation_history(history: List[MessageHistory]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "Esta es la primera pregunta de la conversación."

        # Limit to last 4 messages for efficiency and context window management
        recent = history[-4:] if len(history) > 4 else history

        formatted_messages = []
        for msg in recent:
            role = "Usuario" if msg.role == "user" else "Asistente"
            formatted_messages.append(f"{role}: {msg.content}")

        return "\n".join(formatted_messages)

    @staticmethod
    def _add_contextual_greeting(answer: str) -> str:
        """Add contextual greeting based on time of day"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            greeting = "¡Buenos días!"
        elif 12 <= hour < 19:
            greeting = "¡Buenas tardes!"
        else:
            greeting = "¡Buenas noches!"

        # Check if answer already contains a greeting
        greeting_indicators = ["buenos días", "buenas tardes", "buenas noches", "hola", "saludos"]
        if any(indicator in answer.lower() for indicator in greeting_indicators):
            return answer

        return f"{greeting} Soy Sebastián, tu asistente virtual de Bank BorjaM.\n\n{answer}"

    @staticmethod
    def _format_sources(docs) -> List[Dict[str, Any]]:
        """Format source documents for the response"""
        sources = []

        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            if len(content) > 400:
                content = content[:400] + "..."

            # Handle both old and new metadata key formats
            section = doc.metadata.get('section') or doc.metadata.get('sección', 'N/A')
            subsection = doc.metadata.get('subsection') or doc.metadata.get('subsección', 'N/A')
            source_file = doc.metadata.get('source', 'N/A')

            # Extract filename from path
            if source_file != 'N/A':
                source_file = Path(source_file).name

            sources.append({
                "id": i,
                "source": source_file,
                "section": section,
                "subsection": subsection,
                "content": content,
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i}'),
                "relevance_score": doc.metadata.get('relevance_score')
            })

        return sources

    @staticmethod
    def _calculate_confidence(docs) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not docs:
            return 0.0

        base_confidence = 0.6  # Base confidence for having documents

        # Boost for more documents (up to a limit)
        doc_boost = min(len(docs) * 0.05, 0.2)

        # Boost for content length (longer content often more informative)
        avg_content_length = sum(len(doc.page_content) for doc in docs) / len(docs)
        length_boost = min(avg_content_length / 1000 * 0.1, 0.1)

        # Boost for metadata quality (having sections, etc.)
        metadata_boost = 0
        for doc in docs:
            if doc.metadata.get('section') or doc.metadata.get('sección'):
                metadata_boost += 0.02
        metadata_boost = min(metadata_boost, 0.1)

        final_confidence = min(
            base_confidence + doc_boost + length_boost + metadata_boost,
            0.95  # Cap at 95%
        )

        return round(final_confidence, 2)

    @staticmethod
    def _create_metadata(
            session_id: Optional[str],
            processing_time: float,
            retrieval_stats: Dict[str, Any]
    ) -> ResponseMetadata:
        """Create comprehensive response metadata"""

        model_info = ModelInfo(
            llm_model=settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL,
            embedding_model=settings.LOCAL_EMBEDDING_MODEL if settings.MODE == "local" else settings.OPENAI_EMBEDDING_MODEL,
            embedding_dimension=get_embedding_dimension(),
            mode=settings.MODE
        )

        return ResponseMetadata(
            timestamp=datetime.now(),
            session_id=session_id,
            processing_time_ms=round(processing_time, 2),
            model_info=model_info,
            retrieval_stats=retrieval_stats
        )

    @staticmethod
    def _log_interaction(
            session_id: str,
            question: str,
            answer: str,
            sources_count: int,
            metadata: Dict[str, Any]
    ):
        """Log interaction for analytics and monitoring"""
        interaction_logger = get_logger(__name__)
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "sources_count": sources_count,
                "processing_time_ms": metadata.get("processing_time_ms"),
                "confidence": metadata.get("confidence"),
                "model_info": metadata.get("model_info")
            }

            log_path = Path(settings.LOGS_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            interaction_logger.warning(f"Failed to log interaction: {e}")

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health metrics"""
        try:
            retriever_stats = self.retriever.get_retrieval_stats()

            return {
                "status": "healthy",
                "retriever": retriever_stats,
                "models": {
                    "llm": settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL,
                    "embeddings": settings.LOCAL_EMBEDDING_MODEL if settings.MODE == "local" else settings.OPENAI_EMBEDDING_MODEL
                },
                "mode": settings.MODE
            }
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"status": "error", "error": str(e)}