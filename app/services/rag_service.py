# app/services/rag_service.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import get_settings
from app.schemas.chat import AnswerResponse, SourceInfo
from app.rag.retriever import get_retriever
from app.rag.llm import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGService:
    def __init__(self):
        self.llm = get_llm()
        self.retriever = get_retriever()
        self.prompt = self._load_prompt()

    async def process_question(
            self,
            question: str,
            conversation_history: list[dict[str, str]],
            session_id: str | None,
            is_first_message: bool
    ) -> AnswerResponse:
        docs = self.retriever.invoke(question)

        if not docs:
            raise ValueError("No relevant documents found")

        context = self._format_context(docs)
        history = self._format_history(conversation_history)
        answer = self._generate_answer(question, context, history)

        if is_first_message:
            answer = self._add_greeting(answer)

        sources = self._format_sources(docs)
        confidence = self._calculate_confidence(docs)

        self._log_interaction(
            session_id or f"anon_{int(datetime.now().timestamp())}",
            question,
            answer,
            sources
        )

        return AnswerResponse(
            answer=answer,
            sources=[SourceInfo(**src) for src in sources],
            confidence=confidence,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "models": {
                    "llm": settings.LOCAL_LLM_MODEL if settings.is_local else settings.CLOUD_LLM_MODEL,
                    "embeddings": settings.LOCAL_EMBEDDING_MODEL if settings.is_local else settings.OPENAI_EMBEDDING_MODEL
                }
            }
        )

    def _load_prompt(self) -> str:
        return """Eres Sebastián, asistente de Bank BorjaM. Responde con información del contexto, Tu objetivo es ayudar a clientes con información precisa y segura.

HISTORIAL: {conversation_history}

CONTEXTO: {context}

PREGUNTA: {question}

INSTRUCCIONES:
1. Responde EXCLUSIVAMENTE con información del contexto documental proporcionado
2. Para datos sensibles (saldos, números de cuenta): "Para tu seguridad, valida esta información en tu portal bancario o llama al 01 8000 515 050"
3. Si la información NO está en el contexto: "No encuentro esa información específica en mis documentos. Te recomiendo contactar a un asesor en 01 8000 515 050"
4. Sé breve pero completo. Usa listas numeradas y tablas cuando sea apropiado
5. NO inventes tasas de interés, números de teléfono, ni procedimientos
6. Mantén un tono profesional y empático
7. Responde en español colombiano
8. Si mencionas productos o servicios, siempre incluye el contexto bancario

RESPUESTA:"""

    def _format_context(self, docs) -> str:
        return "\n\n---\n\n".join([
            f"[Fuente {i + 1} - {doc.metadata.get('section', 'N/A')}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

    def _format_history(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "Primera pregunta del usuario."
        recent = history[-3:]
        return "\n".join([
            f"{'Usuario' if msg.get('role') == 'user' else 'Asistente'}: {msg.get('content', '')}"
            for msg in recent
        ])

    def _generate_answer(self, question: str, context: str, history: str) -> str:
        prompt = self.prompt.format(
            conversation_history=history,
            context=context,
            question=question
        )
        return self.llm.invoke(prompt).content

    def _add_greeting(self, answer: str) -> str:
        hour = datetime.now().hour
        greeting = "Buenos días" if 5 <= hour < 12 else "Buenas tardes" if 12 < 19 else "Buenas noches"
        return f"{greeting}, soy Sebastián de Bank BorjaM.\n\n{answer}" if "buenos" not in answer.lower() else answer

    def _format_sources(self, docs) -> list[dict[str, Any]]:
        return [
            {
                "id": i + 1,
                "source": Path(doc.metadata.get('source', 'N/A')).name,
                "section": doc.metadata.get('section', 'N/A'),
                "subsection": doc.metadata.get('subsection', 'N/A'),
                "content": (doc.page_content[:400] + "...") if len(doc.page_content) > 400 else doc.page_content,
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i + 1}')
            }
            for i, doc in enumerate(docs)
        ]

    def _calculate_confidence(self, docs) -> float:
        return min(0.7 + (len(docs) * 0.05), 0.95) if docs else 0.0

    def _log_interaction(self, session_id: str, question: str, answer: str, sources: list[dict]):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "sources": sources
            }
            settings.LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with settings.LOGS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log: {e}")
