# ================================
# IMPORTS
# ================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
from functools import lru_cache  # Añadido para caching

# Configuración
from config.settings import settings
from rag.llm import get_llm
from rag.vectorstore import get_vectorstore
from rag.embeddings import get_embeddings

# LangChain
from langchain_core.prompts import ChatPromptTemplate
import structlog


# ================================
# LOGGING CONFIGURACIÓN
# ================================
log = structlog.get_logger()
os.makedirs(Path(settings.LOGS_PATH).parent, exist_ok=True)


# ================================
# MODELOS DE DATOS
# ================================
class QuestionRequest(BaseModel):
    question: str
    conversation_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None
    is_first_message: bool = True


class SourceInfo(BaseModel):
    id: int
    source: str
    section: str
    subsection: str
    content: str
    chunk_id: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: float  # Score de relevancia promedio
    metadata: Dict[str, Any]


# ================================
# INICIALIZAR APLICACIÓN
# ================================
app = FastAPI(
    title="Bank BorjaM RAG - Local Edition",
    description="Asistente virtual con RAG local para banca privada",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ================================
# DEPENDENCIAS CON CACHE
# ================================
@lru_cache()
def get_rag_components():
    """Carga modelos una sola vez y los cachea"""
    log.info("Inicializando componentes RAG...")
    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        embeddings = get_embeddings()

        # Configurar retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVER_K,
                "fetch_k": settings.RETRIEVER_FETCH_K,
                "lambda_mult": settings.LAMBDA_MULT
            }
        )

        log.info("✓ Componentes cargados exitosamente")
        return vectorstore, llm, retriever
    except Exception as e:
        log.error(f"Error inicializando RAG: {e}")
        raise RuntimeError(f"Fallo inicialización: {e}")


# ================================
# FUNCIONES UTILITARIAS
# ================================
def get_greeting() -> str:
    """Saludo contextual por hora"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "¡Buenos días!"
    elif 12 <= hour < 19:
        return "¡Buenas tardes!"
    else:
        return "¡Buenas noches!"


def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Formatea historial para el prompt"""
    if not history:
        return "Esta es la primera pregunta de la conversación."

    return "\n".join([
        f"{'Usuario' if msg['role'] == 'user' else 'Asistente'}: {msg['content']}"
        for msg in history
    ])


def log_conversation(session_id: str, question: str, answer: str, sources: List[Dict]):
    """Log de conversaciones para análisis local"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": sources
        }
        with open(settings.LOGS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"No se pudo guardar log: {e}")


# ================================
# PROMPT OPTIMIZADO PARA BANCOS
# ================================
TEMPLATE = """Eres Sebastián, el asistente virtual de Bank BorjaM. Tu objetivo es ayudar a clientes con información precisa y segura.

REGISTRO DE CONVERSACIÓN ACTUAL:
{conversation_history}

CONTEXTO DOCUMENTAL RELEVANTE (últimos 3 mensajes):
{context}

PREGUNTA DEL CLIENTE:
{question}

INSTRUCCIONES CRÍTICAS:
1. Responde EXCLUSIVAMENTE con información del contexto documental
2. Para datos sensibles (saldo, números de cuenta), DERIVA: "Para tu seguridad, te recomiendo validar ingresando a tu portal o llamando al 01 8000 515 050"
3. Si no encuentras información específica, responde: "No tengo esa información en mis documentos. Te sugiero contactar a un asesor en 01 8000 515 050"
4. Sé breve pero completo. Usa listas y tablas cuando sea necesario
5. NO inventes tasas, números de teléfono ni procedimientos
6. Mantén un tono profesional y empático
7. Responde en español colombiano

RESPUESTA DE SEBASTIÁN:"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)


# ================================
# FUNCIÓN PRINCIPAL RAG
# ================================
def execute_rag(
        question: str,
        is_first_message: bool,
        conversation_history: List[Dict[str, str]],
        session_id: str
) -> tuple:
    """Ejecuta pipeline RAG completo"""
    _, llm, retriever = get_rag_components()

    # 1. Recuperar documentos relevantes
    docs = retriever.invoke(question)

    # 2. Calcular score de confianza
    avg_score = sum(
        doc.metadata.get('relevance_score', 0) for doc in docs
    ) / len(docs) if docs else 0

    # 3. Formatear contexto
    context = "\n\n---\n\n".join([
        f"[Fuente {i + 1} - {doc.metadata.get('sección', 'N/A')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # 4. Formatear historial (solo últimos 3 mensajes para eficiencia)
    recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
    history_text = format_conversation_history(recent_history)

    # 5. Generar respuesta
    response = llm.invoke(prompt.format(
        conversation_history=history_text,
        context=context,
        question=question
    ))

    answer = response.content

    # 6. Agregar saludo si es primer mensaje
    if is_first_message:
        greeting = get_greeting()
        if greeting.lower().split()[0] not in answer.lower():
            answer = f"{greeting} Soy Sebastián, tu asistente de Bank BorjaM.\n\n{answer}"

    # 7. Formatear fuentes
    sources = [
        {
            "id": i + 1,
            "source": doc.metadata.get('source', 'N/A'),
            "section": doc.metadata.get('sección', 'N/A'),
            "subsection": doc.metadata.get('subsección', 'N/A'),
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "chunk_id": doc.metadata.get('chunk_id', 'N/A')
        }
        for i, doc in enumerate(docs)
    ]

    return answer, sources, avg_score, docs


# ================================
# ENDPOINTS
# ================================
@app.get("/")
async def root():
    return {
        "message": "Bank BorjaM RAG API - Local Mode",
        "mode": settings.MODE,
        "status": "running",
        "health": "/health"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Pregunta vacía")

        # Ejecutar RAG
        answer, sources, confidence, _ = execute_rag(
            question=request.question,
            is_first_message=request.is_first_message,
            conversation_history=request.conversation_history,
            session_id=request.session_id or "anon"
        )

        # Log para análisis
        log_conversation(
            session_id=request.session_id or f"anon_{datetime.now().timestamp()}",
            question=request.question,
            answer=answer,
            sources=sources
        )

        return AnswerResponse(
            answer=answer,
            sources=[SourceInfo(**src) for src in sources],
            confidence=round(confidence, 2),
            metadata={
                "timestamp": datetime.now().isoformat(),
                "session_id": request.session_id,
                "mode": settings.MODE,
                "model": settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error procesando pregunta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del sistema")


@app.get("/health")
async def health_check():
    vectorstore, _, _ = get_rag_components()
    return {
        "status": "healthy",
        "mode": settings.MODE,
        "vectorstore_count": vectorstore._collection.count(),
        "models": {
            "embedding": settings.LOCAL_EMBEDDING_MODEL if settings.MODE == "local" else settings.OPENAI_EMBEDDING_MODEL,
            "llm": settings.LOCAL_LLM_MODEL if settings.MODE == "local" else settings.CLOUD_LLM_MODEL
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    vectorstore, _, _ = get_rag_components()
    return {
        "total_chunks": vectorstore._collection.count(),
        "collection_name": "banco_rag",
        "mode": settings.MODE,
        "config": {
            "retriever_k": settings.RETRIEVER_K,
            "chunk_size": 800,
            "chunk_overlap": 120
        }
    }


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
