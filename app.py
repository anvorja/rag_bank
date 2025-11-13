from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any

# ================================
# CONFIGURACIÓN
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no está definida en las variables de entorno")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DOCS_FOLDER = "./docs"  # Carpeta para Markdown y PDFs
VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "banco_rag"


# ================================
# MODELOS DE DATOS
# ================================
class QuestionRequest(BaseModel):
    question: str
    conversation_history: List[Dict[str, str]] = []  # Historial de conversación
    session_id: str = None  # ID de sesión para tracking
    is_first_message: bool = True  # Indica si es el primer mensaje de la sesión


class SourceInfo(BaseModel):
    id: int
    source: str
    section: str
    subsection: str
    content: str
    relevance_score: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any] = {}


# ================================
# INICIALIZAR FASTAPI
# ================================
app = FastAPI(
    title="Banco de BorjaM - Chatbot RAG API",
    description="API de chatbot con RAG para atención al cliente bancario",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# UTILIDADES
# ================================
def get_greeting():
    """Retorna saludo según la hora del día"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "¡Buenos días!"
    elif 12 <= hour < 19:
        return "¡Buenas tardes!"
    else:
        return "¡Buenas noches!"


def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Formatea el historial de conversación para el prompt"""
    if not history or len(history) == 0:
        return "Esta es la primera pregunta de la conversación."

    formatted_lines = []
    for msg in history:
        role = "Usuario" if msg.get("role") == "user" else "Asistente"
        content = msg.get("content", "")
        formatted_lines.append(f"{role}: {content}")

    return "\n".join(formatted_lines)


# ================================
# CARGAR O CREAR VECTORSTORE
# ================================
def get_vectorstore():
    """Carga el vectorstore existente o lanza error si no existe"""
    if not os.path.exists(VECTORSTORE_PATH) or not os.listdir(VECTORSTORE_PATH):
        raise Exception(
            "Vectorstore no encontrado. "
            "Por favor ejecuta 'python rebuild_vectorstore.py' primero."
        )

    print("Cargando vectorstore existente...")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    print(f"✓ Vectorstore cargado con {vectorstore._collection.count()} chunks")
    return vectorstore


# ================================
# INICIALIZAR RAG
# ================================
def init_rag():
    """Inicializa el sistema RAG con cadena optimizada"""
    vectorstore = get_vectorstore()

    # Retriever con configuración optimizada
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance para diversidad
        search_kwargs={
            "k": 5,  # Top 5 documentos más relevantes
            "fetch_k": 20,  # Fetch 20 para MMR
            "lambda_mult": 0.7  # Balance relevancia vs diversidad
        }
    )

    # LLM con mejor modelo
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # Un poco de creatividad pero consistente
        max_tokens=800
    )

    # Prompt mejorado con historial conversacional
    template = """Eres Sebastián, el asistente virtual de Bank BorjaM.

HISTORIAL DE LA CONVERSACIÓN:
{conversation_history}

INSTRUCCIONES IMPORTANTES:
1. Responde ÚNICAMENTE basándote en el contexto proporcionado
2. USA el historial de conversación para entender referencias implícitas y mantener coherencia
3. Si el usuario pregunta "¿cuánto cuesta?" después de hablar de un producto, infiere a qué se refiere
4. Si el usuario dice "¿y eso?" o "¿qué más?" refiriéndose a algo anterior, usa el contexto del historial
5. Mantén coherencia con tus respuestas anteriores en esta conversación
6. Si la información no está en el contexto, indica claramente: "No encuentro esa información en mi base de conocimientos. Te recomiendo contactar directamente con..."
7. Sé profesional pero cercano y empático
8. Proporciona información completa y estructurada
9. NUNCA inventes información, tasas de interés, requisitos o procedimientos
10. Para consultas sobre cuentas personales o transacciones, deriva a canales seguros

CONTEXTO RELEVANTE DE LA BASE DE CONOCIMIENTOS:
{context}

PREGUNTA ACTUAL DEL CLIENTE:
{question}

RESPUESTA (como Sebastián, asistente de Bank BorjaM):"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        """Formatea documentos con metadata"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            section = doc.metadata.get('seccion', 'General')
            formatted.append(f"[Fuente {i} - {section}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def rag_with_sources(
            question: str,
            is_first_message: bool = True,
            conversation_history: List[Dict[str, str]] = None
    ):
        """Ejecuta RAG y retorna respuesta con fuentes y contexto conversacional"""

        # Obtener documentos relevantes del vectorstore
        docs = retriever.invoke(question)

        # Formatear contexto de documentos
        context = format_docs(docs)

        # Formatear historial conversacional
        history_text = format_conversation_history(conversation_history or [])

        # Generar respuesta usando el LLM
        response = llm.invoke(
            prompt.format(
                conversation_history=history_text,
                context=context,
                question=question
            )
        )

        answer = response.content

        # Solo agregar saludo si es el primer mensaje de la sesión
        if is_first_message:
            greeting = get_greeting()
            # Verificar si ya tiene un saludo
            has_greeting = any(
                saludo in answer.lower()
                for saludo in ["buenos días", "buenas tardes", "buenas noches", "hola"]
            )

            if not has_greeting:
                answer = f"{greeting} Soy Sebastián, tu asistente virtual de Bank BorjaM. ¿En qué puedo ayudarte?\n\n{answer}"

        # Formatear fuentes con metadata enriquecido
        sources_info = []
        for i, doc in enumerate(docs, 1):
            source_name = doc.metadata.get('source', 'Documento')
            section = doc.metadata.get('seccion', 'N/A')
            subsection = doc.metadata.get('subseccion', 'N/A')

            # Truncar contenido para preview
            content_preview = doc.page_content
            if len(content_preview) > 300:
                content_preview = content_preview[:300] + "..."

            sources_info.append({
                "id": i,
                "source": os.path.basename(source_name),
                "section": section,
                "subsection": subsection,
                "content": content_preview,
                "relevance_score": str(doc.metadata.get('relevance_score', 'N/A'))
            })

        return answer, sources_info, docs

    return rag_with_sources


# Inicializar el sistema RAG al iniciar la aplicación
try:
    rag_system = init_rag()
    print("✓ Sistema RAG inicializado correctamente")
except Exception as e:
    print(f"⚠ Error inicializando RAG: {e}")
    rag_system = None


# ================================
# ENDPOINTS
# ================================
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Banco de BorjaM - Chatbot RAG API",
        "status": "running",
        "version": "1.0.0",
        "features": {
            "conversational_context": True,
            "session_management": True,
            "rag_retrieval": True
        },
        "endpoints": {
            "health": "/health",
            "ask": "/ask (POST)",
            "stats": "/stats"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Procesa una pregunta y retorna respuesta con fuentes"""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="Sistema RAG no disponible. Verifica que el vectorstore esté creado."
        )

    try:
        # Validar que la pregunta no esté vacía
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="La pregunta no puede estar vacía"
            )

        # Ejecutar RAG con contexto conversacional
        answer, sources, docs = rag_system(
            question=request.question,
            is_first_message=request.is_first_message,
            conversation_history=request.conversation_history
        )

        # Preparar metadata de la respuesta
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_sources_used": len(sources),
            "model": "gpt-4o-mini",
            "retrieval_type": "mmr",
            "is_first_message": request.is_first_message,
            "session_id": request.session_id,
            "conversation_length": len(request.conversation_history)
        }

        return AnswerResponse(
            answer=answer,
            sources=sources,
            metadata=metadata
        )

    except HTTPException:
        # Re-lanzar HTTPExceptions tal como están
        raise
    except Exception as e:
        # Capturar cualquier otro error y retornar 500
        print(f"Error procesando pregunta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la pregunta: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Verifica el estado del sistema"""
    vectorstore_exists = os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH)

    return {
        "status": "healthy" if rag_system else "degraded",
        "vectorstore_exists": vectorstore_exists,
        "vectorstore_path": VECTORSTORE_PATH,
        "rag_system_initialized": rag_system is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/stats")
async def get_stats():
    """Retorna estadísticas del vectorstore"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema RAG no disponible")

    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        return {
            "total_chunks": collection.count(),
            "collection_name": COLLECTION_NAME,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "vectorstore_path": VECTORSTORE_PATH,
            "retrieval_method": "mmr",
            "features": {
                "conversational_context": True,
                "session_tracking": True,
                "contextual_greeting": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# ENDPOINT DE DEBUG (OPCIONAL)
# ================================
@app.post("/debug/context")
async def debug_context(request: QuestionRequest):
    """
    Endpoint de debug para ver cómo se procesa el contexto conversacional
    SOLO PARA DESARROLLO - Eliminar en producción
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema RAG no disponible")

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
        )

        # Obtener documentos relevantes
        docs = retriever.invoke(request.question)

        # Formatear historial
        history_text = format_conversation_history(request.conversation_history or [])

        return {
            "question": request.question,
            "is_first_message": request.is_first_message,
            "session_id": request.session_id,
            "conversation_history": request.conversation_history,
            "formatted_history": history_text,
            "retrieved_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in docs
            ],
            "num_docs_retrieved": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# EJECUCIÓN
# ================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )