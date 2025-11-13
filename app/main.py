# app/main.py
"""
FastAPI Application Entry Point
Clean Architecture Implementation for Bank BorjaM RAG System
"""
import warnings
from pathlib import Path
import sys

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_application() -> FastAPI:
    """
    Application factory pattern - Best practice for FastAPI
    Implements Dependency Injection and Inversion of Control
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Asistente virtual RAG para atención bancaria con privacidad local",
        version=settings.VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None
    )

    # CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )

    # Include API router
    app.include_router(v1_router, prefix=settings.API_V1_STR)

    # Health check endpoint at root
    @app.get("/", tags=["health"])
    async def root():
        return {
            "message": f"{settings.PROJECT_NAME} - {settings.ENVIRONMENT} Environment",
            "version": settings.VERSION,
            "mode": settings.MODE,
            "health_check": "/api/v1/health",
            "docs": "/docs" if settings.ENABLE_DOCS else "disabled"
        }

    @app.on_event("startup")
    async def startup_event():
        """Initialization tasks on startup"""
        logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")

        # Check vectorstore exists
        if not settings.VECTORSTORE_PATH.exists() or not any(settings.VECTORSTORE_PATH.iterdir()):
            logger.warning("Vectorstore not found. Run: python scripts/rebuild_vectorstore.py")

        # Verify Ollama connectivity in local mode
        if settings.MODE == "local":
            try:
                from app.rag.llm import get_llm
                get_llm()
                logger.info("Ollama LLM connected successfully")
            except Exception as e:
                logger.error(f"Ollama connection failed: {e}")
                logger.warning("API will start but RAG will not work. Ensure 'ollama serve' is running.")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup tasks on shutdown"""
        logger.info(f"Shutting down {settings.PROJECT_NAME}")

    return app


# Create the application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )