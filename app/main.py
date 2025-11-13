# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config.settings import get_settings
from app.utils.logger import get_logger
from app.rag.llm import verify_llm_connection
from app.rag.vectorstore import verify_vectorstore

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} ({settings.ENVIRONMENT})")

    if not verify_vectorstore():
        logger.warning("Vectorstore not found. Run: python scripts/rebuild_vectorstore.py")

    if settings.is_local:
        try:
            verify_llm_connection()
            logger.info("Ollama LLM connected")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(api_v1_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["system"])
    async def root():
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "mode": settings.MODE,
            "environment": settings.ENVIRONMENT,
            "health": f"{settings.API_V1_STR}/health"
        }

    return app


app = create_app()
