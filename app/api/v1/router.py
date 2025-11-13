# app/api/v1/router.py
"""
Main Router for API v1
Centralizes all endpoint routing with proper organization
"""
from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, stats

# Create main v1 router
router = APIRouter()

# Include endpoint routers with appropriate prefixes and tags
router.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}}
)

router.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}}
)

router.include_router(
    stats.router,
    prefix="/stats",
    tags=["statistics"],
    responses={404: {"description": "Not found"}}
)


# Add version info endpoint
@router.get("/", tags=["info"])
async def api_info():
    """
    API version and information endpoint
    """
    from app.core.config import settings

    return {
        "api_version": "v1",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "mode": settings.MODE,
        "environment": settings.ENVIRONMENT,
        "endpoints": {
            "chat": "/api/v1/chat/ask",
            "health": "/api/v1/health",
            "stats": "/api/v1/stats"
        },
        "documentation": "/docs"
    }