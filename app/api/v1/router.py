# app/api/v1/router.py
from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, stats

api_v1_router = APIRouter()

api_v1_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_v1_router.include_router(health.router, prefix="/health", tags=["health"])
api_v1_router.include_router(stats.router, prefix="/stats", tags=["stats"])
