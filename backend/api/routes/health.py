"""
Health check endpoints.

Used by monitoring, Docker healthchecks, and the frontend
to verify the API and its dependencies are reachable.
"""

import httpx
import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.config import settings

router = APIRouter()
logger = structlog.get_logger(__name__)


class HealthStatus(BaseModel):
    status: str
    version: str
    ollama_reachable: bool


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Returns API status and whether the Ollama service is reachable.
    A healthy response means the full pipeline can run.
    """
    ollama_ok = await _check_ollama()

    return HealthStatus(
        status="ok" if ollama_ok else "degraded",
        version=settings.app_version,
        ollama_reachable=ollama_ok,
    )


async def _check_ollama() -> bool:
    """Ping the Ollama API. Returns True if reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            return response.status_code == 200
    except Exception:
        logger.warning("Ollama not reachable", url=settings.ollama_base_url)
        return False
