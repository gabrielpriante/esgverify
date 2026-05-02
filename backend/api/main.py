"""
ESGVerify — FastAPI application entry point.

This module wires together the API router, middleware, and startup
lifecycle. Keep it thin: configuration and business logic live elsewhere.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import analysis, health
from backend.core.config import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: runs once at startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application-level resources."""
    logger.info("ESGVerify starting", version=settings.app_version)
    # Future: warm up ClimateBERT model, connect to ChromaDB
    yield
    logger.info("ESGVerify shutting down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="ESGVerify",
        description="LLM-powered ESG claim analysis for sustainability professionals.",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI
    )

    # Allow the React dev server to call the API without CORS errors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route groups
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

    return app


app = create_app()
