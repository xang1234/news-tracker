"""
FastAPI application factory.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import cleanup_dependencies
from src.api.routes import embed, health, search
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Embedding API starting up")
    yield
    logger.info("Embedding API shutting down")
    await cleanup_dependencies()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title="News Tracker Embedding API",
        description="""
API for generating embeddings from financial text using FinBERT and MiniLM models.

## Models

- **FinBERT**: 768-dimensional embeddings optimized for financial text
- **MiniLM**: 384-dimensional lightweight embeddings for fast processing

## Authentication

Requires `X-API-KEY` header for all requests except `/health`.
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": "internal"},
        )

    # Include routers
    app.include_router(embed.router, tags=["embeddings"])
    app.include_router(health.router, tags=["health"])
    app.include_router(search.router, tags=["search"])

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "News Tracker Embedding API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app
