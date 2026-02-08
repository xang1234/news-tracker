"""
FastAPI application factory.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import cleanup_dependencies, get_alert_broadcaster, stop_alert_broadcaster
from src.api.routes import alerts, embed, events, feedback, graph, health, search, sentiment, themes
from src.api.routes import ws_alerts
from src.api.routes.ws_alerts import set_broadcaster
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Embedding API starting up")

    # Initialize tracing if enabled
    settings = get_settings()
    if settings.tracing_enabled:
        from src.observability.tracing import setup_tracing

        setup_tracing(
            service_name=settings.otel_service_name,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        )
    if settings.ws_alerts_enabled:
        try:
            broadcaster = await get_alert_broadcaster()
            set_broadcaster(broadcaster)
            logger.info("WebSocket alert broadcaster started")
        except Exception as e:
            logger.warning("Failed to start WebSocket alert broadcaster: %s", e)

    yield

    logger.info("Embedding API shutting down")
    await stop_alert_broadcaster()
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

    # Request logging and tracing middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        from src.observability.tracing import get_tracer, is_tracing_enabled

        start_time = time.perf_counter()

        if is_tracing_enabled():
            tracer = get_tracer("news-tracker.api")
            with tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                },
            ) as span:
                response = await call_next(request)
                duration = time.perf_counter() - start_time
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.duration_ms", round(duration * 1000, 2))
        else:
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
    app.include_router(sentiment.router, tags=["sentiment"])
    app.include_router(health.router, tags=["health"])
    app.include_router(search.router, tags=["search"])
    app.include_router(themes.router, tags=["themes"])
    app.include_router(events.router, tags=["events"])
    app.include_router(alerts.router, tags=["alerts"])
    app.include_router(feedback.router, tags=["feedback"])
    app.include_router(graph.router, tags=["graph"])
    app.include_router(ws_alerts.router, tags=["websocket"])

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "News Tracker Embedding API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app
