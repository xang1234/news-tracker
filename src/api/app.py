"""
FastAPI application factory.
"""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import cleanup_dependencies, get_alert_broadcaster, stop_alert_broadcaster
from src.api.middleware.timeout import TimeoutMiddleware
from src.api.routes import alerts, documents, embed, entities, events, events_extract, feedback, graph, health, keywords_route, ner, search, securities, sentiment, themes
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

    openapi_tags = [
        {"name": "health", "description": "Service health checks"},
        {"name": "embeddings", "description": "Batch text embedding with FinBERT/MiniLM"},
        {"name": "sentiment", "description": "Financial sentiment analysis"},
        {"name": "search", "description": "Semantic similarity search"},
        {"name": "themes", "description": "Theme listing, detail, metrics, and sentiment"},
        {"name": "events", "description": "Theme-linked events"},
        {"name": "events-extract", "description": "Event extraction playground"},
        {"name": "alerts", "description": "Alert management"},
        {"name": "feedback", "description": "User quality feedback"},
        {"name": "graph", "description": "Causal graph nodes, subgraphs, and propagation"},
        {"name": "documents", "description": "Document explorer"},
        {"name": "ner", "description": "Named entity recognition playground"},
        {"name": "keywords", "description": "Keyword extraction playground"},
        {"name": "entities", "description": "Entity directory and analytics"},
        {"name": "securities", "description": "Security master CRUD"},
        {"name": "websocket", "description": "Real-time WebSocket alerts"},
    ]

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
        openapi_tags=openapi_tags,
    )

    # Add CORS middleware (origins from CORS_ORIGINS env var, comma-separated)
    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Request timeout middleware (must be added before logging middleware
    # so timeout wraps the entire request lifecycle)
    if settings.request_timeout_seconds > 0:
        app.add_middleware(
            TimeoutMiddleware,
            timeout_seconds=settings.request_timeout_seconds,
        )

    # Request logging, correlation ID, and tracing middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        from src.observability.tracing import get_tracer, is_tracing_enabled

        # Correlation ID: use incoming header or generate a new one
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or str(uuid.uuid4())
        )

        # Bind to structlog contextvars for automatic log correlation
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.perf_counter()

        try:
            if is_tracing_enabled():
                tracer = get_tracer("news-tracker.api")
                with tracer.start_as_current_span(
                    f"{request.method} {request.url.path}",
                    attributes={
                        "http.method": request.method,
                        "http.url": str(request.url),
                        "http.route": request.url.path,
                        "http.request_id": request_id,
                    },
                ) as span:
                    response = await call_next(request)
                    duration = time.perf_counter() - start_time
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.duration_ms", round(duration * 1000, 2))
            else:
                response = await call_next(request)
                duration = time.perf_counter() - start_time

            # Add correlation ID to response headers
            response.headers["X-Request-ID"] = request_id

            logger.info(
                "HTTP request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()

    # Rate limiting (opt-in via RATE_LIMIT_ENABLED=true)
    if settings.rate_limit_enabled:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        from src.api.rate_limit import limiter

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    app.include_router(documents.router, tags=["documents"])
    app.include_router(ner.router, tags=["ner"])
    app.include_router(keywords_route.router, tags=["keywords"])
    app.include_router(events_extract.router, tags=["events-extract"])
    app.include_router(entities.router, tags=["entities"])
    app.include_router(securities.router, tags=["securities"])
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
