"""
OpenTelemetry distributed tracing for the news-tracker pipeline.

Provides:
- setup_tracing(): Initialize TracerProvider with OTLP exporter
- get_tracer(): Get a named tracer instance
- traced(): Async context manager for creating spans
- inject_trace_context() / extract_trace_context(): Redis Streams propagation
- OtelLogProcessor: structlog processor that injects trace_id/span_id into logs

Trace context propagates through Redis Streams by encoding the W3C traceparent
header into message fields. This connects spans across worker boundaries:

    Ingestion (publish) → Processing (consume/publish) → Embedding (consume)

Usage:
    from src.observability.tracing import setup_tracing, get_tracer

    setup_tracing("news-tracker", "http://localhost:4317")
    tracer = get_tracer("my_service")

    with tracer.start_as_current_span("process_batch") as span:
        span.set_attribute("batch_size", len(batch))
        ...
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from opentelemetry import context, trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import StatusCode, Tracer
from opentelemetry.trace.propagation import get_current_span

logger = logging.getLogger(__name__)

# Module-level flag so callers can check without importing settings
_tracing_enabled = False

# W3C traceparent field name used in Redis Streams messages
TRACE_PARENT_FIELD = "traceparent"


def setup_tracing(
    service_name: str,
    otlp_endpoint: str | None = None,
    *,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """
    Initialize the OpenTelemetry TracerProvider.

    Uses OTLP gRPC exporter by default. Pass a custom exporter for testing
    (e.g., InMemorySpanExporter).

    Args:
        service_name: Logical service name (appears in Jaeger/Tempo).
        otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        exporter: Optional custom exporter (overrides OTLP).

    Returns:
        The configured TracerProvider.
    """
    global _tracing_enabled

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint or "http://localhost:4317",
            insecure=True,
        )
        # Production: batch for efficiency
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        # Custom exporter (e.g., InMemorySpanExporter for tests):
        # use SimpleSpanProcessor for synchronous export
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracing_enabled = True
    logger.info(
        "OpenTelemetry tracing initialized: service=%s endpoint=%s",
        service_name,
        otlp_endpoint or "(custom exporter)",
    )
    return provider


def get_tracer(name: str) -> Tracer:
    """
    Get a named tracer from the global TracerProvider.

    Safe to call even when tracing is not enabled — returns a no-op tracer.

    Args:
        name: Instrumentation scope name (typically module or service name).

    Returns:
        Tracer instance.
    """
    return trace.get_tracer(name)


def is_tracing_enabled() -> bool:
    """Check whether tracing has been initialized."""
    return _tracing_enabled


# ── Redis Streams trace context propagation ──────────────────────────


def inject_trace_context() -> dict[str, str]:
    """
    Extract the current trace context as a dict suitable for Redis XADD fields.

    Returns a dict with a single ``traceparent`` key in W3C format, or an
    empty dict if no active span exists.
    """
    span = get_current_span()
    ctx = span.get_span_context()

    if not ctx.is_valid:
        return {}

    # W3C traceparent format: version-trace_id-span_id-trace_flags
    traceparent = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"
    return {TRACE_PARENT_FIELD: traceparent}


def extract_trace_context(fields: dict[str, str]) -> Context | None:
    """
    Parse a W3C traceparent from Redis message fields and return an OTel Context.

    The returned context can be passed to ``tracer.start_as_current_span``
    via the ``context`` kwarg to link the new span as a child of the
    publishing span.

    Args:
        fields: Redis message fields (must contain ``traceparent`` key).

    Returns:
        OTel Context with a remote span context, or None if not present.
    """
    traceparent = fields.get(TRACE_PARENT_FIELD)
    if not traceparent:
        return None

    try:
        from opentelemetry.trace import SpanContext, TraceFlags

        parts = traceparent.split("-")
        if len(parts) != 4:
            return None

        trace_id = int(parts[1], 16)
        span_id = int(parts[2], 16)
        trace_flags = TraceFlags(int(parts[3], 16))

        remote_ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=trace_flags,
        )
        span = trace.NonRecordingSpan(remote_ctx)
        return trace.set_span_in_context(span)
    except Exception:
        logger.debug("Failed to parse traceparent: %s", traceparent)
        return None


# ── Convenience span helper ──────────────────────────────────────────


@contextmanager
def traced(
    tracer: Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
    parent_context: Context | None = None,
):
    """
    Context manager that creates a span and records exceptions.

    Usage:
        tracer = get_tracer("embedding")
        with traced(tracer, "process_batch", {"batch_size": 32}):
            ...

    Args:
        tracer: Tracer instance.
        name: Span name.
        attributes: Optional span attributes.
        parent_context: Optional parent context (from extract_trace_context).
    """
    kwargs: dict[str, Any] = {}
    if parent_context is not None:
        kwargs["context"] = parent_context

    with tracer.start_as_current_span(name, **kwargs) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


# ── Structlog processor for trace correlation ────────────────────────


def add_trace_context(
    logger_: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Structlog processor that injects trace_id and span_id into log entries.

    Add to structlog's processor chain so that every log message includes
    the active trace context for log-trace correlation.

    Usage:
        structlog.configure(processors=[
            ...,
            add_trace_context,
            ...,
        ])
    """
    span = get_current_span()
    ctx = span.get_span_context()

    if ctx.is_valid:
        event_dict["trace_id"] = f"{ctx.trace_id:032x}"
        event_dict["span_id"] = f"{ctx.span_id:016x}"

    return event_dict
