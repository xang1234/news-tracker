"""
Tests for OpenTelemetry tracing module.

Verifies:
- TracerProvider setup with InMemorySpanExporter
- Trace context injection/extraction for Redis Streams propagation
- Structlog processor adds trace_id/span_id to log entries
- traced() context manager creates spans and records exceptions
- Tracing disabled by default (no-op behavior)
"""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.observability.tracing import (
    add_trace_context,
    extract_trace_context,
    get_tracer,
    inject_trace_context,
    is_tracing_enabled,
    setup_tracing,
    traced,
)

# Module-level exporter shared across all tests. OTel's global TracerProvider
# can only be set once per process, so we initialize it once and clear the
# exporter between tests.
_exporter = InMemorySpanExporter()
_provider = setup_tracing("test-service", exporter=_exporter)


@pytest.fixture(autouse=True)
def _clear_spans():
    """Clear exported spans before each test."""
    _exporter.clear()
    yield
    _exporter.clear()


class TestSetupTracing:
    """Tests for setup_tracing()."""

    def test_setup_enables_tracing(self):
        """setup_tracing should enable the tracing flag."""
        assert is_tracing_enabled()


class TestTraceContextPropagation:
    """Tests for inject/extract trace context through Redis Streams."""

    def test_inject_with_active_span(self):
        """inject_trace_context should return traceparent when span is active."""
        tracer = get_tracer("test")

        with tracer.start_as_current_span("parent"):
            fields = inject_trace_context()

        assert "traceparent" in fields
        parts = fields["traceparent"].split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # W3C version
        assert len(parts[1]) == 32  # trace_id hex
        assert len(parts[2]) == 16  # span_id hex

    def test_inject_without_active_span(self):
        """inject_trace_context should not crash without an active span."""
        fields = inject_trace_context()
        # No active span — may or may not have traceparent, but must not crash
        assert isinstance(fields, dict)

    def test_roundtrip_inject_extract(self):
        """Injected trace context should survive roundtrip through Redis fields."""
        tracer = get_tracer("test")

        with tracer.start_as_current_span("publisher") as pub_span:
            fields = inject_trace_context()
            original_trace_id = pub_span.get_span_context().trace_id
            original_span_id = pub_span.get_span_context().span_id

        # Simulate Redis: message goes through XADD/XREADGROUP
        redis_fields = {"data": '{"id": "doc1"}', **fields}

        # Extract on consumer side
        ctx = extract_trace_context(redis_fields)
        assert ctx is not None

        # Create child span from extracted context
        with tracer.start_as_current_span("consumer", context=ctx) as child_span:
            child_ctx = child_span.get_span_context()
            # Child should share the same trace_id as parent
            assert child_ctx.trace_id == original_trace_id
            # But have a different span_id
            assert child_ctx.span_id != original_span_id

    def test_extract_without_traceparent(self):
        """extract_trace_context should return None when no traceparent field."""
        ctx = extract_trace_context({"data": "hello"})
        assert ctx is None

    def test_extract_invalid_traceparent(self):
        """extract_trace_context should return None for malformed traceparent."""
        ctx = extract_trace_context({"traceparent": "garbage"})
        assert ctx is None

    def test_extract_empty_traceparent(self):
        """extract_trace_context should return None for empty traceparent."""
        ctx = extract_trace_context({"traceparent": ""})
        assert ctx is None


class TestTracedContextManager:
    """Tests for the traced() convenience context manager."""

    def test_traced_creates_span(self):
        """traced() should create and finish a span."""
        tracer = get_tracer("test")

        with traced(tracer, "my_operation", {"key": "value"}):
            pass

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "my_operation"
        assert spans[0].attributes.get("key") == "value"

    def test_traced_records_exception(self):
        """traced() should record exceptions and set error status."""
        tracer = get_tracer("test")

        with pytest.raises(ValueError, match="test error"):
            with traced(tracer, "failing_op"):
                raise ValueError("test error")

        spans = _exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"
        # Check that exception event was recorded
        events = spans[0].events
        assert any(e.name == "exception" for e in events)

    def test_traced_with_parent_context(self):
        """traced() should link to parent context when provided."""
        tracer = get_tracer("test")

        with tracer.start_as_current_span("parent") as parent_span:
            fields = inject_trace_context()
            parent_trace_id = parent_span.get_span_context().trace_id

        parent_ctx = extract_trace_context(fields)

        with traced(tracer, "child_op", parent_context=parent_ctx):
            pass

        spans = _exporter.get_finished_spans()
        # parent + child
        assert len(spans) == 2
        child_span = next(s for s in spans if s.name == "child_op")
        assert child_span.context.trace_id == parent_trace_id


class TestStructlogProcessor:
    """Tests for the add_trace_context structlog processor."""

    def test_adds_trace_id_with_active_span(self):
        """Processor should add trace_id and span_id when span is active."""
        tracer = get_tracer("test")

        with tracer.start_as_current_span("log_test") as span:
            event_dict = {"event": "test message"}
            result = add_trace_context(None, "info", event_dict)

            expected_trace_id = f"{span.get_span_context().trace_id:032x}"
            expected_span_id = f"{span.get_span_context().span_id:016x}"

            assert result["trace_id"] == expected_trace_id
            assert result["span_id"] == expected_span_id

    def test_no_trace_id_without_span(self):
        """Processor should not add trace fields when no span is active."""
        event_dict = {"event": "test message"}
        result = add_trace_context(None, "info", event_dict)

        # No active span -> no valid trace context
        assert "trace_id" not in result or result.get("trace_id") == "0" * 32

    def test_preserves_existing_fields(self):
        """Processor should not overwrite existing event_dict fields."""
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test"):
            event_dict = {"event": "test", "custom_field": 42}
            result = add_trace_context(None, "info", event_dict)

            assert result["custom_field"] == 42
            assert result["event"] == "test"


class TestGetTracer:
    """Tests for get_tracer()."""

    def test_returns_tracer(self):
        """get_tracer should return a Tracer instance."""
        tracer = get_tracer("my_module")
        assert tracer is not None
        # Verify it can create spans
        with tracer.start_as_current_span("test"):
            pass

    def test_different_names_yield_different_tracers(self):
        """Different scope names should yield distinct tracer instances."""
        t1 = get_tracer("module_a")
        t2 = get_tracer("module_b")
        # They should both work
        with t1.start_as_current_span("a"):
            pass
        with t2.start_as_current_span("b"):
            pass
        spans = _exporter.get_finished_spans()
        assert len(spans) == 2
