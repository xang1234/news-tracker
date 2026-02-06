"""Tests for EventRecord schema."""

import json
from datetime import datetime, timezone
from uuid import uuid4

from src.event_extraction.schemas import EventRecord, VALID_EVENT_TYPES


class TestEventRecord:
    """Tests for EventRecord dataclass."""

    def test_create_minimal(self):
        """Test creation with only required fields."""
        event = EventRecord(
            doc_id="news_123",
            event_type="product_launch",
            action="launched",
            span_start=0,
            span_end=50,
            extractor_version="1.0.0",
        )
        assert event.doc_id == "news_123"
        assert event.event_type == "product_launch"
        assert event.action == "launched"
        assert event.actor is None
        assert event.object is None
        assert event.time_ref is None
        assert event.quantity is None
        assert event.tickers == []
        assert event.confidence == 0.7
        assert event.event_id  # Auto-generated UUID

    def test_create_full(self):
        """Test creation with all fields."""
        event = EventRecord(
            event_id="test-uuid",
            doc_id="news_456",
            event_type="capacity_expansion",
            actor="TSMC",
            action="is expanding",
            object="fab capacity",
            time_ref="2026-Q3",
            quantity="$40 billion",
            tickers=["TSM"],
            confidence=0.95,
            span_start=10,
            span_end=80,
            extractor_version="1.0.0",
        )
        assert event.event_id == "test-uuid"
        assert event.actor == "TSMC"
        assert event.object == "fab capacity"
        assert event.tickers == ["TSM"]
        assert event.confidence == 0.95

    def test_auto_generated_uuid(self):
        """Test that event_id is unique across instances."""
        e1 = EventRecord(
            doc_id="d1", event_type="price_change", action="raised",
            span_start=0, span_end=10, extractor_version="1.0.0",
        )
        e2 = EventRecord(
            doc_id="d2", event_type="price_change", action="raised",
            span_start=0, span_end=10, extractor_version="1.0.0",
        )
        assert e1.event_id != e2.event_id

    def test_created_at_default(self):
        """Test that created_at defaults to UTC now."""
        event = EventRecord(
            doc_id="d1", event_type="product_launch", action="launched",
            span_start=0, span_end=10, extractor_version="1.0.0",
        )
        assert event.created_at.tzinfo is not None
        # Should be within the last minute
        delta = datetime.now(timezone.utc) - event.created_at
        assert delta.total_seconds() < 60


class TestEventRecordSerialization:
    """Tests for to_dict/from_dict round-trip."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = EventRecord(
            event_id="uuid-1",
            doc_id="news_123",
            event_type="product_launch",
            actor="NVIDIA",
            action="launched",
            object="H200 GPU",
            time_ref="2026-Q1",
            quantity=None,
            tickers=["NVDA"],
            confidence=0.9,
            span_start=0,
            span_end=50,
            extractor_version="1.0.0",
        )
        d = event.to_dict()

        assert d["event_id"] == "uuid-1"
        assert d["doc_id"] == "news_123"
        assert d["event_type"] == "product_launch"
        assert d["actor"] == "NVIDIA"
        assert d["action"] == "launched"
        assert d["object"] == "H200 GPU"
        assert d["time_ref"] == "2026-Q1"
        assert d["quantity"] is None
        assert d["tickers"] == ["NVDA"]
        assert d["confidence"] == 0.9
        assert d["span_start"] == 0
        assert d["span_end"] == 50
        assert d["extractor_version"] == "1.0.0"
        assert "created_at" in d

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output can be JSON-serialized."""
        event = EventRecord(
            doc_id="d1", event_type="price_change", action="raised",
            tickers=["TSM"], span_start=0, span_end=10,
            extractor_version="1.0.0",
        )
        json_str = json.dumps(event.to_dict())
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "price_change"
        assert parsed["tickers"] == ["TSM"]

    def test_round_trip(self):
        """Test from_dict(to_dict()) preserves all fields."""
        original = EventRecord(
            event_id="uuid-rt",
            doc_id="news_rt",
            event_type="guidance_change",
            actor="AMD",
            action="raised",
            object="revenue guidance",
            time_ref="2026-Q4",
            quantity="$6.5 billion",
            tickers=["AMD"],
            confidence=0.85,
            span_start=5,
            span_end=60,
            extractor_version="1.0.0",
        )
        d = original.to_dict()
        restored = EventRecord.from_dict(d)

        assert restored.event_id == original.event_id
        assert restored.doc_id == original.doc_id
        assert restored.event_type == original.event_type
        assert restored.actor == original.actor
        assert restored.action == original.action
        assert restored.object == original.object
        assert restored.time_ref == original.time_ref
        assert restored.quantity == original.quantity
        assert restored.tickers == original.tickers
        assert restored.confidence == original.confidence
        assert restored.span_start == original.span_start
        assert restored.span_end == original.span_end
        assert restored.extractor_version == original.extractor_version

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict with only required fields."""
        data = {
            "doc_id": "d1",
            "event_type": "product_launch",
            "action": "launched",
            "span_start": 0,
            "span_end": 10,
            "extractor_version": "1.0.0",
        }
        event = EventRecord.from_dict(data)
        assert event.actor is None
        assert event.object is None
        assert event.tickers == []
        assert event.confidence == 0.7

    def test_from_dict_parses_iso_datetime(self):
        """Test from_dict correctly parses ISO datetime strings."""
        ts = "2026-01-15T10:30:00+00:00"
        data = {
            "doc_id": "d1",
            "event_type": "price_change",
            "action": "raised",
            "span_start": 0,
            "span_end": 10,
            "extractor_version": "1.0.0",
            "created_at": ts,
        }
        event = EventRecord.from_dict(data)
        assert event.created_at.year == 2026
        assert event.created_at.month == 1


class TestEventTypes:
    """Tests for event type validation."""

    def test_valid_event_types(self):
        """Test all documented event types are in the valid set."""
        expected = {
            "capacity_expansion", "capacity_constraint",
            "product_launch", "product_delay",
            "price_change", "guidance_change",
        }
        assert VALID_EVENT_TYPES == expected

    def test_event_record_accepts_valid_types(self):
        """Test EventRecord can be created with each valid type."""
        for event_type in VALID_EVENT_TYPES:
            event = EventRecord(
                doc_id="d1", event_type=event_type, action="test",
                span_start=0, span_end=5, extractor_version="1.0.0",
            )
            assert event.event_type == event_type
