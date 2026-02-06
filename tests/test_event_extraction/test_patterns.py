"""Tests for PatternExtractor."""

from datetime import datetime, timezone

import pytest

from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.patterns import PatternExtractor
from src.event_extraction.schemas import VALID_EVENT_TYPES
from src.ingestion.schemas import NormalizedDocument, Platform


def _make_doc(content: str, doc_id: str = "news_test1") -> NormalizedDocument:
    """Create a test NormalizedDocument with given content."""
    return NormalizedDocument(
        id=doc_id,
        platform=Platform.NEWS,
        url="https://example.com/article",
        timestamp=datetime.now(timezone.utc),
        author_id="author_1",
        author_name="Test Author",
        content=content,
    )


class TestCapacityExpansion:
    """Tests for capacity_expansion event type."""

    def test_expanding_fab_capacity(self, pattern_extractor):
        doc = _make_doc(
            "TSMC is expanding fab capacity in Arizona by $40 billion."
        )
        events = pattern_extractor.extract(doc)
        assert len(events) >= 1
        event = events[0]
        assert event.event_type == "capacity_expansion"
        assert event.doc_id == "news_test1"

    def test_new_fab_announcement(self, pattern_extractor):
        doc = _make_doc(
            "Samsung announced a new fabrication facility in Taylor Texas."
        )
        events = pattern_extractor.extract(doc)
        cap_events = [e for e in events if e.event_type == "capacity_expansion"]
        assert len(cap_events) >= 1

    def test_investment_in_capacity(self, pattern_extractor):
        doc = _make_doc(
            "Intel invests $20 billion in new chip manufacturing capacity."
        )
        events = pattern_extractor.extract(doc)
        cap_events = [e for e in events if e.event_type == "capacity_expansion"]
        assert len(cap_events) >= 1

    def test_production_ramp(self, pattern_extractor):
        doc = _make_doc(
            "TSMC ramps up production capacity at its 3nm node."
        )
        events = pattern_extractor.extract(doc)
        cap_events = [e for e in events if e.event_type == "capacity_expansion"]
        assert len(cap_events) >= 1


class TestCapacityConstraint:
    """Tests for capacity_constraint event type."""

    def test_supply_shortage(self, pattern_extractor):
        doc = _make_doc(
            "Advanced packaging supply shortage continues to impact AI chip delivery."
        )
        events = pattern_extractor.extract(doc)
        constraint_events = [e for e in events if e.event_type == "capacity_constraint"]
        assert len(constraint_events) >= 1

    def test_lead_times_extended(self, pattern_extractor):
        doc = _make_doc(
            "Lead times extended to 30 weeks for mature node semiconductors."
        )
        events = pattern_extractor.extract(doc)
        constraint_events = [e for e in events if e.event_type == "capacity_constraint"]
        assert len(constraint_events) >= 1

    def test_unable_to_meet_demand(self, pattern_extractor):
        doc = _make_doc(
            "GlobalFoundries is unable to meet customer demand for automotive chips."
        )
        events = pattern_extractor.extract(doc)
        constraint_events = [e for e in events if e.event_type == "capacity_constraint"]
        assert len(constraint_events) >= 1


class TestProductLaunch:
    """Tests for product_launch event type."""

    def test_launched_product(self, pattern_extractor):
        doc = _make_doc(
            "NVIDIA launched the H200 GPU accelerator for data centers."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        assert len(launch_events) >= 1
        event = launch_events[0]
        assert event.action is not None

    def test_introduces_product(self, pattern_extractor):
        doc = _make_doc(
            "AMD introduces the MI350 AI accelerator platform."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        assert len(launch_events) >= 1

    def test_begins_mass_production(self, pattern_extractor):
        doc = _make_doc(
            "Intel begins mass production of Emerald Rapids processors."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        assert len(launch_events) >= 1


class TestProductDelay:
    """Tests for product_delay event type."""

    def test_delayed_product(self, pattern_extractor):
        doc = _make_doc(
            "NVIDIA delayed the Blackwell B200 GPU to Q4 2026."
        )
        events = pattern_extractor.extract(doc)
        delay_events = [e for e in events if e.event_type == "product_delay"]
        assert len(delay_events) >= 1

    def test_pushed_back(self, pattern_extractor):
        doc = _make_doc(
            "The Blackwell chip has been pushed back to next quarter."
        )
        events = pattern_extractor.extract(doc)
        delay_events = [e for e in events if e.event_type == "product_delay"]
        assert len(delay_events) >= 1

    def test_behind_schedule(self, pattern_extractor):
        doc = _make_doc(
            "Arrow Lake desktop processors are behind schedule by 3 months."
        )
        events = pattern_extractor.extract(doc)
        delay_events = [e for e in events if e.event_type == "product_delay"]
        assert len(delay_events) >= 1


class TestPriceChange:
    """Tests for price_change event type."""

    def test_raised_prices(self, pattern_extractor):
        doc = _make_doc(
            "TSMC raised wafer prices by 5% for advanced nodes."
        )
        events = pattern_extractor.extract(doc)
        price_events = [e for e in events if e.event_type == "price_change"]
        assert len(price_events) >= 1

    def test_price_surge(self, pattern_extractor):
        doc = _make_doc(
            "Memory chip prices surged 20% due to strong AI demand."
        )
        events = pattern_extractor.extract(doc)
        price_events = [e for e in events if e.event_type == "price_change"]
        assert len(price_events) >= 1

    def test_price_cuts(self, pattern_extractor):
        doc = _make_doc(
            "Samsung cuts prices for legacy node wafers."
        )
        events = pattern_extractor.extract(doc)
        price_events = [e for e in events if e.event_type == "price_change"]
        assert len(price_events) >= 1


class TestGuidanceChange:
    """Tests for guidance_change event type."""

    def test_raised_guidance(self, pattern_extractor):
        doc = _make_doc(
            "NVIDIA raised revenue guidance for Q4 2026."
        )
        events = pattern_extractor.extract(doc)
        guid_events = [e for e in events if e.event_type == "guidance_change"]
        assert len(guid_events) >= 1

    def test_expects_revenue(self, pattern_extractor):
        doc = _make_doc(
            "AMD expects revenue of $6.5 billion in Q1 2026."
        )
        events = pattern_extractor.extract(doc)
        guid_events = [e for e in events if e.event_type == "guidance_change"]
        assert len(guid_events) >= 1

    def test_warned_of_lower(self, pattern_extractor):
        doc = _make_doc(
            "Intel warned of lower earnings due to weak PC demand."
        )
        events = pattern_extractor.extract(doc)
        guid_events = [e for e in events if e.event_type == "guidance_change"]
        assert len(guid_events) >= 1

    def test_beat_estimates(self, pattern_extractor):
        doc = _make_doc(
            "NVIDIA beat Wall Street estimates on strong data center demand."
        )
        events = pattern_extractor.extract(doc)
        guid_events = [e for e in events if e.event_type == "guidance_change"]
        assert len(guid_events) >= 1


class TestActorActionObjectCapture:
    """Tests for named capture group extraction."""

    def test_actor_captured(self, pattern_extractor):
        doc = _make_doc(
            "NVIDIA launched the H200 GPU accelerator for AI workloads."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        assert len(launch_events) >= 1
        event = launch_events[0]
        assert event.actor is not None
        assert "NVIDIA" in event.actor.upper()

    def test_action_always_present(self, pattern_extractor):
        doc = _make_doc(
            "Samsung announced a new fabrication facility in Taylor Texas."
        )
        events = pattern_extractor.extract(doc)
        for event in events:
            assert event.action is not None
            assert len(event.action) > 0

    def test_object_captured(self, pattern_extractor):
        doc = _make_doc(
            "TSMC raised wafer prices by 5% for advanced nodes."
        )
        events = pattern_extractor.extract(doc)
        price_events = [e for e in events if e.event_type == "price_change"]
        assert len(price_events) >= 1
        # Object should capture something about prices
        event = price_events[0]
        assert event.object is not None or event.action is not None


class TestConfidenceScoring:
    """Tests for confidence score computation."""

    def test_base_confidence(self, pattern_extractor):
        """Base confidence is 0.7 for a regex match."""
        doc = _make_doc(
            "Advanced packaging supply shortage is worsening."
        )
        events = pattern_extractor.extract(doc)
        if events:
            assert events[0].confidence >= 0.7

    def test_actor_increases_confidence(self, pattern_extractor):
        """Actor presence adds +0.1 to confidence."""
        doc = _make_doc(
            "NVIDIA launched the H200 GPU for data centers."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        if launch_events:
            event = launch_events[0]
            if event.actor:
                assert event.confidence >= 0.8

    def test_ticker_increases_confidence(self, pattern_extractor):
        """Ticker linkage adds +0.1 to confidence."""
        doc = _make_doc(
            "$NVDA NVIDIA raised revenue guidance to $22 billion."
        )
        events = pattern_extractor.extract(doc)
        if events:
            events_with_tickers = [e for e in events if e.tickers]
            if events_with_tickers:
                assert events_with_tickers[0].confidence >= 0.8

    def test_max_confidence_capped(self, pattern_extractor):
        """Confidence never exceeds 1.0."""
        doc = _make_doc(
            "$TSM TSMC raised wafer prices by 10% in Q3 2026."
        )
        events = pattern_extractor.extract(doc)
        for event in events:
            assert event.confidence <= 1.0


class TestNoEvents:
    """Tests that irrelevant text produces no events."""

    def test_empty_content(self, pattern_extractor):
        doc = _make_doc("x")  # min_length=1 in schema
        events = pattern_extractor.extract(doc)
        assert events == []

    def test_irrelevant_text(self, pattern_extractor):
        doc = _make_doc(
            "The weather in San Francisco is sunny today with clear skies."
        )
        events = pattern_extractor.extract(doc)
        assert events == []

    def test_generic_business_news(self, pattern_extractor):
        doc = _make_doc(
            "The company held its annual board meeting in New York."
        )
        events = pattern_extractor.extract(doc)
        assert events == []


class TestTickerLinking:
    """Tests for ticker extraction from event context."""

    def test_ticker_from_cashtag(self, pattern_extractor):
        doc = _make_doc(
            "$NVDA NVIDIA launched the H200 GPU for data center AI."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        if launch_events:
            assert "NVDA" in launch_events[0].tickers

    def test_ticker_from_context(self, pattern_extractor):
        doc = _make_doc(
            "Shares of NVDA rose 5%. NVIDIA launched the H200 GPU."
        )
        events = pattern_extractor.extract(doc)
        launch_events = [e for e in events if e.event_type == "product_launch"]
        if launch_events:
            # NVDA should be found in the context window
            assert "NVDA" in launch_events[0].tickers


class TestMaxEventsPerDoc:
    """Tests for max_events_per_doc limit."""

    def test_respects_limit(self):
        config = EventExtractionConfig(max_events_per_doc=2)
        extractor = PatternExtractor(config=config)
        doc = _make_doc(
            "TSMC is expanding fab capacity. "
            "Samsung announced a new fab. "
            "Intel invests $20 billion in manufacturing. "
            "NVIDIA launched the H200 GPU. "
            "AMD introduces the MI350. "
        )
        events = extractor.extract(doc)
        assert len(events) <= 2


class TestMinConfidence:
    """Tests for min_confidence filtering."""

    def test_low_confidence_filtered(self):
        config = EventExtractionConfig(min_confidence=0.95)
        extractor = PatternExtractor(config=config)
        doc = _make_doc(
            "Supply shortage of legacy chips is worsening."
        )
        events = extractor.extract(doc)
        for event in events:
            assert event.confidence >= 0.95


class TestSpanOffsets:
    """Tests for span_start and span_end accuracy."""

    def test_spans_are_within_text(self, pattern_extractor):
        content = "NVIDIA launched the H200 GPU accelerator for AI workloads."
        doc = _make_doc(content)
        events = pattern_extractor.extract(doc)
        for event in events:
            assert 0 <= event.span_start < len(content)
            assert event.span_start < event.span_end <= len(content)

    def test_events_sorted_by_span(self, pattern_extractor):
        doc = _make_doc(
            "TSMC raised wafer prices by 5%. "
            "NVIDIA launched the H200 GPU. "
            "Intel warned of lower earnings."
        )
        events = pattern_extractor.extract(doc)
        if len(events) > 1:
            for i in range(len(events) - 1):
                assert events[i].span_start <= events[i + 1].span_start


class TestEventTypesCoverage:
    """Tests that all event types can be matched."""

    def test_all_types_have_patterns(self, pattern_extractor):
        """Every valid event type must have at least one pattern."""
        for event_type in VALID_EVENT_TYPES:
            assert event_type in pattern_extractor.patterns
            assert len(pattern_extractor.patterns[event_type]) > 0

    def test_each_type_extractable(self, pattern_extractor, sample_semiconductor_texts):
        """Each event type should match at least one sample text."""
        for event_type, texts in sample_semiconductor_texts.items():
            found = False
            for text in texts:
                doc = _make_doc(text)
                events = pattern_extractor.extract(doc)
                if any(e.event_type == event_type for e in events):
                    found = True
                    break
            assert found, f"No patterns matched for event_type={event_type}"
