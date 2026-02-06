"""Tests for PatternExtractor integration with Preprocessor."""

from datetime import datetime, timezone

from src.event_extraction.patterns import PatternExtractor
from src.ingestion.preprocessor import Preprocessor
from src.ingestion.schemas import NormalizedDocument, Platform


class TestPreprocessorEventsIntegration:
    """Tests for event extraction integration in the preprocessing pipeline."""

    def _make_doc(self, content: str) -> NormalizedDocument:
        """Create a test document with given content."""
        return NormalizedDocument(
            id="test_1",
            platform=Platform.NEWS,
            url="https://example.com/article",
            timestamp=datetime.now(timezone.utc),
            author_id="author_1",
            author_name="Test Author",
            content=content,
        )

    def test_events_disabled_no_extraction(self):
        """Events should not be extracted when disabled."""
        preprocessor = Preprocessor(enable_events=False)
        doc = self._make_doc(
            "NVIDIA launched the H200 GPU for data center AI workloads."
        )
        processed = preprocessor.process(doc)
        assert processed.events_extracted == []

    def test_events_enabled_no_extractor(self):
        """Events enabled but no extractor provided should not crash."""
        preprocessor = Preprocessor(enable_events=True, event_extractor=None)
        doc = self._make_doc(
            "NVIDIA launched the H200 GPU for data center AI workloads."
        )
        processed = preprocessor.process(doc)
        assert processed.events_extracted == []

    def test_events_enabled_with_extractor(self):
        """Events should be extracted when enabled with extractor."""
        extractor = PatternExtractor()
        preprocessor = Preprocessor(
            event_extractor=extractor,
            enable_events=True,
        )
        doc = self._make_doc(
            "NVIDIA launched the H200 GPU accelerator for AI workloads."
        )
        processed = preprocessor.process(doc)
        assert len(processed.events_extracted) > 0
        for ev in processed.events_extracted:
            assert "event_type" in ev
            assert "action" in ev
            assert "confidence" in ev

    def test_events_are_json_serializable(self):
        """Extracted events must be JSON-serializable for DB storage."""
        import json

        extractor = PatternExtractor()
        preprocessor = Preprocessor(
            event_extractor=extractor,
            enable_events=True,
        )
        doc = self._make_doc(
            "TSMC raised wafer prices by 5% for advanced nodes."
        )
        processed = preprocessor.process(doc)
        json_str = json.dumps(processed.events_extracted)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_events_combined_with_tickers_and_keywords(self):
        """Events extraction works alongside other preprocessing steps."""
        extractor = PatternExtractor()
        preprocessor = Preprocessor(
            event_extractor=extractor,
            enable_events=True,
        )
        doc = self._make_doc(
            "$NVDA NVIDIA raised revenue guidance to $22 billion for Q4 2026."
        )
        processed = preprocessor.process(doc)

        # Tickers should be extracted
        assert "NVDA" in processed.tickers_mentioned
        # Events should be extracted
        assert len(processed.events_extracted) > 0

    def test_extractor_error_handled_gracefully(self):
        """Extraction errors should be caught and logged, not raise."""

        class BrokenExtractor(PatternExtractor):
            def extract(self, doc):
                raise RuntimeError("simulated failure")

        preprocessor = Preprocessor(
            event_extractor=BrokenExtractor(),
            enable_events=True,
        )
        doc = self._make_doc("NVIDIA launched the H200 GPU.")
        processed = preprocessor.process(doc)
        # Should not crash, events list should be empty
        assert processed.events_extracted == []

    def test_irrelevant_text_no_events(self):
        """Irrelevant text should produce no events."""
        extractor = PatternExtractor()
        preprocessor = Preprocessor(
            event_extractor=extractor,
            enable_events=True,
        )
        doc = self._make_doc(
            "The weather in San Francisco is sunny today with clear skies."
        )
        processed = preprocessor.process(doc)
        assert processed.events_extracted == []
