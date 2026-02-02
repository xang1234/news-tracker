"""Tests for preprocessing pipeline."""

import pytest

from src.ingestion.deduplication import Deduplicator, DeduplicationIndex
from src.ingestion.preprocessor import (
    BotDetector,
    Preprocessor,
    SpamDetector,
    TickerExtractor,
)
from src.ingestion.schemas import NormalizedDocument, Platform


class TestSpamDetector:
    """Tests for SpamDetector."""

    def test_normal_content_low_score(self, sample_document):
        """Normal content should have low spam score."""
        detector = SpamDetector()
        score, signals = detector.detect(sample_document)

        assert score < 0.3
        assert not detector.is_spam(sample_document)

    def test_spam_content_high_score(self, spam_document):
        """Spam content should have high spam score."""
        detector = SpamDetector()
        score, signals = detector.detect(spam_document)

        assert score >= 0.5
        assert len(signals) > 0

    def test_ticker_spam_detected(self, spam_document):
        """Many tickers in one post should be flagged."""
        detector = SpamDetector()
        score, signals = detector.detect(spam_document)

        ticker_signals = [s for s in signals if "ticker" in s.reason.lower()]
        assert len(ticker_signals) > 0

    def test_promotional_content_detected(self, spam_document):
        """Promotional content should be flagged."""
        detector = SpamDetector()
        score, signals = detector.detect(spam_document)

        promo_signals = [s for s in signals if "promotional" in s.reason.lower()]
        assert len(promo_signals) > 0

    def test_short_content_flagged(self, sample_document):
        """Very short content should be flagged."""
        sample_document.content = "Buy $NVDA"
        detector = SpamDetector()
        score, signals = detector.detect(sample_document)

        short_signals = [s for s in signals if "short" in s.reason.lower()]
        assert len(short_signals) > 0

    def test_custom_threshold(self, spam_document):
        """Custom threshold should be respected."""
        detector = SpamDetector(threshold=0.9)
        # Even spam might not hit 0.9 threshold
        is_spam = detector.is_spam(spam_document)

        detector_lenient = SpamDetector(threshold=0.3)
        is_spam_lenient = detector_lenient.is_spam(spam_document)

        # Lenient detector should catch more
        assert is_spam_lenient or not is_spam


class TestBotDetector:
    """Tests for BotDetector."""

    def test_verified_user_low_probability(self, sample_document):
        """Verified users should have low bot probability."""
        sample_document.author_verified = True
        detector = BotDetector()
        prob = detector.detect(sample_document)

        assert prob < 0.5

    def test_suspicious_username_high_probability(self, sample_document):
        """Suspicious usernames should have higher bot probability."""
        sample_document.author_name = "user12345678"  # Default pattern
        sample_document.platform = Platform.TWITTER
        sample_document.author_verified = False  # Not verified increases bot probability
        detector = BotDetector()
        prob = detector.detect(sample_document)

        assert prob > 0.3


class TestTickerExtractor:
    """Tests for TickerExtractor."""

    def test_cashtag_extraction(self):
        """Should extract $TICKER cashtags."""
        extractor = TickerExtractor()
        text = "Looking at $NVDA and $AMD for my portfolio"
        tickers = extractor.extract(text)

        assert "NVDA" in tickers
        assert "AMD" in tickers

    def test_company_name_mapping(self):
        """Should map company names to tickers."""
        extractor = TickerExtractor()
        text = "Nvidia and Taiwan Semiconductor are leading the AI chip race"
        tickers = extractor.extract(text)

        assert "NVDA" in tickers
        assert "TSM" in tickers

    def test_direct_ticker_mention(self):
        """Should find direct ticker mentions."""
        extractor = TickerExtractor()
        text = "INTC is undervalued compared to AMD"
        tickers = extractor.extract(text)

        assert "INTC" in tickers
        assert "AMD" in tickers

    def test_fuzzy_matching(self):
        """Should use fuzzy matching for variations."""
        extractor = TickerExtractor(fuzzy_threshold=80)
        text = "NVIDIA's earnings were impressive"
        tickers = extractor.extract(text)

        assert "NVDA" in tickers

    def test_semiconductor_relevance(self):
        """Should detect semiconductor-relevant content."""
        extractor = TickerExtractor()

        relevant = extractor.is_semiconductor_relevant(
            "The new 3nm process node is impressive"
        )
        assert relevant

        not_relevant = extractor.is_semiconductor_relevant(
            "The weather is nice today"
        )
        assert not not_relevant

    def test_no_false_positives(self):
        """Should not extract non-tickers."""
        extractor = TickerExtractor()
        text = "AI is transforming the industry"
        tickers = extractor.extract(text)

        # AI is not a tracked semiconductor ticker
        assert "AI" not in tickers


class TestPreprocessor:
    """Tests for Preprocessor."""

    def test_full_preprocessing(self, sample_document):
        """Should run all preprocessing steps."""
        preprocessor = Preprocessor()
        processed = preprocessor.process(sample_document)

        assert processed.spam_score >= 0
        assert processed.bot_probability >= 0
        assert len(processed.tickers_mentioned) > 0

    def test_batch_preprocessing(self, batch_documents):
        """Should process batches correctly."""
        preprocessor = Preprocessor()
        processed = preprocessor.process_batch(batch_documents)

        # Some might be filtered
        assert len(processed) <= len(batch_documents)
        # All processed docs should have scores
        for doc in processed:
            assert doc.spam_score >= 0
            assert doc.bot_probability >= 0


class TestDeduplicationIndex:
    """Tests for DeduplicationIndex."""

    def test_insert_unique(self, sample_document):
        """Should insert unique documents."""
        index = DeduplicationIndex()
        result = index.insert(sample_document)

        assert result is True
        assert index.size == 1

    def test_detect_duplicate(self, duplicate_documents):
        """Should detect near-duplicate content."""
        # Use threshold of 0.5 because MinHash with 3-word shingles
        # is strict - even small word changes (reports->reported) create different shingles
        index = DeduplicationIndex(threshold=0.5)

        # Insert first document
        doc1 = duplicate_documents[0]
        index.insert(doc1)

        # Second document is similar
        doc2 = duplicate_documents[1]
        result = index.check_duplicate(doc2)

        assert result.is_duplicate
        assert doc1.id in result.similar_doc_ids

    def test_different_content_not_duplicate(self, duplicate_documents):
        """Different content should not be flagged as duplicate."""
        index = DeduplicationIndex(threshold=0.8)

        # Insert first document
        doc1 = duplicate_documents[0]
        index.insert(doc1)

        # Third document is different
        doc3 = duplicate_documents[2]
        result = index.check_duplicate(doc3)

        assert not result.is_duplicate

    def test_insert_if_unique(self, sample_document):
        """insert_if_unique should return result and add to index."""
        index = DeduplicationIndex()

        result = index.insert_if_unique(sample_document)
        assert not result.is_duplicate
        assert index.size == 1

        # Same document again
        result2 = index.insert_if_unique(sample_document)
        # Should still be size 1 (not inserted again)
        assert index.size == 1


class TestDeduplicator:
    """Tests for Deduplicator."""

    def test_process_unique(self, sample_document):
        """Should process unique documents."""
        dedup = Deduplicator()
        result = dedup.process(sample_document)

        assert result is True
        assert dedup.stats["total_processed"] == 1
        assert dedup.stats["duplicates_found"] == 0

    def test_process_batch(self, batch_documents):
        """Should process batches correctly."""
        dedup = Deduplicator()
        unique = dedup.process_batch(batch_documents)

        # All should be unique (different content)
        assert len(unique) == len(batch_documents)
        assert dedup.stats["duplicates_found"] == 0

    def test_index_rotation(self):
        """Should rotate index when max size reached."""
        dedup = Deduplicator(max_index_size=5)

        # Add more than max_index_size documents
        for i in range(10):
            doc = NormalizedDocument(
                id=f"test_{i}",
                platform=Platform.TWITTER,
                timestamp="2024-01-01T00:00:00Z",
                author_id="author",
                author_name="name",
                content=f"Unique content about semiconductors number {i} with different words",
            )
            dedup.process(doc)

        # Index should have rotated
        assert dedup.stats["index_size"] < 10
