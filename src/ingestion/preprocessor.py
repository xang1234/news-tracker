"""
Document preprocessing pipeline.

Implements spam detection, bot detection, ticker extraction, NER, and
content enrichment. All preprocessing is rule-based with ML-ready
structure for future model integration.

Components:
- SpamDetector: Rule-based spam scoring
- BotDetector: Bot probability estimation
- TickerExtractor: Ticker extraction with fuzzy matching
- NERService: Named entity recognition (optional)
- Preprocessor: Orchestrates all preprocessing steps
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from src.config.settings import get_settings
from src.config.tickers import (
    COMPANY_TO_TICKER,
    SEMICONDUCTOR_KEYWORDS,
    SEMICONDUCTOR_TICKERS,
)
from src.ingestion.schemas import NormalizedDocument, Platform

if TYPE_CHECKING:
    from src.ner.service import NERService

logger = logging.getLogger(__name__)


@dataclass
class SpamSignal:
    """A single spam detection signal with score and reason."""

    score: float
    reason: str


class SpamDetector:
    """
    Rule-based spam detection.

    Analyzes documents for spam indicators and returns a score from 0.0-1.0.
    Documents with spam_score > threshold (default 0.7) should be filtered.

    ML-Ready Structure:
        - Each rule produces a signal with score and reason
        - Signals can be used as features for future ML models
        - Easy to add new rules or adjust weights

    Common spam patterns detected:
        - Ticker spam (>5 tickers in one post)
        - Emoji spam (excessive rocket/moon emojis)
        - Promotional content (discord/telegram links)
        - ALL CAPS content
        - Very short content
        - URL spam
    """

    def __init__(
        self,
        threshold: float | None = None,
    ):
        """
        Initialize spam detector.

        Args:
            threshold: Spam threshold (0.0-1.0), default from settings
        """
        settings = get_settings()
        self.threshold = threshold or settings.spam_threshold

        # Promotional keywords (case-insensitive)
        self.promo_patterns = [
            r'\b(join|check out)\s+(my|our)\s+(discord|telegram|channel)\b',
            r'\bDM\s+me\b',
            r'\blink\s+in\s+bio\b',
            r'\bfree\s+(signals?|picks?|tips?)\b',
            r'\bguaranteed\s+(gains?|profits?|returns?)\b',
            r'\b\d+%\s+guaranteed\b',
        ]

    def detect(self, doc: NormalizedDocument) -> tuple[float, list[SpamSignal]]:
        """
        Analyze document for spam indicators.

        Args:
            doc: Document to analyze

        Returns:
            Tuple of (spam_score, list of signals)
        """
        signals: list[SpamSignal] = []

        # Platform-specific signals
        if doc.platform == Platform.TWITTER:
            signals.extend(self._twitter_signals(doc))

        # Content signals (all platforms)
        signals.extend(self._content_signals(doc))

        # Author signals
        signals.extend(self._author_signals(doc))

        # Calculate total score (capped at 1.0)
        total_score = min(1.0, sum(s.score for s in signals))

        return total_score, signals

    def _twitter_signals(self, doc: NormalizedDocument) -> list[SpamSignal]:
        """Twitter-specific spam signals."""
        signals = []

        # Low follower count
        if doc.author_followers is not None and doc.author_followers < 10:
            signals.append(SpamSignal(0.2, "Very low follower count (<10)"))

        # Ticker spam (many tickers in one tweet)
        if len(doc.tickers_mentioned) > 5:
            signals.append(SpamSignal(0.3, f"Ticker spam ({len(doc.tickers_mentioned)} tickers)"))

        # Emoji spam (excessive bullish emojis)
        rocket_count = doc.content.count("🚀")
        if rocket_count > 3:
            signals.append(SpamSignal(0.2, f"Emoji spam ({rocket_count} rockets)"))

        return signals

    def _content_signals(self, doc: NormalizedDocument) -> list[SpamSignal]:
        """Content-based spam signals."""
        signals = []
        content = doc.content

        # Very short content
        if len(content) < 20:
            signals.append(SpamSignal(0.2, "Very short content (<20 chars)"))

        # ALL CAPS
        if content.isupper() and len(content) > 20:
            signals.append(SpamSignal(0.15, "ALL CAPS content"))

        # Promotional patterns
        for pattern in self.promo_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(SpamSignal(0.4, f"Promotional pattern: {pattern}"))
                break  # One promotional signal is enough

        # Excessive URLs
        url_count = len(doc.urls_mentioned)
        if url_count > 3:
            signals.append(SpamSignal(0.2, f"Many URLs ({url_count})"))

        # Repetitive content (same word repeated many times)
        words = content.lower().split()
        if words:
            word_freq = max((words.count(w) / len(words)) for w in set(words))
            if word_freq > 0.3 and len(words) > 10:
                signals.append(SpamSignal(0.2, "Repetitive content"))

        return signals

    def _author_signals(self, doc: NormalizedDocument) -> list[SpamSignal]:
        """Author-based spam signals."""
        signals = []

        # Suspicious username patterns (common bot patterns)
        username = doc.author_name.lower()

        # Default Twitter username pattern: word + 6+ numbers
        if re.match(r'^[a-z]+\d{6,}$', username):
            signals.append(SpamSignal(0.3, "Default username pattern"))

        # Crypto/trading guru patterns
        if any(x in username for x in ['crypto_', 'trading_', 'forex_', '_signals']):
            signals.append(SpamSignal(0.2, "Suspicious username keywords"))

        return signals

    def is_spam(self, doc: NormalizedDocument) -> bool:
        """Quick check if document is spam."""
        score, _ = self.detect(doc)
        return score >= self.threshold


class BotDetector:
    """
    Bot probability estimation.

    Uses heuristics to estimate if an author is likely a bot.
    Returns probability from 0.0 (definitely human) to 1.0 (definitely bot).

    NOTE: For production, consider training a classifier on labeled data.
    """

    def detect(self, doc: NormalizedDocument) -> float:
        """
        Estimate bot probability.

        Args:
            doc: Document to analyze

        Returns:
            Bot probability (0.0-1.0)
        """
        signals: list[float] = []

        # Platform-specific signals
        if doc.platform == Platform.TWITTER:
            signals.extend(self._twitter_signals(doc))

        # Generic signals
        signals.extend(self._generic_signals(doc))

        # Calculate probability (average of signals, capped at 0-1)
        if not signals:
            return 0.0

        prob = sum(signals) / len(signals)
        return max(0.0, min(1.0, prob))

    def _twitter_signals(self, doc: NormalizedDocument) -> list[float]:
        """Twitter-specific bot signals."""
        signals = []

        # Default username pattern
        if re.match(r'^[A-Za-z]+\d{6,}$', doc.author_name):
            signals.append(0.6)

        # Very low engagement ratio (followers but no engagement)
        if doc.author_followers and doc.author_followers > 1000:
            engagement = doc.engagement.engagement_score
            if engagement < 5:
                signals.append(0.4)

        # Verified accounts are less likely to be bots
        if doc.author_verified:
            signals.append(-0.5)

        return signals

    def _generic_signals(self, doc: NormalizedDocument) -> list[float]:
        """Generic bot signals."""
        signals = []

        # Very repetitive/templated content structure
        content = doc.content
        if content.count("!") > 5:
            signals.append(0.2)

        return signals


class TickerExtractor:
    """
    Ticker extraction with fuzzy matching.

    Extracts stock ticker symbols from text using:
    1. Cashtag pattern matching ($NVDA)
    2. Company name lookup (Nvidia -> NVDA)
    3. Fuzzy matching for variations (NVIDIA Corp -> NVDA)
    """

    def __init__(
        self,
        fuzzy_threshold: int = 80,
    ):
        """
        Initialize ticker extractor.

        Args:
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.tickers = SEMICONDUCTOR_TICKERS
        self.company_map = COMPANY_TO_TICKER
        self.keywords = SEMICONDUCTOR_KEYWORDS

    def extract(self, text: str) -> list[str]:
        """
        Extract ticker symbols from text.

        Args:
            text: Text to search

        Returns:
            List of unique ticker symbols
        """
        tickers_found: set[str] = set()

        # 1. Extract cashtags ($NVDA)
        cashtag_pattern = r'\$([A-Z]{1,5})\b'
        for match in re.finditer(cashtag_pattern, text.upper()):
            ticker = match.group(1)
            if ticker in self.tickers:
                tickers_found.add(ticker)

        # 2. Direct ticker mention (all caps, standalone)
        for ticker in self.tickers:
            if re.search(rf'\b{ticker}\b', text.upper()):
                tickers_found.add(ticker)

        # 3. Company name lookup
        text_lower = text.lower()
        for company, ticker in self.company_map.items():
            if company in text_lower:
                if ticker in self.tickers:
                    tickers_found.add(ticker)

        # 4. Fuzzy matching for variations
        # Only if we haven't found many tickers yet
        if len(tickers_found) < 3:
            for company, ticker in self.company_map.items():
                if ticker in tickers_found:
                    continue

                # Check fuzzy match
                for word in text_lower.split():
                    if len(word) >= 4:  # Skip short words
                        score = fuzz.ratio(company, word)
                        if score >= self.fuzzy_threshold:
                            if ticker in self.tickers:
                                tickers_found.add(ticker)
                                break

        return sorted(tickers_found)

    def is_semiconductor_relevant(self, text: str) -> bool:
        """
        Check if text is relevant to semiconductor sector.

        Args:
            text: Text to check

        Returns:
            True if text mentions semiconductor-related content
        """
        text_lower = text.lower()

        # Check for keywords (using word boundaries to avoid false positives like "nic" in "nice")
        for keyword in self.keywords:
            # Use regex word boundary for accurate matching
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                return True

        # Check for tickers
        tickers = self.extract(text)
        return len(tickers) > 0


class Preprocessor:
    """
    Main preprocessing pipeline orchestrator.

    Applies all preprocessing steps to documents:
    1. Spam detection
    2. Bot detection
    3. Ticker extraction
    4. NER extraction (optional)
    5. Content enrichment

    Usage:
        preprocessor = Preprocessor()
        doc = preprocessor.process(doc)

        if doc.should_filter:
            continue  # Skip this document

        # With NER enabled:
        from src.ner.service import NERService
        preprocessor = Preprocessor(ner_service=NERService(), enable_ner=True)
    """

    def __init__(
        self,
        spam_detector: SpamDetector | None = None,
        bot_detector: BotDetector | None = None,
        ticker_extractor: TickerExtractor | None = None,
        ner_service: "NERService | None" = None,
        enable_ner: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            spam_detector: Custom spam detector (or use default)
            bot_detector: Custom bot detector (or use default)
            ticker_extractor: Custom ticker extractor (or use default)
            ner_service: NER service for entity extraction (optional)
            enable_ner: Whether to run NER extraction (default False)
        """
        self.spam_detector = spam_detector or SpamDetector()
        self.bot_detector = bot_detector or BotDetector()
        self.ticker_extractor = ticker_extractor or TickerExtractor()
        self._ner_service = ner_service
        self._enable_ner = enable_ner

    def process(self, doc: NormalizedDocument) -> NormalizedDocument:
        """
        Process a document through the preprocessing pipeline.

        Args:
            doc: Document to preprocess

        Returns:
            Preprocessed document with updated fields
        """
        # 1. Extract tickers (if not already done by adapter)
        if not doc.tickers_mentioned:
            tickers = self.ticker_extractor.extract(doc.content)
            doc.tickers_mentioned = tickers

        # 2. Calculate spam score
        spam_score, _ = self.spam_detector.detect(doc)
        doc.spam_score = spam_score

        # 3. Calculate bot probability
        bot_prob = self.bot_detector.detect(doc)
        doc.bot_probability = bot_prob

        # 4. NER extraction (if enabled)
        if self._enable_ner and self._ner_service is not None:
            try:
                entities = self._ner_service.extract_sync(doc.content)
                doc.entities_mentioned = [e.to_dict() for e in entities]
            except Exception as e:
                logger.warning(f"NER extraction failed for {doc.id}: {e}")
                doc.entities_mentioned = []

        logger.debug(
            f"Preprocessed {doc.id}: "
            f"spam={doc.spam_score:.2f}, "
            f"bot={doc.bot_probability:.2f}, "
            f"tickers={doc.tickers_mentioned}, "
            f"entities={len(doc.entities_mentioned)}"
        )

        return doc

    def process_batch(
        self,
        docs: list[NormalizedDocument],
    ) -> list[NormalizedDocument]:
        """
        Process a batch of documents.

        Args:
            docs: Documents to preprocess

        Returns:
            List of preprocessed documents (filtered documents removed)
        """
        processed = []
        filtered = 0

        for doc in docs:
            doc = self.process(doc)
            if doc.should_filter:
                filtered += 1
                continue
            processed.append(doc)

        logger.info(
            f"Preprocessed batch: {len(docs)} input, "
            f"{len(processed)} passed, {filtered} filtered"
        )

        return processed
