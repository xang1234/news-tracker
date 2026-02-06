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
    from src.event_extraction.patterns import PatternExtractor
    from src.keywords.service import KeywordsService
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
        self.threshold = settings.spam_threshold if threshold is None else threshold

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
        tickers: set[str] | None = None,
        company_map: dict[str, str] | None = None,
        keywords: set[str] | None = None,
        *,
        ambiguous_tickers: set[str] | None = None,
        require_context_for_ambiguous: bool = True,
        enable_company_lookup: bool = True,
        enable_fuzzy: bool = True,
    ):
        """
        Initialize ticker extractor.

        Args:
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            tickers: Universe of valid ticker symbols (normalized form)
            company_map: Mapping of company/alias text -> ticker
            keywords: Domain keywords for relevance checks
            ambiguous_tickers: Tickers that are common words (e.g., COIN, STOP).
                These are only accepted when they appear in a stronger context
                (cashtag/exchange/parenthetical) or near market-language.
            require_context_for_ambiguous: Whether to apply extra context checks
                for ambiguous tickers when they appear as bare symbols.
            enable_company_lookup: Whether to map company/alias text to tickers.
            enable_fuzzy: Whether to run fuzzy matching (best for small maps).
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.tickers = tickers or SEMICONDUCTOR_TICKERS
        self.company_map = company_map or COMPANY_TO_TICKER
        self.keywords = keywords or SEMICONDUCTOR_KEYWORDS
        self.ambiguous_tickers = ambiguous_tickers or set()
        self.require_context_for_ambiguous = require_context_for_ambiguous
        self.enable_company_lookup = enable_company_lookup
        self.enable_fuzzy = enable_fuzzy

        # Pre-compiled patterns for efficient, candidate-first extraction.
        # Note: direct/bare symbol extraction is intentionally case-sensitive;
        # we do NOT uppercase the whole input, to avoid false positives like
        # "stop" -> "STOP" when scaling to a large ticker universe.
        self._cashtag_re = re.compile(
            r'(?<![A-Za-z0-9_])\$([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b'
        )
        self._exchange_re = re.compile(
            r'\b(?:NASDAQ|NYSE|AMEX|OTC|TSX|LSE|ASX|HKEX|TSE|SSE|SZSE)\s*[:\-]\s*'
            r'([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\b',
            re.IGNORECASE,
        )
        self._paren_symbol_re = re.compile(
            r'\(([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)\)'
        )
        self._bare_symbol_re = re.compile(
            r'\b[A-Z]{2,5}(?:[.-][A-Z]{1,2})?\b'
        )
        self._market_context_re = re.compile(
            r'\b('
            r'stock|shares?|ticker|symbol|equity|earnings|guidance|revenue|'
            r'price|target|pt|upgrade|downgrade|analyst|'
            r'options?|calls?|puts?|volume|market\s+cap|'
            r'buy|sell|hold|long|short|'
            r'nasdaq|nyse|amex|otc'
            r')\b',
            re.IGNORECASE,
        )

        # Compile company lookup regexes (kept for curated/small maps).
        # This avoids substring matches like "lam" in "lamb".
        self._company_patterns: list[tuple[re.Pattern[str], str]] = []
        if self.enable_company_lookup and self.company_map:
            # Longest first reduces accidental matches on shorter aliases.
            for company, ticker in sorted(
                self.company_map.items(), key=lambda kv: len(kv[0]), reverse=True
            ):
                company_norm = company.strip().lower()
                if not company_norm:
                    continue
                escaped = re.escape(company_norm).replace(r"\ ", r"\s+")
                pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
                self._company_patterns.append((pattern, ticker))

    @staticmethod
    def _normalize_symbol(raw: str) -> str:
        raw = raw.strip()
        if not raw:
            return ""
        # Normalize segments while preserving separators like "." and "-"
        parts = re.split(r'([.-])', raw)
        return "".join(p.upper() if p not in {".", "-"} else p for p in parts)

    def _has_market_context(self, text: str, start: int, end: int) -> bool:
        """
        Heuristic: treat nearby market language or price/percent patterns as
        evidence that a bare token is likely a ticker mention.
        """
        window = 48
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        snippet = text[lo:hi]

        if self._market_context_re.search(snippet) is not None:
            return True
        # Simple numeric/price context (+5%, -2.3%, $123, 123%)
        if re.search(r'[\+\-]?\d+(\.\d+)?\s*%|\$\s*\d', snippet):
            return True
        return False

    def extract(self, text: str) -> list[str]:
        """
        Extract ticker symbols from text.

        Args:
            text: Text to search

        Returns:
            List of unique ticker symbols
        """
        tickers_found: set[str] = set()

        if not text:
            return []

        # 1) Strong signal: cashtags ($NVDA, $coin)
        for match in self._cashtag_re.finditer(text):
            symbol = self._normalize_symbol(match.group(1))
            if symbol and symbol in self.tickers:
                tickers_found.add(symbol)

        # 2) Strong signal: exchange-tagged symbols (NASDAQ: NVDA)
        for match in self._exchange_re.finditer(text):
            symbol = self._normalize_symbol(match.group(1))
            if symbol and symbol in self.tickers:
                tickers_found.add(symbol)

        # 3) Medium signal: parenthetical symbols (Company (NVDA))
        for match in self._paren_symbol_re.finditer(text):
            raw = match.group(1)
            # Avoid normal words in parentheses unless explicitly symbol-like.
            if raw != raw.upper():
                continue
            symbol = self._normalize_symbol(raw)
            if symbol and symbol in self.tickers:
                tickers_found.add(symbol)

        # 4) Bare symbol candidates (case-sensitive, single pass).
        # This scales to large ticker universes: O(len(text) + matches).
        for match in self._bare_symbol_re.finditer(text):
            symbol = match.group(0)
            if symbol not in self.tickers:
                continue
            if (
                self.require_context_for_ambiguous
                and symbol in self.ambiguous_tickers
                and symbol not in tickers_found
                and not self._has_market_context(text, match.start(), match.end())
            ):
                continue
            tickers_found.add(symbol)

        # 5) Company name / alias lookup (curated lists)
        text_lower = text.lower()
        if self.enable_company_lookup:
            for pattern, ticker in self._company_patterns:
                if ticker in tickers_found:
                    continue
                if pattern.search(text_lower) is not None:
                    if ticker in self.tickers:
                        tickers_found.add(ticker)

        # 6) Fuzzy matching for variations (best for small curated maps)
        if self.enable_fuzzy and len(tickers_found) < 3 and len(self.company_map) <= 250:
            for company, ticker in self.company_map.items():
                if ticker in tickers_found or ticker not in self.tickers:
                    continue
                company_norm = company.strip().lower()
                if not company_norm or " " in company_norm:
                    continue

                for word in re.findall(r"[a-zA-Z]{4,}", text_lower):
                    score = fuzz.ratio(company_norm, word)
                    if score >= self.fuzzy_threshold:
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
        keywords_service: "KeywordsService | None" = None,
        enable_keywords: bool = False,
        event_extractor: "PatternExtractor | None" = None,
        enable_events: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            spam_detector: Custom spam detector (or use default)
            bot_detector: Custom bot detector (or use default)
            ticker_extractor: Custom ticker extractor (or use default)
            ner_service: NER service for entity extraction (optional)
            enable_ner: Whether to run NER extraction (default False)
            keywords_service: Keywords service for keyword extraction (optional)
            enable_keywords: Whether to run keyword extraction (default False)
            event_extractor: PatternExtractor for event extraction (optional)
            enable_events: Whether to run event extraction (default False)
        """
        self.spam_detector = spam_detector or SpamDetector()
        self.bot_detector = bot_detector or BotDetector()
        self.ticker_extractor = ticker_extractor or TickerExtractor()
        self._ner_service = ner_service
        self._enable_ner = enable_ner
        self._keywords_service = keywords_service
        self._enable_keywords = enable_keywords
        self._event_extractor = event_extractor
        self._enable_events = enable_events

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

        # 5. Keywords extraction (if enabled)
        if self._enable_keywords and self._keywords_service is not None:
            try:
                keywords = self._keywords_service.extract_sync(doc.content)
                doc.keywords_extracted = [kw.to_dict() for kw in keywords]
            except Exception as e:
                logger.warning(f"Keywords extraction failed for {doc.id}: {e}")
                doc.keywords_extracted = []

        # 6. Event extraction (if enabled)
        if self._enable_events and self._event_extractor is not None:
            try:
                events = self._event_extractor.extract(doc)
                doc.events_extracted = [ev.to_dict() for ev in events]
            except Exception as e:
                logger.warning(f"Event extraction failed for {doc.id}: {e}")
                doc.events_extracted = []

        logger.debug(
            f"Preprocessed {doc.id}: "
            f"spam={doc.spam_score:.2f}, "
            f"bot={doc.bot_probability:.2f}, "
            f"tickers={doc.tickers_mentioned}, "
            f"entities={len(doc.entities_mentioned)}, "
            f"keywords={len(doc.keywords_extracted)}, "
            f"events={len(doc.events_extracted)}"
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
