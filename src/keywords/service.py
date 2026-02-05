"""
Keyword Extraction Service using TextRank algorithm.

Provides keyword extraction for financial news content using the rapid-textrank
library with spaCy integration. Follows the established NER service pattern
with lazy initialization and graceful error handling.

Architecture:
- Lazy model loading (spaCy + rapid_textrank pipeline loaded on first extract())
- spaCy pipeline component integration for efficient tokenization
- Graceful degradation (returns empty list on errors)
- Configurable via environment variables (KEYWORDS_*)
"""

import logging
from typing import Any

from src.keywords.config import KeywordsConfig
from src.keywords.schemas import ExtractedKeyword

logger = logging.getLogger(__name__)


class KeywordsService:
    """
    Keyword extraction service using TextRank algorithm.

    Extracts important keywords and phrases from text using the
    rapid-textrank library as a spaCy pipeline component, which implements
    the TextRank algorithm for unsupervised keyword extraction.

    Usage:
        >>> service = KeywordsService()
        >>> keywords = service.extract_sync("Nvidia announced HBM3E support for H200 GPUs")
        >>> for kw in keywords:
        ...     print(f"{kw.rank}. {kw.text} (score: {kw.score:.3f})")
        1. HBM3E support (score: 0.125)
        2. H200 GPUs (score: 0.108)
        3. Nvidia (score: 0.095)

    Note:
        The spaCy model and TextRank pipeline are loaded lazily on first
        extract() call to avoid memory overhead when keywords extraction
        is disabled.
    """

    def __init__(
        self,
        config: KeywordsConfig | None = None,
    ):
        """
        Initialize keywords service.

        Args:
            config: Keywords configuration. If None, uses default config.
        """
        self.config = config or KeywordsConfig()
        self._nlp: Any = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the spaCy pipeline is loaded."""
        return self._initialized

    def _initialize(self) -> None:
        """
        Load the spaCy model and add rapid_textrank pipeline component.

        This is called lazily on first extract() call.
        """
        if self._initialized:
            return

        try:
            import spacy
            import rapid_textrank.spacy_component  # noqa: F401 - registers the pipeline factory
        except ImportError as e:
            logger.error(
                "spacy and rapid-textrank[spacy] are required for keyword extraction. "
                "Install with: pip install rapid-textrank[spacy]"
            )
            raise ImportError(
                "spacy and rapid-textrank[spacy] are required for keyword extraction. "
                "Install with: pip install rapid-textrank[spacy]"
            ) from e

        # Load spaCy model
        try:
            logger.info(f"Loading spaCy model: {self.config.spacy_model}")
            self._nlp = spacy.load(self.config.spacy_model)
        except OSError as e:
            # Only attempt fallback if configured model isn't already en_core_web_sm
            if self.config.spacy_model != "en_core_web_sm":
                logger.warning(
                    f"Model {self.config.spacy_model} not found, falling back to en_core_web_sm"
                )
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError as fallback_error:
                    logger.error("No spaCy model available. Download with: python -m spacy download en_core_web_sm")
                    raise ImportError(
                        "No spaCy model available. Download with: python -m spacy download en_core_web_sm"
                    ) from fallback_error
            else:
                logger.error("No spaCy model available. Download with: python -m spacy download en_core_web_sm")
                raise ImportError(
                    "No spaCy model available. Download with: python -m spacy download en_core_web_sm"
                ) from e

        # Add rapid_textrank to the pipeline
        self._nlp.add_pipe("rapid_textrank")
        self._initialized = True
        logger.info("Keywords service initialized with spaCy pipeline")

    def extract_sync(self, text: str) -> list[ExtractedKeyword]:
        """
        Extract keywords from text (synchronous).

        Args:
            text: Text to extract keywords from.

        Returns:
            List of extracted ExtractedKeyword objects, sorted by score descending.
        """
        if not text or (isinstance(text, str) and not text.strip()):
            return []

        # Lazy initialization
        if not self._initialized:
            try:
                self._initialize()
            except ImportError:
                logger.warning("Keywords extraction unavailable - library not installed")
                return []

        # Truncate if too long
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]
            logger.debug(f"Text truncated to {self.config.max_text_length} chars")

        try:
            # Process text through spaCy pipeline with rapid_textrank
            doc = self._nlp(text)

            # Extract phrases from doc extension attribute
            phrases = doc._.phrases[: self.config.top_n]

            # Convert Phrase objects to ExtractedKeyword
            keywords: list[ExtractedKeyword] = []
            for rank, phrase in enumerate(phrases, start=1):
                # Filter by minimum score
                if phrase.score < self.config.min_score:
                    continue

                # Get lemma - may be available as attribute or text fallback
                lemma = getattr(phrase, "lemma", None) or phrase.text.lower()
                # Get count - may be available as attribute or default to 1
                count = getattr(phrase, "count", 1)

                keywords.append(
                    ExtractedKeyword(
                        text=phrase.text,
                        score=phrase.score,
                        rank=rank,
                        lemma=lemma,
                        count=count,
                        metadata={"algorithm": "rapid_textrank"},
                    )
                )

            logger.debug(f"Extracted {len(keywords)} keywords from text")
            return keywords

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    async def extract(self, text: str) -> list[ExtractedKeyword]:
        """
        Extract keywords from text (async wrapper).

        Note: TextRank processing is CPU-bound, so this is a simple
        wrapper around extract_sync. For true parallelism, use
        extract_batch with multiple texts.

        Args:
            text: Text to extract keywords from.

        Returns:
            List of extracted ExtractedKeyword objects.
        """
        return self.extract_sync(text)

    async def extract_batch(
        self, texts: list[str]
    ) -> list[list[ExtractedKeyword]]:
        """
        Extract keywords from multiple texts.

        Args:
            texts: List of texts to process.

        Returns:
            List of keyword lists, one per input text.
        """
        if not texts:
            return []

        # Lazy initialization
        if not self._initialized:
            try:
                self._initialize()
            except ImportError:
                return [[] for _ in texts]

        results: list[list[ExtractedKeyword]] = []
        for text in texts:
            keywords = self.extract_sync(text)
            results.append(keywords)

        return results
