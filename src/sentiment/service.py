"""
Sentiment analysis service using FinBERT.

Provides async sentiment classification for financial documents with:
- Document-level sentiment (positive/neutral/negative)
- Entity-level sentiment from context windows
- Lazy model loading for efficient resource usage
- Automatic device detection (GPU/CPU/MPS)
- Redis caching to avoid recomputation
- FP16 support for GPU acceleration
- Batch processing for efficiency
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.observability.metrics import get_metrics
from src.sentiment.config import SentimentConfig
from src.sentiment.emoji_lookup import compute_emoji_modifier

logger = structlog.get_logger(__name__)

# ProsusAI/finbert label mapping (verified from model config)
# The model outputs: 0=positive, 1=negative, 2=neutral
# We map to standard names
LABEL_MAPPING = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentService:
    """
    Service for financial sentiment analysis using FinBERT.

    Uses lazy initialization to defer model loading until first use,
    which is important for services that may not need sentiment analysis.

    Features:
    - Document-level classification (positive/neutral/negative)
    - Entity-level sentiment extraction with context windows
    - Automatic device selection (CUDA > MPS > CPU)
    - FP16 inference for GPU acceleration
    - Redis caching using content hash keys
    - Batch processing for efficiency

    Usage:
        service = SentimentService()

        # Document-level sentiment
        result = await service.analyze("NVIDIA stock surged 10% today")
        print(result["label"], result["confidence"])

        # With entity-level sentiment
        entities = [{"text": "NVIDIA", "start": 0, "end": 6, "type": "COMPANY"}]
        result = await service.analyze_with_entities(text, entities)
        for e_sent in result.get("entity_sentiments", []):
            print(f"{e_sent['entity']}: {e_sent['label']}")
    """

    def __init__(
        self,
        config: SentimentConfig | None = None,
        redis_client: redis.Redis | None = None,
    ):
        """
        Initialize the sentiment service.

        Model loading is deferred until first analyze() call.

        Args:
            config: Sentiment configuration (uses defaults if None)
            redis_client: Redis client for caching (optional)
        """
        self._config = config or SentimentConfig()
        self._redis = redis_client

        # Model components (lazy loaded)
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: torch.device | None = None
        self._initialized = False

        logger.info(
            "SentimentService created",
            extra={
                "model": self._config.model_name,
                "cache_enabled": self._config.cache_enabled,
                "entity_sentiment_enabled": self._config.enable_entity_sentiment,
            },
        )

    def _detect_device(self) -> torch.device:
        """Detect the best available device for inference."""
        if self._device is not None:
            return self._device

        if self._config.device != "auto":
            self._device = torch.device(self._config.device)
            return self._device

        if torch.cuda.is_available():
            logger.info("Using CUDA device for sentiment analysis")
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device for sentiment analysis")
            self._device = torch.device("mps")
        else:
            logger.info("Using CPU for sentiment analysis")
            self._device = torch.device("cpu")

        return self._device

    def _initialize(self) -> None:
        """
        Load FinBERT model and tokenizer on first use.

        This is called lazily on first analyze() call.
        """
        if self._initialized:
            return

        device = self._detect_device()

        logger.info(f"Loading sentiment model: {self._config.model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            model_max_length=self._config.max_sequence_length,
        )

        # Load model for sequence classification
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._config.model_name,
        )
        self._model.to(device)
        self._model.eval()

        # Enable FP16 for CUDA devices
        if self._config.use_fp16 and device.type == "cuda":
            self._model = self._model.half()
            logger.info("FP16 inference enabled for sentiment analysis")

        self._initialized = True

        logger.info(
            "Sentiment model loaded",
            extra={
                "device": str(device),
                "fp16": self._config.use_fp16 and device.type == "cuda",
            },
        )

    @property
    def model(self) -> AutoModelForSequenceClassification:
        """Get model, initializing if needed."""
        self._initialize()
        return self._model  # type: ignore

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get tokenizer, initializing if needed."""
        self._initialize()
        return self._tokenizer  # type: ignore

    @property
    def device(self) -> torch.device:
        """Get device, detecting if needed."""
        return self._detect_device()

    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized

    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _make_cache_key(self, text: str) -> str:
        """Create cache key with content hash."""
        content_hash = self._compute_content_hash(text)
        return f"{self._config.cache_key_prefix}{content_hash}"

    async def _get_cached_result(self, text: str) -> dict[str, Any] | None:
        """Try to retrieve sentiment result from cache."""
        if not self._config.cache_enabled or not self._redis:
            return None

        cache_key = self._make_cache_key(text)
        metrics = get_metrics()

        try:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for sentiment: {cache_key}")
                metrics.record_sentiment_cache(hit=True)
                return json.loads(cached)
            else:
                metrics.record_sentiment_cache(hit=False)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            metrics.record_sentiment_cache(hit=False)

        return None

    async def _cache_result(self, text: str, result: dict[str, Any]) -> None:
        """Store sentiment result in cache."""
        if not self._config.cache_enabled or not self._redis:
            return

        cache_key = self._make_cache_key(text)

        try:
            await self._redis.setex(
                cache_key,
                self._config.cache_ttl_seconds,
                json.dumps(result),
            )
            logger.debug(f"Cached sentiment: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _classify_text(self, text: str) -> tuple[str, float, dict[str, float]]:
        """
        Classify sentiment for a single text (internal, synchronous).

        Returns:
            (label, confidence, scores) tuple
        """
        if not text.strip():
            # Return neutral for empty text
            return (
                "neutral",
                1.0,
                {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_sequence_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Classify
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Convert to numpy
        probs_np = probs.cpu().numpy()[0]

        # Get predicted class and confidence
        predicted_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_idx])

        # Map to labels using ProsusAI/finbert mapping
        label = LABEL_MAPPING[predicted_idx]
        scores = {LABEL_MAPPING[i]: float(probs_np[i]) for i in range(len(LABEL_MAPPING))}

        return label, confidence, scores

    def _clean_for_model(self, text: str) -> str:
        """Clean text for tokenization, handling emojis."""
        # Remove emojis that can confuse the tokenizer
        text = re.sub(r"[^\w\s$%.,!?'\"-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_emoji_modifier(self, text: str) -> tuple[float, dict[str, float]]:
        """
        Compute emoji sentiment modifier from original text.

        Must be called BEFORE _clean_for_model() as emojis are stripped during cleaning.

        Args:
            text: Original text with emojis

        Returns:
            Tuple of (modifier, emoji_breakdown):
            - modifier: Clamped value in [-max, +max] where max is config.emoji_modifier_max
            - emoji_breakdown: Dict mapping found emojis to their weights
        """
        if not self._config.emoji_modifier_enabled:
            return 0.0, {}

        return compute_emoji_modifier(text, self._config.emoji_modifier_max)

    def _apply_emoji_modifier(
        self,
        scores: dict[str, float],
        modifier: float,
    ) -> tuple[dict[str, float], str]:
        """
        Apply emoji modifier to sentiment probability distribution.

        Positive modifiers shift probability toward positive sentiment.
        Negative modifiers shift probability toward negative sentiment.

        Args:
            scores: Original sentiment scores {"positive": p, "negative": n, "neutral": u}
            modifier: Emoji modifier in [-max, +max] range

        Returns:
            Tuple of (adjusted_scores, new_label):
            - adjusted_scores: Modified probability distribution (sums to 1)
            - new_label: New predicted label based on adjusted scores
        """
        if modifier == 0.0:
            # No modification needed
            label = max(scores, key=scores.get)  # type: ignore
            return scores, label

        # Extract current scores
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)

        # Compute adjustment amounts
        # Positive modifier: increase positive, decrease negative
        # Negative modifier: increase negative, decrease positive
        if modifier > 0:
            # Bullish: boost positive at expense of negative and neutral
            pos_delta = modifier * (1 - pos)  # Room to increase
            neg_delta = -modifier * neg * 0.7  # Reduce negative more
            neu_delta = -modifier * neu * 0.3  # Reduce neutral less
        else:
            # Bearish: boost negative at expense of positive and neutral
            abs_mod = abs(modifier)
            neg_delta = abs_mod * (1 - neg)  # Room to increase
            pos_delta = -abs_mod * pos * 0.7  # Reduce positive more
            neu_delta = -abs_mod * neu * 0.3  # Reduce neutral less

        # Apply deltas
        new_pos = max(0.0, min(1.0, pos + pos_delta))
        new_neg = max(0.0, min(1.0, neg + neg_delta))
        new_neu = max(0.0, min(1.0, neu + neu_delta))

        # Re-normalize to sum to 1
        total = new_pos + new_neg + new_neu
        if total > 0:
            new_pos /= total
            new_neg /= total
            new_neu /= total

        adjusted = {
            "positive": round(new_pos, 4),
            "negative": round(new_neg, 4),
            "neutral": round(new_neu, 4),
        }

        # Determine new label
        label = max(adjusted, key=adjusted.get)  # type: ignore

        return adjusted, label

    async def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze document-level sentiment.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment results:
            {
                "label": "positive" | "negative" | "neutral",
                "confidence": float (0-1),
                "scores": {"positive": float, "negative": float, "neutral": float},
                "model": "ProsusAI/finbert",
                "analyzed_at": "2026-02-03T12:00:00Z",
                "emoji_modifier": float (optional, only if emojis found),
                "emojis_found": {"🚀": 0.3, ...} (optional, only if emojis found)
            }
        """
        start_time = time.perf_counter()
        metrics = get_metrics()

        # Check cache
        cached = await self._get_cached_result(text)
        if cached is not None:
            # Update processing time to reflect cache hit latency
            latency = time.perf_counter() - start_time
            cached["processing_time_ms"] = round(latency * 1000, 2)
            cached["cached"] = True
            metrics.record_sentiment_latency("single", latency)
            return cached

        # Compute emoji modifier BEFORE cleaning (emojis are stripped during cleaning)
        emoji_modifier, emoji_breakdown = self._compute_emoji_modifier(text)

        # Clean text for model
        clean_text = self._clean_for_model(text)

        # Classify
        label, confidence, scores = self._classify_text(clean_text)

        # Apply emoji modifier if present
        if emoji_modifier != 0.0:
            scores, label = self._apply_emoji_modifier(scores, emoji_modifier)
            # Update confidence to reflect the adjusted label's score
            confidence = scores[label]

        # Record latency
        latency = time.perf_counter() - start_time
        metrics.record_sentiment_latency("single", latency)

        # Create result
        result: dict[str, Any] = {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "model": self._config.model_name,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(latency * 1000, 2),
        }

        # Include emoji info only when emojis were found
        if emoji_breakdown:
            result["emoji_modifier"] = round(emoji_modifier, 4)
            result["emojis_found"] = emoji_breakdown

        # Cache
        await self._cache_result(text, result)

        return result

    async def analyze_with_entities(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze sentiment with entity-level extraction.

        For each entity, extracts context window and classifies sentiment
        specific to that entity mention.

        Args:
            text: Full document text
            entities: Extracted entities from NER, each with:
                - text: Entity text
                - start: Character start position
                - end: Character end position
                - type: Entity type (COMPANY, PRODUCT, etc.)
                - normalized: Normalized form (optional)

        Returns:
            Dictionary with document and entity-level sentiment:
            {
                "label": "positive",
                "confidence": 0.87,
                "scores": {...},
                "entity_sentiments": [
                    {
                        "entity": "NVIDIA",
                        "type": "COMPANY",
                        "label": "positive",
                        "confidence": 0.92,
                        "scores": {...},
                        "context": "NVIDIA surged 10% after..."
                    }
                ],
                ...
            }
        """
        start_time = time.perf_counter()
        metrics = get_metrics()

        # Skip entity-level if disabled or no entities
        if not self._config.enable_entity_sentiment or not entities:
            return await self.analyze(text)

        # For entity-level analysis, we don't use the document-level cache
        # because the result structure is different (includes entity_sentiments)
        # Instead, classify document-level directly
        clean_text = self._clean_for_model(text)
        doc_label, doc_confidence, doc_scores = self._classify_text(clean_text)

        # Extract entity-level sentiment
        entity_sentiments: list[dict[str, Any]] = []

        for entity in entities:
            # Skip entities without position info
            if "start" not in entity or "end" not in entity:
                continue

            start = entity["start"]
            end = entity["end"]

            # Validate entity position
            if start >= end or start < 0 or end > len(text):
                logger.warning(
                    f"Invalid entity position: start={start}, end={end}, text_len={len(text)}"
                )
                continue

            # Extract context window
            context_start = max(0, start - self._config.entity_context_window)
            context_end = min(len(text), end + self._config.entity_context_window)
            context = text[context_start:context_end].strip()

            # Skip empty contexts
            if not context:
                continue

            # Clean and classify entity context
            clean_context = self._clean_for_model(context)
            ent_label, ent_confidence, ent_scores = self._classify_text(clean_context)

            entity_sentiments.append({
                "entity": entity.get("normalized") or entity.get("text", ""),
                "type": entity.get("type", "UNKNOWN"),
                "label": ent_label,
                "confidence": round(ent_confidence, 4),
                "scores": {k: round(v, 4) for k, v in ent_scores.items()},
                "context": context[:200],  # Truncate for storage
            })

        # Record latency and entity count
        latency = time.perf_counter() - start_time
        metrics.record_sentiment_latency("entity", latency)
        metrics.record_sentiment_entities(len(entity_sentiments))

        # Create result with entity sentiments
        result = {
            "label": doc_label,
            "confidence": round(doc_confidence, 4),
            "scores": {k: round(v, 4) for k, v in doc_scores.items()},
            "model": self._config.model_name,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(latency * 1000, 2),
            "entity_sentiments": entity_sentiments,
        }

        return result

    async def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Analyze sentiment for multiple texts (document-level only).

        Processes texts individually but checks cache for each.
        For truly batched GPU inference, consider extending this method.

        Args:
            texts: List of texts to analyze
            show_progress: Whether to log progress

        Returns:
            List of sentiment result dictionaries
        """
        if not texts:
            return []

        results: list[dict[str, Any]] = []
        cached_count = 0

        for i, text in enumerate(texts):
            if not text.strip():
                # Empty text -> neutral
                results.append({
                    "label": "neutral",
                    "confidence": 1.0,
                    "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                    "model": self._config.model_name,
                })
                continue

            # Check cache
            cached = await self._get_cached_result(text)
            if cached is not None:
                results.append(cached)
                cached_count += 1
            else:
                # Analyze and cache
                result = await self.analyze(text)
                results.append(result)

        if show_progress:
            logger.info(
                f"Sentiment batch: {len(texts)} texts, "
                f"{cached_count} cached, "
                f"{len(texts) - cached_count} analyzed"
            )

        return results

    async def close(self) -> None:
        """Clean up resources."""
        # Model cleaned up by Python garbage collection
        # Redis client managed externally
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False
        logger.info("SentimentService closed")

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "initialized": self._initialized,
            "model": self._config.model_name,
            "device": str(self._device) if self._device else "not initialized",
            "cache_enabled": self._config.cache_enabled,
            "entity_sentiment_enabled": self._config.enable_entity_sentiment,
        }

    async def is_cache_available(self) -> bool:
        """Check if Redis cache is available and responding."""
        if not self._config.cache_enabled or not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False
