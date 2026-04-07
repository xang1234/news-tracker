"""Sentiment analysis service using FinBERT."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference.runtime import (
    ONNX_AVAILABLE,
    TORCH_AVAILABLE,
    RuntimeSelection,
    ort,
    resolve_runtime,
    torch,
)
from src.observability.metrics import get_metrics
from src.sentiment.config import SentimentConfig
from src.sentiment.emoji_lookup import compute_emoji_modifier

logger = structlog.get_logger(__name__)

# ProsusAI/finbert label mapping (verified from model config)
LABEL_MAPPING = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentService:
    """Service for financial sentiment analysis using FinBERT."""

    def __init__(
        self,
        config: SentimentConfig | None = None,
        redis_client: redis.Redis | None = None,
    ):
        self._config = config or SentimentConfig()
        self._redis = redis_client

        self._model: Any | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: Any | None = None
        self._runtime: RuntimeSelection | None = None
        self._initialized = False

        logger.info(
            "SentimentService created",
            extra={
                "model": self._config.model_name,
                "backend": self._config.backend,
                "cache_enabled": self._config.cache_enabled,
                "entity_sentiment_enabled": self._config.enable_entity_sentiment,
            },
        )

    def _detect_device(self) -> Any:
        """Detect the active runtime/device for inference."""
        if self._device is not None:
            return self._device

        runtime = self._resolve_runtime()
        if runtime.backend == "torch":
            assert TORCH_AVAILABLE
            self._device = torch.device(runtime.torch_device)
        else:
            self._device = runtime.accelerator
        return self._device

    def _resolve_runtime(self) -> RuntimeSelection:
        """Resolve the concrete inference runtime once per service."""
        if self._runtime is None:
            runtime = resolve_runtime(
                backend=self._config.backend,
                device=self._config.device,
                execution_provider=self._config.execution_provider,
            )
            if (
                self._config.backend == "auto"
                and runtime.backend == "onnx"
                and TORCH_AVAILABLE
                and not self._config.onnx_model_path
            ):
                runtime = resolve_runtime(
                    backend="torch",
                    device=self._config.device,
                    execution_provider=self._config.execution_provider,
                )
            self._runtime = runtime
        return self._runtime

    def _initialize(self) -> None:
        """Load model/tokenizer on first use."""
        if self._initialized:
            return

        # Validate ONNX model path — fall back to HuggingFace if missing
        onnx_path = self._config.onnx_model_path
        if onnx_path and not Path(onnx_path).exists():
            logger.warning(
                "ONNX model path configured but not found, falling back to HuggingFace model",
                onnx_path=onnx_path,
                model_name=self._config.model_name,
            )
            onnx_path = None

        runtime = self._resolve_runtime()

        # Force torch backend when ONNX was selected but exported models are missing
        if runtime.backend == "onnx" and not onnx_path:
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    f"ONNX model path '{self._config.onnx_model_path}' not found and "
                    "torch is not available as a fallback. Please provide a valid "
                    "onnx_model_path or install torch."
                )
            runtime = resolve_runtime(
                backend="torch",
                device=self._config.device,
                execution_provider=self._config.execution_provider,
            )
            self._runtime = runtime

        model_source = onnx_path or self._config.model_name

        logger.info(
            "Loading sentiment model",
            model=self._config.model_name,
            backend=runtime.backend,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            model_max_length=self._config.max_sequence_length,
        )

        if runtime.backend == "torch":
            assert TORCH_AVAILABLE
            device = self._detect_device()
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._config.model_name,
            )
            self._model.to(device)
            self._model.eval()

            if self._config.use_fp16 and runtime.accelerator == "cuda":
                self._model = self._model.half()
                logger.info("FP16 inference enabled for sentiment analysis")
        else:
            self._model = self._load_onnx_session(runtime, onnx_path)

        self._initialized = True

        logger.info(
            "Sentiment model loaded",
            backend=runtime.backend,
            accelerator=runtime.accelerator,
            execution_provider=runtime.onnx_provider,
        )

    def _load_onnx_session(self, runtime: RuntimeSelection, onnx_path: str | None = None) -> Any:
        """Load an ONNX Runtime session for a pre-exported sentiment model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX backend selected but onnxruntime is not installed.")
        effective_path = onnx_path or self._config.onnx_model_path
        if not effective_path:
            raise RuntimeError(
                "ONNX backend selected for sentiment, but no exported model path is configured."
            )

        model_dir = Path(effective_path)
        model_file = model_dir / "model.onnx"
        if not model_file.exists():
            candidates = sorted(model_dir.glob("*.onnx"))
            if len(candidates) == 1:
                model_file = candidates[0]
            else:
                raise RuntimeError(
                    f"Expected exported ONNX model under '{model_dir}', "
                    "but could not find model.onnx."
                )

        return ort.InferenceSession(
            str(model_file),
            providers=[runtime.onnx_provider or "CPUExecutionProvider"],
        )

    @property
    def model(self) -> Any:
        """Get model, initializing if needed."""
        self._initialize()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get tokenizer, initializing if needed."""
        self._initialize()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def device(self) -> Any:
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
        return f"{self._config.cache_key_prefix}{self._compute_content_hash(text)}"

    async def _get_cached_result(self, text: str) -> dict[str, Any] | None:
        """Try to retrieve sentiment result from cache."""
        if not self._config.cache_enabled or not self._redis:
            return None

        cache_key = self._make_cache_key(text)
        metrics = get_metrics()

        try:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug("Cache hit for sentiment", cache_key=cache_key)
                metrics.record_sentiment_cache(hit=True)
                return json.loads(cached)
            metrics.record_sentiment_cache(hit=False)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
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
            logger.debug("Cached sentiment", cache_key=cache_key)
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))

    def _softmax_numpy(self, logits: np.ndarray) -> np.ndarray:
        """Compute a stable softmax over the last axis."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _predict_scores(self, texts: list[str]) -> np.ndarray:
        """Predict sentiment scores for a batch of already-cleaned texts."""
        if not texts:
            return np.empty((0, len(LABEL_MAPPING)), dtype=np.float32)

        runtime = self._resolve_runtime()
        tokenizer = self.tokenizer
        model = self.model

        if runtime.backend == "torch":
            assert TORCH_AVAILABLE
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.max_sequence_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()

        inputs = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=self._config.max_sequence_length,
            padding=True,
        )
        session_inputs = {}
        expected_inputs = {meta.name for meta in model.get_inputs()}
        for name, value in inputs.items():
            if name in expected_inputs:
                session_inputs[name] = value.astype(np.int64)

        outputs = model.run(None, session_inputs)
        logits = outputs[0]
        return self._softmax_numpy(logits)

    def _scores_to_prediction(self, probs_np: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Convert probabilities into label/confidence/score structures."""
        predicted_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_idx])
        label = LABEL_MAPPING[predicted_idx]
        scores = {LABEL_MAPPING[i]: float(probs_np[i]) for i in range(len(LABEL_MAPPING))}
        return label, confidence, scores

    def _classify_text(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Classify sentiment for a single cleaned text."""
        if not text.strip():
            return (
                "neutral",
                1.0,
                {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            )

        probs_np = self._predict_scores([text])[0]
        return self._scores_to_prediction(probs_np)

    def _clean_for_model(self, text: str) -> str:
        """Clean text for tokenization, handling emojis."""
        text = re.sub(r"[^\w\s$%.,!?'\"-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_emoji_modifier(self, text: str) -> tuple[float, dict[str, float]]:
        """Compute emoji sentiment modifier from original text."""
        if not self._config.emoji_modifier_enabled:
            return 0.0, {}
        return compute_emoji_modifier(text, self._config.emoji_modifier_max)

    def _apply_emoji_modifier(
        self,
        scores: dict[str, float],
        modifier: float,
    ) -> tuple[dict[str, float], str]:
        """Apply emoji modifier to a sentiment probability distribution."""
        if modifier == 0.0:
            label = max(scores, key=scores.get)  # type: ignore[arg-type]
            return scores, label

        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)

        if modifier > 0:
            pos_delta = modifier * (1 - pos)
            neg_delta = -modifier * neg * 0.7
            neu_delta = -modifier * neu * 0.3
        else:
            abs_mod = abs(modifier)
            neg_delta = abs_mod * (1 - neg)
            pos_delta = -abs_mod * pos * 0.7
            neu_delta = -abs_mod * neu * 0.3

        new_pos = max(0.0, min(1.0, pos + pos_delta))
        new_neg = max(0.0, min(1.0, neg + neg_delta))
        new_neu = max(0.0, min(1.0, neu + neu_delta))

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
        label = max(adjusted, key=adjusted.get)  # type: ignore[arg-type]
        return adjusted, label

    def _build_result(
        self,
        label: str,
        confidence: float,
        scores: dict[str, float],
        latency_ms: float,
        emoji_modifier: float = 0.0,
        emoji_breakdown: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Create a response payload from a classification result."""
        result: dict[str, Any] = {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "model": self._config.model_name,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "processing_time_ms": round(latency_ms, 2),
        }

        if emoji_breakdown:
            result["emoji_modifier"] = round(emoji_modifier, 4)
            result["emojis_found"] = emoji_breakdown

        return result

    async def analyze(self, text: str) -> dict[str, Any]:
        """Analyze document-level sentiment."""
        start_time = time.perf_counter()
        metrics = get_metrics()

        cached = await self._get_cached_result(text)
        if cached is not None:
            latency = time.perf_counter() - start_time
            cached["processing_time_ms"] = round(latency * 1000, 2)
            cached["cached"] = True
            metrics.record_sentiment_latency("single", latency)
            return cached

        emoji_modifier, emoji_breakdown = self._compute_emoji_modifier(text)
        clean_text = self._clean_for_model(text)
        label, confidence, scores = self._classify_text(clean_text)

        if emoji_modifier != 0.0:
            scores, label = self._apply_emoji_modifier(scores, emoji_modifier)
            confidence = scores[label]

        latency = time.perf_counter() - start_time
        metrics.record_sentiment_latency("single", latency)

        result = self._build_result(
            label=label,
            confidence=confidence,
            scores=scores,
            latency_ms=latency * 1000,
            emoji_modifier=emoji_modifier,
            emoji_breakdown=emoji_breakdown,
        )

        await self._cache_result(text, result)
        return result

    async def analyze_with_entities(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze sentiment with entity-level extraction."""
        start_time = time.perf_counter()
        metrics = get_metrics()

        if not self._config.enable_entity_sentiment or not entities:
            return await self.analyze(text)

        clean_doc = self._clean_for_model(text)
        contexts: list[tuple[dict[str, Any], str]] = []
        clean_contexts: list[str] = []

        for entity in entities:
            if "start" not in entity or "end" not in entity:
                continue

            start = entity["start"]
            end = entity["end"]
            if start >= end or start < 0 or end > len(text):
                logger.warning(
                    "Invalid entity position",
                    start=start,
                    end=end,
                    text_len=len(text),
                )
                continue

            context_start = max(0, start - self._config.entity_context_window)
            context_end = min(len(text), end + self._config.entity_context_window)
            context = text[context_start:context_end].strip()
            if not context:
                continue

            contexts.append((entity, context))
            clean_contexts.append(self._clean_for_model(context))

        batched_scores = self._predict_scores([clean_doc] + clean_contexts)
        doc_label, doc_confidence, doc_scores = self._scores_to_prediction(batched_scores[0])

        entity_sentiments: list[dict[str, Any]] = []
        for idx, (entity, context) in enumerate(contexts, start=1):
            ent_label, ent_confidence, ent_scores = self._scores_to_prediction(batched_scores[idx])
            entity_sentiments.append(
                {
                    "entity": entity.get("normalized") or entity.get("text", ""),
                    "type": entity.get("type", "UNKNOWN"),
                    "label": ent_label,
                    "confidence": round(ent_confidence, 4),
                    "scores": {k: round(v, 4) for k, v in ent_scores.items()},
                    "context": context[:200],
                }
            )

        latency = time.perf_counter() - start_time
        metrics.record_sentiment_latency("entity", latency)
        metrics.record_sentiment_entities(len(entity_sentiments))

        return {
            "label": doc_label,
            "confidence": round(doc_confidence, 4),
            "scores": {k: round(v, 4) for k, v in doc_scores.items()},
            "model": self._config.model_name,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "processing_time_ms": round(latency * 1000, 2),
            "entity_sentiments": entity_sentiments,
        }

    async def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        """Analyze sentiment for multiple texts with true batch inference."""
        if not texts:
            return []

        start_time = time.perf_counter()
        metrics = get_metrics()
        results: list[dict[str, Any] | None] = [None] * len(texts)
        uncached: list[tuple[int, str, str, float, dict[str, float]]] = []
        cached_count = 0

        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = {
                    "label": "neutral",
                    "confidence": 1.0,
                    "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                    "model": self._config.model_name,
                }
                continue

            cached = await self._get_cached_result(text)
            if cached is not None:
                results[i] = cached
                cached_count += 1
                continue

            emoji_modifier, emoji_breakdown = self._compute_emoji_modifier(text)
            uncached.append((i, text, self._clean_for_model(text), emoji_modifier, emoji_breakdown))

        batch_size = self._config.batch_size
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start : batch_start + batch_size]
            score_rows = self._predict_scores([clean_text for _, _, clean_text, _, _ in batch])

            for row_idx, (
                result_idx,
                original_text,
                _,
                emoji_modifier,
                emoji_breakdown,
            ) in enumerate(batch):
                label, confidence, scores = self._scores_to_prediction(score_rows[row_idx])
                if emoji_modifier != 0.0:
                    scores, label = self._apply_emoji_modifier(scores, emoji_modifier)
                    confidence = scores[label]

                result = self._build_result(
                    label=label,
                    confidence=confidence,
                    scores=scores,
                    latency_ms=0.0,
                    emoji_modifier=emoji_modifier,
                    emoji_breakdown=emoji_breakdown,
                )
                results[result_idx] = result
                await self._cache_result(original_text, result)

        elapsed = time.perf_counter() - start_time
        metrics.record_sentiment_latency("batch", elapsed)

        # Spread total batch latency across uncached results to keep the field populated.
        uncached_count = max(1, len(uncached))
        per_item_latency_ms = round((elapsed * 1000) / uncached_count, 2) if uncached else 0.0
        for result in results:
            if (
                result is not None
                and "processing_time_ms" in result
                and result["processing_time_ms"] == 0.0
            ):
                result["processing_time_ms"] = per_item_latency_ms

        if show_progress:
            logger.info(
                "Sentiment batch",
                total=len(texts),
                cached=cached_count,
                analyzed=len(uncached),
                backend=self._resolve_runtime().backend,
            )

        return results  # type: ignore[return-value]

    async def close(self) -> None:
        """Clean up resources."""
        self._model = None
        self._tokenizer = None
        self._device = None
        self._runtime = None
        self._initialized = False
        logger.info("SentimentService closed")

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        runtime = self._runtime
        return {
            "initialized": self._initialized,
            "model": self._config.model_name,
            "device": str(self._device) if self._device is not None else "not initialized",
            "backend": runtime.backend if runtime else self._config.backend,
            "accelerator": runtime.accelerator if runtime else "not initialized",
            "execution_provider": runtime.onnx_provider if runtime else None,
            "cache_enabled": self._config.cache_enabled,
            "entity_sentiment_enabled": self._config.enable_entity_sentiment,
            "onnx_model_path": self._config.onnx_model_path,
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
