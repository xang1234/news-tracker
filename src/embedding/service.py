"""Embedding generation service using FinBERT and MiniLM."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog
from transformers import AutoModel, AutoTokenizer

from src.embedding.config import EmbeddingConfig
from src.inference.runtime import (
    ONNX_AVAILABLE,
    TORCH_AVAILABLE,
    RuntimeSelection,
    ort,
    resolve_runtime,
    torch,
)

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Available embedding models."""

    FINBERT = "finbert"
    MINILM = "minilm"


class EmbeddingService:
    """Service for generating document embeddings using FinBERT and MiniLM."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        redis_client: redis.Redis | None = None,
    ):
        self._config = config or EmbeddingConfig()
        self._redis = redis_client

        self._models: dict[ModelType, Any] = {}
        self._tokenizers: dict[ModelType, AutoTokenizer] = {}
        self._device: Any | None = None
        self._runtime: RuntimeSelection | None = None

        self._initialized: dict[ModelType, bool] = {
            ModelType.FINBERT: False,
            ModelType.MINILM: False,
        }

        logger.info(
            "EmbeddingService created",
            extra={
                "finbert_model": self._config.model_name,
                "minilm_model": self._config.minilm_model_name,
                "backend": self._config.backend,
                "cache_enabled": self._config.cache_enabled,
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
                and not self._has_configured_onnx_models()
            ):
                runtime = resolve_runtime(
                    backend="torch",
                    device=self._config.device,
                    execution_provider=self._config.execution_provider,
                )
            self._runtime = runtime
        return self._runtime

    def _has_configured_onnx_models(self) -> bool:
        """Return whether both embedding ONNX model directories are configured."""
        return bool(self._config.onnx_model_path and self._config.onnx_minilm_model_path)

    def _initialize(self, model_type: ModelType = ModelType.FINBERT) -> None:
        """Initialize model and tokenizer on first use."""
        if self._initialized.get(model_type, False):
            return

        runtime = self._resolve_runtime()
        device = self._detect_device()
        model_name, max_length, tokenizer_source = self._model_config(model_type)

        logger.info(
            "Loading embedding model",
            model=model_name,
            model_type=model_type.value,
            backend=runtime.backend,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            model_max_length=max_length,
        )

        if runtime.backend == "torch":
            assert TORCH_AVAILABLE
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()

            if self._config.use_fp16 and runtime.accelerator == "cuda":
                model = model.half()
                logger.info("FP16 inference enabled for embeddings", model_type=model_type.value)
        else:
            model = self._load_onnx_session(model_type, runtime)

        self._models[model_type] = model
        self._tokenizers[model_type] = tokenizer
        self._initialized[model_type] = True

        logger.info(
            "Embedding model loaded",
            model_type=model_type.value,
            backend=runtime.backend,
            accelerator=runtime.accelerator,
            execution_provider=runtime.onnx_provider,
        )

    def _model_config(self, model_type: ModelType) -> tuple[str, int, str]:
        """Resolve model, sequence length, and tokenizer source."""
        if model_type == ModelType.FINBERT:
            model_name = self._config.model_name
            max_length = self._config.max_sequence_length
            tokenizer_source = self._config.onnx_model_path or model_name
        else:
            model_name = self._config.minilm_model_name
            max_length = 256
            tokenizer_source = self._config.onnx_minilm_model_path or model_name
        return model_name, max_length, tokenizer_source

    def _load_onnx_session(
        self,
        model_type: ModelType,
        runtime: RuntimeSelection,
    ) -> Any:
        """Load an ONNX Runtime session for a pre-exported model."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX backend selected but onnxruntime is not installed.")

        model_dir = self._onnx_model_dir(model_type)
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

    def _onnx_model_dir(self, model_type: ModelType) -> Path:
        """Return the configured directory for an exported ONNX model."""
        model_dir = (
            self._config.onnx_model_path
            if model_type == ModelType.FINBERT
            else self._config.onnx_minilm_model_path
        )
        if not model_dir:
            raise RuntimeError(
                f"ONNX backend selected for {model_type.value}, "
                "but no exported model path is configured."
            )
        return Path(model_dir)

    def _get_model(self, model_type: ModelType) -> Any:
        """Get model, initializing if needed."""
        self._initialize(model_type)
        return self._models[model_type]

    def _get_tokenizer(self, model_type: ModelType) -> AutoTokenizer:
        """Get tokenizer, initializing if needed."""
        self._initialize(model_type)
        return self._tokenizers[model_type]

    @property
    def model(self) -> Any:
        """Get FinBERT model/session (backward compatibility)."""
        return self._get_model(ModelType.FINBERT)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get FinBERT tokenizer (backward compatibility)."""
        return self._get_tokenizer(ModelType.FINBERT)

    @property
    def device(self) -> Any:
        """Get active device or accelerator."""
        return self._detect_device()

    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _make_cache_key(self, text: str, model_type: ModelType) -> str:
        """Create cache key with model prefix to avoid collisions."""
        content_hash = self._compute_content_hash(text)
        return f"{self._config.cache_key_prefix}{model_type.value}:{content_hash}"

    async def _get_cached_embedding(
        self, text: str, model_type: ModelType = ModelType.FINBERT
    ) -> list[float] | None:
        """Try to retrieve embedding from cache."""
        if not self._config.cache_enabled or not self._redis:
            return None

        cache_key = self._make_cache_key(text, model_type)

        try:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug("Cache hit for embedding", cache_key=cache_key)
                return json.loads(cached)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))

        return None

    async def _cache_embedding(
        self, text: str, embedding: list[float], model_type: ModelType = ModelType.FINBERT
    ) -> None:
        """Store embedding in cache."""
        if not self._config.cache_enabled or not self._redis:
            return

        cache_key = self._make_cache_key(text, model_type)

        try:
            await self._redis.setex(
                cache_key,
                self._config.cache_ttl_seconds,
                json.dumps(embedding),
            )
            logger.debug("Cached embedding", cache_key=cache_key)
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))

    def _chunk_text(self, text: str, model_type: ModelType = ModelType.FINBERT) -> list[str]:
        """Split text into overlapping chunks that fit within model context."""
        tokenizer = self._get_tokenizer(model_type)
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if model_type == ModelType.MINILM:
            max_tokens = 256 - 2
        else:
            max_tokens = self._config.max_sequence_length - 2

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        stride = max_tokens - self._config.chunk_overlap
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break
            start += stride

        logger.debug("Split text into chunks", count=len(chunks), model_type=model_type.value)
        return chunks

    def _mean_pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Mean pool multiple embeddings into one."""
        return np.mean(embeddings, axis=0)

    def _mean_pool_token_embeddings(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Mean-pool token embeddings into sentence embeddings."""
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _embed_single(self, text: str, model_type: ModelType = ModelType.FINBERT) -> np.ndarray:
        """Generate embedding for a single text (internal, synchronous)."""
        return self._embed_text_batch([text], model_type)[0]

    def _embed_text_batch(
        self,
        texts: list[str],
        model_type: ModelType = ModelType.FINBERT,
    ) -> np.ndarray:
        """Generate embeddings for a batch of already chunked texts."""
        if not texts:
            dim = (
                self._config.embedding_dim
                if model_type == ModelType.FINBERT
                else self._config.minilm_embedding_dim
            )
            return np.empty((0, dim), dtype=np.float32)

        runtime = self._resolve_runtime()
        tokenizer = self._get_tokenizer(model_type)
        model = self._get_model(model_type)
        max_length = 256 if model_type == ModelType.MINILM else self._config.max_sequence_length

        if runtime.backend == "torch":
            assert TORCH_AVAILABLE
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            device = self.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            return embedding.cpu().numpy()

        inputs = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        session_inputs = {}
        expected_inputs = {meta.name for meta in model.get_inputs()}
        for name, value in inputs.items():
            if name in expected_inputs:
                session_inputs[name] = value.astype(np.int64)

        outputs = model.run(None, session_inputs)
        token_embeddings = outputs[0]
        attention_mask = session_inputs["attention_mask"]
        return self._mean_pool_token_embeddings(token_embeddings, attention_mask)

    async def embed(
        self,
        text: str,
        model_type: ModelType = ModelType.FINBERT,
    ) -> list[float]:
        """Generate embedding for a single text using the specified model."""
        if not text.strip():
            dim = (
                self._config.embedding_dim
                if model_type == ModelType.FINBERT
                else self._config.minilm_embedding_dim
            )
            return [0.0] * dim

        cached = await self._get_cached_embedding(text, model_type)
        if cached is not None:
            return cached

        chunks = self._chunk_text(text, model_type)
        chunk_embeddings = self._embed_text_batch(chunks, model_type)
        embedding = chunk_embeddings[0] if len(chunks) == 1 else self._mean_pool(chunk_embeddings)

        result = embedding.tolist()
        await self._cache_embedding(text, result, model_type)
        return result

    async def embed_finbert(self, text: str) -> list[float]:
        """Generate FinBERT embedding (768-dim) for financial text."""
        return await self.embed(text, ModelType.FINBERT)

    async def embed_minilm(self, text: str) -> list[float]:
        """Generate MiniLM embedding (384-dim) for short text."""
        return await self.embed(text, ModelType.MINILM)

    async def embed_batch(
        self,
        texts: list[str],
        model_type: ModelType = ModelType.FINBERT,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with true batch inference."""
        if not texts:
            return []

        dim = (
            self._config.embedding_dim
            if model_type == ModelType.FINBERT
            else self._config.minilm_embedding_dim
        )
        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = [0.0] * dim
                continue

            cached = await self._get_cached_embedding(text, model_type)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append((i, text))

        if show_progress:
            logger.info(
                "Embedding batch",
                model_type=model_type.value,
                total=len(texts),
                cached=len(texts) - len(to_embed),
                uncached=len(to_embed),
                backend=self._resolve_runtime().backend,
            )

        batch_size = self._config.batch_size
        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start : batch_start + batch_size]
            single_chunk_items: list[tuple[int, str]] = []
            chunked_items: list[tuple[int, list[str], str]] = []

            for idx, text in batch:
                chunks = self._chunk_text(text, model_type)
                if len(chunks) == 1:
                    single_chunk_items.append((idx, chunks[0]))
                else:
                    chunked_items.append((idx, chunks, text))

            if single_chunk_items:
                embeddings = self._embed_text_batch(
                    [chunk for _, chunk in single_chunk_items],
                    model_type,
                )
                for emb_idx, (result_idx, _) in enumerate(single_chunk_items):
                    embedding_list = embeddings[emb_idx].tolist()
                    original_text = texts[result_idx]
                    results[result_idx] = embedding_list
                    await self._cache_embedding(original_text, embedding_list, model_type)

            for result_idx, chunks, original_text in chunked_items:
                chunk_embeddings = self._embed_text_batch(chunks, model_type)
                embedding = self._mean_pool(chunk_embeddings).tolist()
                results[result_idx] = embedding
                await self._cache_embedding(original_text, embedding, model_type)

        return results  # type: ignore[return-value]

    async def close(self) -> None:
        """Clean up resources."""
        self._models.clear()
        self._tokenizers.clear()
        self._runtime = None
        self._device = None
        self._initialized = dict.fromkeys(self._initialized, False)
        logger.info("EmbeddingService closed")

    @property
    def is_initialized(self) -> bool:
        """Check if any model is loaded."""
        return any(self._initialized.values())

    def is_model_initialized(self, model_type: ModelType) -> bool:
        """Check if specific model is loaded."""
        return self._initialized.get(model_type, False)

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        runtime = self._runtime
        return {
            "initialized": self._initialized.copy(),
            "finbert_model": self._config.model_name,
            "minilm_model": self._config.minilm_model_name,
            "device": str(self._device) if self._device is not None else "not initialized",
            "backend": runtime.backend if runtime else self._config.backend,
            "accelerator": runtime.accelerator if runtime else "not initialized",
            "execution_provider": runtime.onnx_provider if runtime else None,
            "cache_enabled": self._config.cache_enabled,
            "finbert_dim": self._config.embedding_dim,
            "minilm_dim": self._config.minilm_embedding_dim,
            "onnx_model_path": self._config.onnx_model_path,
            "onnx_minilm_model_path": self._config.onnx_minilm_model_path,
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


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()
