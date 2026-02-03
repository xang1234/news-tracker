"""
Embedding generation service using FinBERT and MiniLM.

Provides async embedding generation for financial documents with:
- Multi-model support (FinBERT 768-dim, MiniLM 384-dim)
- Model selection based on platform and content length
- Lazy model loading for efficient resource usage
- Automatic device detection (GPU/CPU/MPS)
- Chunking with mean pooling for long documents
- Redis caching to avoid recomputation
- FP16 support for GPU acceleration
"""

import hashlib
import json
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog
import torch
from transformers import AutoModel, AutoTokenizer

from src.embedding.config import EmbeddingConfig

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Available embedding models."""

    FINBERT = "finbert"
    MINILM = "minilm"


class EmbeddingService:
    """
    Service for generating document embeddings using FinBERT and MiniLM.

    Uses lazy initialization to defer model loading until first use,
    which is important for services that may not need embeddings.

    Features:
    - Multi-model support (FinBERT 768-dim, MiniLM 384-dim)
    - Model selection based on platform and content length
    - Automatic device selection (CUDA > MPS > CPU)
    - FP16 inference for GPU acceleration
    - Chunking for documents exceeding 512 tokens
    - Redis caching using content hash keys (per-model)
    - Batch processing for efficiency

    Usage:
        service = EmbeddingService()
        # Auto-select model based on content
        embedding = await service.embed("Financial news about NVIDIA")
        # Explicit model selection
        embedding = await service.embed_minilm("Short tweet about $NVDA")
        embedding = await service.embed_finbert("Long financial analysis...")
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        redis_client: redis.Redis | None = None,
    ):
        """
        Initialize the embedding service.

        Model loading is deferred until first embed call.

        Args:
            config: Embedding configuration (uses defaults if None)
            redis_client: Redis client for caching (optional)
        """
        self._config = config or EmbeddingConfig()
        self._redis = redis_client

        # Model components (lazy loaded per model type)
        self._models: dict[ModelType, AutoModel] = {}
        self._tokenizers: dict[ModelType, AutoTokenizer] = {}
        self._device: torch.device | None = None

        # State tracking per model
        self._initialized: dict[ModelType, bool] = {
            ModelType.FINBERT: False,
            ModelType.MINILM: False,
        }

        logger.info(
            "EmbeddingService created",
            extra={
                "finbert_model": self._config.model_name,
                "minilm_model": self._config.minilm_model_name,
                "cache_enabled": self._config.cache_enabled,
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
            logger.info("Using CUDA device for embeddings")
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device for embeddings")
            self._device = torch.device("mps")
        else:
            logger.info("Using CPU for embeddings")
            self._device = torch.device("cpu")

        return self._device

    def _initialize(self, model_type: ModelType = ModelType.FINBERT) -> None:
        """Initialize model and tokenizer on first use."""
        if self._initialized.get(model_type, False):
            return

        device = self._detect_device()

        if model_type == ModelType.FINBERT:
            model_name = self._config.model_name
            max_length = self._config.max_sequence_length
        else:  # MiniLM
            model_name = self._config.minilm_model_name
            max_length = 256  # MiniLM uses shorter sequences

        logger.info(f"Loading embedding model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_length,
        )

        # Load model
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        # Enable FP16 for CUDA devices
        if self._config.use_fp16 and device.type == "cuda":
            model = model.half()
            logger.info(f"FP16 inference enabled for {model_type.value}")

        self._models[model_type] = model
        self._tokenizers[model_type] = tokenizer
        self._initialized[model_type] = True

        logger.info(
            f"Embedding model loaded: {model_type.value}",
            extra={
                "device": str(device),
                "fp16": self._config.use_fp16 and device.type == "cuda",
            },
        )

    def _get_model(self, model_type: ModelType) -> AutoModel:
        """Get model, initializing if needed."""
        self._initialize(model_type)
        return self._models[model_type]

    def _get_tokenizer(self, model_type: ModelType) -> AutoTokenizer:
        """Get tokenizer, initializing if needed."""
        self._initialize(model_type)
        return self._tokenizers[model_type]

    @property
    def model(self) -> AutoModel:
        """Get FinBERT model (backward compatibility)."""
        return self._get_model(ModelType.FINBERT)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get FinBERT tokenizer (backward compatibility)."""
        return self._get_tokenizer(ModelType.FINBERT)

    @property
    def device(self) -> torch.device:
        """Get device, detecting if needed."""
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
                logger.debug(f"Cache hit for embedding: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

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
            logger.debug(f"Cached embedding: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _chunk_text(self, text: str, model_type: ModelType = ModelType.FINBERT) -> list[str]:
        """
        Split text into overlapping chunks that fit within model context.

        Uses tokenizer to ensure chunks respect token limits.
        """
        tokenizer = self._get_tokenizer(model_type)

        # Tokenize the full text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Determine max length based on model
        if model_type == ModelType.MINILM:
            max_tokens = 256 - 2  # MiniLM has shorter context
        else:
            max_tokens = self._config.max_sequence_length - 2  # Reserve for [CLS] and [SEP]

        if len(tokens) <= max_tokens:
            return [text]

        # Split into overlapping chunks
        chunks = []
        stride = max_tokens - self._config.chunk_overlap
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode chunk back to text
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break
            start += stride

        logger.debug(f"Split text into {len(chunks)} chunks for {model_type.value}")
        return chunks

    def _mean_pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Mean pool multiple embeddings into one."""
        return np.mean(embeddings, axis=0)

    def _embed_single(
        self, text: str, model_type: ModelType = ModelType.FINBERT
    ) -> np.ndarray:
        """Generate embedding for a single text (internal, synchronous)."""
        model = self._get_model(model_type)
        tokenizer = self._get_tokenizer(model_type)
        device = self.device

        # Determine max length based on model
        max_length = 256 if model_type == ModelType.MINILM else self._config.max_sequence_length

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pool over sequence dimension (excluding special tokens)
        # outputs.last_hidden_state shape: [batch, seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state

        # Mask padding tokens
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        return embedding.cpu().numpy()[0]

    async def embed(
        self,
        text: str,
        model_type: ModelType = ModelType.FINBERT,
    ) -> list[float]:
        """
        Generate embedding for a single text using specified model.

        Handles long documents by chunking and mean pooling.
        Uses cache when available.

        Args:
            text: Text to embed
            model_type: Which model to use (FINBERT or MINILM)

        Returns:
            Embedding as list of floats (768-dim for FinBERT, 384-dim for MiniLM)
        """
        if not text.strip():
            # Return zero vector for empty text
            dim = (
                self._config.embedding_dim
                if model_type == ModelType.FINBERT
                else self._config.minilm_embedding_dim
            )
            return [0.0] * dim

        # Check cache first
        cached = await self._get_cached_embedding(text, model_type)
        if cached is not None:
            return cached

        # Chunk if needed and embed
        chunks = self._chunk_text(text, model_type)

        if len(chunks) == 1:
            embedding = self._embed_single(chunks[0], model_type)
        else:
            # Embed each chunk and mean pool
            chunk_embeddings = np.array([
                self._embed_single(chunk, model_type) for chunk in chunks
            ])
            embedding = self._mean_pool(chunk_embeddings)

        result = embedding.tolist()

        # Cache the result
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
        """
        Generate embeddings for multiple texts.

        Processes in batches for efficiency. Handles caching for each text.

        Args:
            texts: List of texts to embed
            model_type: Which model to use (FINBERT or MINILM)
            show_progress: Whether to log progress

        Returns:
            List of embeddings (768-dim for FinBERT, 384-dim for MiniLM)
        """
        if not texts:
            return []

        dim = (
            self._config.embedding_dim
            if model_type == ModelType.FINBERT
            else self._config.minilm_embedding_dim
        )

        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        # Check cache for each text
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
                f"Embedding batch ({model_type.value}): {len(texts)} texts, "
                f"{len(texts) - len(to_embed)} cached, "
                f"{len(to_embed)} to generate"
            )

        # Process uncached texts in batches
        batch_size = self._config.batch_size
        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start : batch_start + batch_size]

            for idx, text in batch:
                # Chunk and embed each text
                chunks = self._chunk_text(text, model_type)

                if len(chunks) == 1:
                    embedding = self._embed_single(chunks[0], model_type)
                else:
                    chunk_embeddings = np.array([
                        self._embed_single(chunk, model_type) for chunk in chunks
                    ])
                    embedding = self._mean_pool(chunk_embeddings)

                embedding_list = embedding.tolist()
                results[idx] = embedding_list

                # Cache the result
                await self._cache_embedding(text, embedding_list, model_type)

        return results  # type: ignore

    async def close(self) -> None:
        """Clean up resources."""
        # Models are cleaned up by Python garbage collection
        # Redis client is managed externally
        self._models.clear()
        self._tokenizers.clear()
        self._initialized = {k: False for k in self._initialized}
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
        return {
            "initialized": self._initialized.copy(),
            "finbert_model": self._config.model_name,
            "minilm_model": self._config.minilm_model_name,
            "device": str(self._device) if self._device else "not initialized",
            "cache_enabled": self._config.cache_enabled,
            "finbert_dim": self._config.embedding_dim,
            "minilm_dim": self._config.minilm_embedding_dim,
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
