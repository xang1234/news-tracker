"""Pytest fixtures for embedding tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from src.embedding.config import EmbeddingConfig
from src.embedding.queue import EmbeddingJob
from src.embedding.service import EmbeddingService, ModelType


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    """Configuration for testing."""
    return EmbeddingConfig(
        model_name="ProsusAI/finbert",
        embedding_dim=768,
        batch_size=4,
        use_fp16=False,
        device="cpu",
        cache_enabled=False,
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample 768-dimensional embedding."""
    # Use a deterministic random seed for reproducibility
    np.random.seed(42)
    embedding = np.random.randn(768).astype(np.float32)
    # Normalize to unit vector
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create multiple sample embeddings."""
    np.random.seed(42)
    embeddings = []
    for i in range(5):
        np.random.seed(42 + i)
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.tolist())
    return embeddings


class MockTokenizerOutput(dict):
    """Mock tokenizer output that behaves like a dict."""

    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self["input_ids"] = input_ids
        self["attention_mask"] = attention_mask

    def items(self):
        return [("input_ids", self["input_ids"]), ("attention_mask", self["attention_mask"])]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    def encode_side_effect(text, add_special_tokens=True):
        # Simple tokenization: split by whitespace
        tokens = text.split()
        return list(range(len(tokens)))

    def decode_side_effect(tokens, skip_special_tokens=True):
        # Return placeholder text
        return " ".join(["word"] * len(tokens))

    def call_side_effect(text, return_tensors=None, truncation=True,
                        max_length=512, padding=True):
        # Simple mock tokenization
        if isinstance(text, str):
            seq_len = max(min(len(text.split()), max_length), 1)
        else:
            seq_len = max(min(max(len(t.split()) for t in text), max_length), 1)

        batch_size = 1 if isinstance(text, str) else len(text)

        return MockTokenizerOutput(
            input_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        )

    tokenizer.encode = MagicMock(side_effect=encode_side_effect)
    tokenizer.decode = MagicMock(side_effect=decode_side_effect)
    tokenizer.side_effect = call_side_effect
    tokenizer.return_value = call_side_effect("test text")
    tokenizer.model_max_length = 512

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    model = MagicMock()

    def forward_side_effect(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            batch_size, seq_len = 1, 10
        else:
            batch_size, seq_len = input_ids.shape

        # Create mock output with last_hidden_state
        output = MagicMock()
        # Generate deterministic embeddings based on batch/seq size
        torch.manual_seed(42)
        hidden_state = torch.randn(batch_size, seq_len, 768)
        output.last_hidden_state = hidden_state
        return output

    model.side_effect = forward_side_effect
    model.return_value = forward_side_effect()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    model.half = MagicMock(return_value=model)

    return model


@pytest.fixture
def mock_embedding_service(
    embedding_config, mock_tokenizer, mock_model, sample_embedding
) -> EmbeddingService:
    """Create an EmbeddingService with mocked model components."""
    service = EmbeddingService(config=embedding_config)

    # Create a callable mock for tokenizer
    def tokenizer_call(text, **kwargs):
        max_length = kwargs.get("max_length", 512)
        if isinstance(text, str):
            seq_len = max(min(len(text.split()), max_length), 1)
        else:
            seq_len = max(min(max(len(t.split()) for t in text), max_length), 1)
        batch_size = 1 if isinstance(text, str) else len(text)
        return MockTokenizerOutput(
            input_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        )

    mock_tokenizer.side_effect = tokenizer_call
    mock_tokenizer.return_value = tokenizer_call("test")

    # Inject mocks for both model types
    service._tokenizers = {
        ModelType.FINBERT: mock_tokenizer,
        ModelType.MINILM: mock_tokenizer,
    }
    service._models = {
        ModelType.FINBERT: mock_model,
        ModelType.MINILM: mock_model,
    }
    service._device = torch.device("cpu")
    service._initialized = {
        ModelType.FINBERT: True,
        ModelType.MINILM: True,
    }

    # Mock _embed_single to avoid numpy compatibility issues in test env
    # Return a deterministic embedding based on the input hash and model type
    def mock_embed_single(text, model_type=ModelType.FINBERT):
        np.random.seed(hash(text) % 2**32)
        dim = 768 if model_type == ModelType.FINBERT else 384
        emb = np.random.randn(dim).astype(np.float32)
        return emb / np.linalg.norm(emb)

    service._embed_single = mock_embed_single

    return service


@pytest.fixture
def sample_embedding_jobs() -> list[EmbeddingJob]:
    """Create sample embedding jobs for testing."""
    return [
        EmbeddingJob(
            message_id=f"msg_{i}",
            document_id=f"doc_{i}",
            priority=0,
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock()
    redis_mock.ping = AsyncMock()
    redis_mock.close = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value="msg_123")
    redis_mock.xreadgroup = AsyncMock(return_value=[])
    redis_mock.xack = AsyncMock()
    redis_mock.xrange = AsyncMock(return_value=[])
    redis_mock.xgroup_create = AsyncMock()
    redis_mock.xlen = AsyncMock(return_value=0)
    redis_mock.xpending = AsyncMock(return_value={"pending": 0})
    redis_mock.pipeline = MagicMock()

    return redis_mock
