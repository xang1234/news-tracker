"""Test fixtures for sentiment analysis service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService


@pytest.fixture
def sentiment_config() -> SentimentConfig:
    """Create test sentiment configuration."""
    return SentimentConfig(
        model_name="ProsusAI/finbert",
        batch_size=4,
        use_fp16=False,
        backend="torch",
        device="cpu",
        cache_enabled=False,
        enable_entity_sentiment=True,
        entity_context_window=50,
    )


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()

    def tokenizer_call(text, **kwargs):
        batch_size = 1 if isinstance(text, str) else len(text)
        input_ids = torch.tensor([[101, 1000, 2000, 102]] * batch_size)
        attention_mask = torch.tensor([[1, 1, 1, 1]] * batch_size)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    tokenizer.side_effect = tokenizer_call
    tokenizer.return_value = tokenizer_call("test")
    return tokenizer


@pytest.fixture
def mock_model():
    """Create mock sentiment model."""
    model = MagicMock()

    def model_call(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        batch_size = 1 if input_ids is None else input_ids.shape[0]
        logits = torch.tensor([[2.5, -1.0, 0.5]] * batch_size)  # Strong positive
        return MagicMock(logits=logits)

    model.side_effect = model_call
    model.return_value = model_call()
    return model


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock()
    redis_mock.ping = AsyncMock()
    return redis_mock


@pytest.fixture
def mock_sentiment_service(
    sentiment_config: SentimentConfig,
    mock_tokenizer,
    mock_model,
    mock_redis,
) -> SentimentService:
    """Create mock sentiment service with mocked model."""
    service = SentimentService(
        config=sentiment_config,
        redis_client=mock_redis,
    )

    # Inject mocks
    service._tokenizer = mock_tokenizer
    service._model = mock_model
    service._device = torch.device("cpu")
    service._initialized = True

    return service


@pytest.fixture
def sample_entities() -> list[dict]:
    """Create sample entities for testing."""
    return [
        {
            "text": "NVIDIA",
            "type": "COMPANY",
            "normalized": "NVIDIA",
            "start": 0,
            "end": 6,
            "confidence": 0.95,
        },
        {
            "text": "AMD",
            "type": "COMPANY",
            "normalized": "AMD",
            "start": 50,
            "end": 53,
            "confidence": 0.90,
        },
    ]


@pytest.fixture
def sample_positive_text() -> str:
    """Sample text with positive sentiment."""
    return "NVIDIA stock surged 10% on strong AI chip demand and record earnings."


@pytest.fixture
def sample_negative_text() -> str:
    """Sample text with negative sentiment."""
    return "NVIDIA shares plunged amid supply chain concerns and weak guidance."


@pytest.fixture
def sample_neutral_text() -> str:
    """Sample text with neutral sentiment."""
    return "NVIDIA will report earnings next week according to the company schedule."
