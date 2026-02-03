"""Pytest fixtures for NER tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.ner.config import NERConfig
from src.ner.schemas import FinancialEntity


@pytest.fixture
def ner_config() -> NERConfig:
    """Default NER configuration for testing."""
    return NERConfig(
        spacy_model="en_core_web_sm",  # Use smaller model for tests
        fallback_model="en_core_web_sm",
        enable_coreference=False,  # Disable coref for faster tests
        fuzzy_threshold=80,
        confidence_threshold=0.5,
    )


@pytest.fixture
def ner_config_with_coref() -> NERConfig:
    """NER configuration with coreference enabled."""
    return NERConfig(
        spacy_model="en_core_web_sm",
        enable_coreference=True,
        fuzzy_threshold=85,
    )


@pytest.fixture
def sample_financial_texts() -> list[str]:
    """Sample financial texts with various entity types."""
    return [
        # Text with company names and products
        "Nvidia announced the H100 GPU will power the next generation of AI workloads.",
        # Text with technologies
        "TSMC's 3nm process with CoWoS packaging enables HBM3E memory integration.",
        # Text with metrics
        "AMD reported Q4 revenue of $5.6 billion, up 10% YoY with gross margin of 51%.",
        # Text with cashtags
        "$NVDA and $AMD are leading the AI chip market. $INTC is catching up.",
        # Complex text with multiple entity types
        (
            "Taiwan Semiconductor Manufacturing Company announced plans to invest "
            "$40 billion in advanced EUV lithography for 2nm production. "
            "The company expects 15% revenue growth in the datacenter segment."
        ),
        # Text with coreference candidates
        (
            "Nvidia unveiled the Blackwell architecture. The company claims "
            "it delivers 4x the performance of Hopper for AI training."
        ),
    ]


@pytest.fixture
def sample_entity() -> FinancialEntity:
    """Sample financial entity for testing."""
    return FinancialEntity(
        text="Nvidia",
        type="COMPANY",
        normalized="NVIDIA",
        start=0,
        end=6,
        confidence=0.95,
        metadata={"ticker": "NVDA"},
    )


@pytest.fixture
def overlapping_entities() -> list[FinancialEntity]:
    """Entities with overlapping spans for deduplication testing."""
    return [
        FinancialEntity(
            text="Taiwan Semiconductor",
            type="COMPANY",
            normalized="TAIWAN SEMICONDUCTOR",
            start=0,
            end=20,
            confidence=0.9,
            metadata={"ticker": "TSM"},
        ),
        FinancialEntity(
            text="Taiwan Semiconductor Manufacturing",
            type="COMPANY",
            normalized="TAIWAN SEMICONDUCTOR MANUFACTURING",
            start=0,
            end=34,
            confidence=0.95,
            metadata={"ticker": "TSM"},
        ),
        FinancialEntity(
            text="Manufacturing",
            type="COMPANY",
            normalized="MANUFACTURING",
            start=21,
            end=34,
            confidence=0.5,
            metadata={},
        ),
    ]


class MockSpacyDoc:
    """Mock spaCy Doc for unit tests."""

    def __init__(self, text: str, entities: list[tuple[str, int, int, str]]):
        """
        Initialize mock doc.

        Args:
            text: Document text
            entities: List of (text, start, end, label) tuples
        """
        self.text = text
        self._ents = []
        for ent_text, start, end, label in entities:
            ent = MagicMock()
            ent.text = ent_text
            ent.start_char = start
            ent.end_char = end
            ent.label_ = label
            self._ents.append(ent)

    @property
    def ents(self):
        return self._ents


class MockSpacyNlp:
    """Mock spaCy nlp pipeline for unit tests."""

    def __init__(self, entities_map: dict[str, list[tuple[str, int, int, str]]] | None = None):
        """
        Initialize mock nlp.

        Args:
            entities_map: Mapping from text to list of entities
        """
        self._entities_map = entities_map or {}
        self.meta = {"name": "mock_model"}

    def __call__(self, text: str) -> MockSpacyDoc:
        """Process text and return mock doc."""
        entities = self._entities_map.get(text, [])
        return MockSpacyDoc(text, entities)

    def pipe(self, texts: list[str]):
        """Process multiple texts."""
        for text in texts:
            yield self(text)

    def add_pipe(self, *args, **kwargs):
        """Mock add_pipe."""
        ruler = MagicMock()
        ruler.add_patterns = MagicMock()
        return ruler


@pytest.fixture
def mock_spacy_nlp():
    """Create a mock spaCy nlp pipeline."""
    entities_map = {
        "Nvidia announced the H100 GPU will power the next generation of AI workloads.": [
            ("Nvidia", 0, 6, "ORG"),
            ("H100", 21, 25, "PRODUCT"),
            ("GPU", 26, 29, "PRODUCT"),
        ],
        "TSMC's 3nm process with CoWoS packaging enables HBM3E memory integration.": [
            ("TSMC", 0, 4, "ORG"),
            ("3nm", 7, 10, "TECHNOLOGY"),
            ("CoWoS", 24, 29, "TECHNOLOGY"),
            ("HBM3E", 49, 54, "TECHNOLOGY"),
        ],
        "AMD reported Q4 revenue of $5.6 billion, up 10% YoY with gross margin of 51%.": [
            ("AMD", 0, 3, "ORG"),
            ("$5.6 billion", 27, 39, "MONEY"),
            ("10%", 44, 47, "PERCENT"),
            ("51%", 73, 76, "PERCENT"),
        ],
    }
    return MockSpacyNlp(entities_map)


@pytest.fixture
def mock_ner_service(ner_config, mock_spacy_nlp):
    """Create NERService with mocked spaCy."""
    from src.ner.service import NERService

    service = NERService(config=ner_config)
    # Manually set up the service without calling _initialize
    service._nlp = mock_spacy_nlp
    service._coref_model = None
    service._initialized = True
    return service
