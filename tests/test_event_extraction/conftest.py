"""Shared fixtures for event extraction tests."""

import pytest

from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.patterns import PatternExtractor


@pytest.fixture
def event_config():
    """Default event extraction config."""
    return EventExtractionConfig()


@pytest.fixture
def pattern_extractor(event_config):
    """PatternExtractor with default config."""
    return PatternExtractor(config=event_config)


@pytest.fixture
def sample_semiconductor_texts():
    """Sample texts containing each event type for testing."""
    return {
        "capacity_expansion": [
            "TSMC is expanding fab capacity in Arizona by $40 billion in Q3 2026.",
            "Samsung announced a $17 billion investment in a new fab in Taylor, Texas.",
            "Intel plans to add new production lines at its Ohio facility.",
        ],
        "capacity_constraint": [
            "TSMC faces supply shortage of advanced packaging capacity.",
            "Lead times extended to 26 weeks for mature node chips.",
            "GlobalFoundries is unable to meet customer demand for automotive chips.",
        ],
        "product_launch": [
            "NVIDIA launched the H200 GPU accelerator for data centers in Q1 2026.",
            "AMD introduces the MI350 AI accelerator with improved performance.",
            "Intel begins mass production of Emerald Rapids processors.",
        ],
        "product_delay": [
            "NVIDIA delayed the Blackwell B200 GPU to Q4 2026.",
            "The H200 chip has been pushed back to next quarter.",
            "Intel's Arrow Lake desktop processors are behind schedule by 3 months.",
        ],
        "price_change": [
            "TSMC raised wafer prices by 5% for advanced nodes.",
            "Memory chip prices surged 20% due to AI demand.",
            "Samsung cuts prices for legacy node wafers by 10%.",
        ],
        "guidance_change": [
            "NVIDIA raised revenue guidance for Q4 to $22 billion.",
            "AMD expects revenue of $6.5 billion in Q1 2026.",
            "Intel warned of lower earnings due to weak PC demand.",
        ],
    }
