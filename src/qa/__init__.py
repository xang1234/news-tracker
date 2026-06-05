"""Cited Q&A over the structured claim layer (RAG epic q7g.3).

``CitedQAService.answer`` retrieves grounding claims for a free-text question,
has the LLM synthesize an answer citing them (citations validated against the
retrieved set), and returns an explicit grounding-sufficiency signal —
refusing (``insufficient``) when the evidence is too thin to answer.
"""

from __future__ import annotations

from src.qa.config import QAConfig
from src.qa.schemas import AnswerSegment, CitedAnswer

__all__ = [
    "AnswerSegment",
    "CitedAnswer",
    "QAConfig",
]
