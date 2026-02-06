"""
BERTopic-based document clustering for financial news themes.

This module provides topic modeling and theme assignment for semiconductor
news content, using BERTopic with HDBSCAN clustering and c-TF-IDF
topic representations over FinBERT embeddings.

Components:
- ClusteringConfig: Configuration for the clustering service
- ThemeCluster: Dataclass representing a discovered theme
- BERTopicService: Service for fitting BERTopic models on document embeddings
"""

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService

__all__ = [
    "ClusteringConfig",
    "ThemeCluster",
    "BERTopicService",
]
