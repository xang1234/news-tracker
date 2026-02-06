"""
BERTopic-based document clustering for financial news themes.

This module provides topic modeling and theme assignment for semiconductor
news content, using BERTopic with HDBSCAN clustering and c-TF-IDF
topic representations over FinBERT embeddings.

Components:
- ClusteringConfig: Configuration for the clustering service
"""

from src.clustering.config import ClusteringConfig

__all__ = [
    "ClusteringConfig",
]
