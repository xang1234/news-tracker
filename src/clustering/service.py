"""
BERTopic clustering service for discovering financial news themes.

Provides batch clustering of documents using pre-computed FinBERT embeddings
to discover thematic groups via UMAP dimensionality reduction, HDBSCAN
density-based clustering, and c-TF-IDF topic representation.

Architecture:
- Sync fit() for CPU-bound UMAP/HDBSCAN/c-TF-IDF operations
- Deferred imports for heavy dependencies (bertopic, hdbscan, umap)
- Model created per fit() call (training artifact, not reused for inference)
- Deterministic theme IDs from topic word hashes for cross-run stability
"""

import logging
import time
from typing import Any

import numpy as np

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster

logger = logging.getLogger(__name__)


class BERTopicService:
    """
    BERTopic-based clustering service for theme discovery.

    Runs BERTopic fit_transform on historical documents with pre-computed
    embeddings to discover baseline themes. Each fit() call produces a
    fresh model and a new set of ThemeCluster objects.

    Usage:
        >>> service = BERTopicService()
        >>> themes = service.fit(documents, embeddings, document_ids)
        >>> for theme_id, theme in themes.items():
        ...     print(f"{theme.name}: {theme.document_count} docs")
        gpu_architecture_nvidia: 25 docs
        memory_hbm3e_bandwidth: 18 docs

    Note:
        BERTopic and its dependencies (hdbscan, umap-learn) are imported
        lazily inside _create_model() to avoid slow module-level imports.
    """

    def __init__(self, config: ClusteringConfig | None = None):
        """
        Initialize clustering service.

        Args:
            config: Clustering configuration. If None, uses default config.
        """
        self.config = config or ClusteringConfig()
        self._model: Any = None
        self._themes: dict[str, ThemeCluster] = {}
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether fit() has been called successfully."""
        return self._initialized

    @property
    def themes(self) -> dict[str, ThemeCluster]:
        """Return a copy of discovered themes."""
        return dict(self._themes)

    @property
    def model(self) -> Any:
        """Access the underlying BERTopic model (None before fit)."""
        return self._model

    def fit(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        document_ids: list[str],
    ) -> dict[str, ThemeCluster]:
        """
        Discover themes from documents using BERTopic.

        Runs UMAP → HDBSCAN → c-TF-IDF pipeline on pre-computed embeddings
        to cluster documents into thematic groups. Outlier documents (BERTopic
        topic -1) are excluded from theme assignments.

        Args:
            documents: List of document text strings.
            embeddings: Pre-computed embedding matrix of shape (n_docs, embedding_dim).
            document_ids: List of document IDs corresponding to each document.

        Returns:
            Dictionary mapping theme_id to ThemeCluster objects.
            Returns empty dict on empty input or internal errors.

        Raises:
            ValueError: If input lengths don't match.
        """
        # Validate inputs
        n_docs = len(documents)
        if n_docs != len(document_ids):
            raise ValueError(
                f"documents ({n_docs}) and document_ids ({len(document_ids)}) "
                f"must have the same length"
            )
        if n_docs != embeddings.shape[0]:
            raise ValueError(
                f"documents ({n_docs}) and embeddings ({embeddings.shape[0]}) "
                f"must have the same length"
            )

        if n_docs == 0:
            logger.info("Empty input, returning no themes")
            return {}

        start_time = time.monotonic()

        try:
            # Create a fresh model for this fit run
            model = self._create_model()

            # Run BERTopic fit_transform with pre-computed embeddings
            topics, _probs = model.fit_transform(documents, embeddings=embeddings)

            # Build ThemeCluster objects from results
            self._themes = self._build_themes(
                model=model,
                topics=topics,
                embeddings=embeddings,
                document_ids=document_ids,
            )
            self._model = model
            self._initialized = True

            elapsed = time.monotonic() - start_time
            n_outliers = sum(1 for t in topics if t == -1)
            logger.info(
                f"Clustering complete: {len(self._themes)} themes discovered, "
                f"{n_outliers} outliers, {elapsed:.2f}s elapsed"
            )

            return dict(self._themes)

        except Exception:
            logger.exception("Clustering failed")
            return {}

    def _create_model(self) -> Any:
        """
        Create a BERTopic model with configured sub-components.

        Imports are deferred to avoid slow module-level loading of
        bertopic, hdbscan, and umap-learn (~2s combined).

        Returns:
            Configured BERTopic instance ready for fit_transform.
        """
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.umap_random_state,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
            prediction_data=self.config.hdbscan_prediction_data,
        )

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        representation_model = KeyBERTInspired()

        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            top_n_words=self.config.top_n_words,
            nr_topics=self.config.nr_topics,
            embedding_model=None,  # We provide pre-computed embeddings
        )

        return model

    def _build_themes(
        self,
        model: Any,
        topics: list[int],
        embeddings: np.ndarray,
        document_ids: list[str],
    ) -> dict[str, ThemeCluster]:
        """
        Build ThemeCluster objects from BERTopic fit results.

        For each non-outlier topic, collects assigned document IDs,
        computes the centroid embedding, and extracts representative
        topic words from the model.

        Args:
            model: Fitted BERTopic model.
            topics: Topic assignment per document (-1 = outlier).
            embeddings: Document embedding matrix.
            document_ids: Document ID list.

        Returns:
            Dictionary mapping theme_id to ThemeCluster.
        """
        themes: dict[str, ThemeCluster] = {}

        # Get unique non-outlier topic IDs
        unique_topics = sorted(set(t for t in topics if t != -1))

        for topic_id in unique_topics:
            # Find document indices for this topic
            indices = [i for i, t in enumerate(topics) if t == topic_id]

            # Compute centroid from document embeddings
            topic_embeddings = embeddings[indices]
            centroid = np.mean(topic_embeddings, axis=0)

            # Get topic words from the model
            topic_words = model.get_topic(topic_id)

            # Collect document IDs
            topic_doc_ids = [document_ids[i] for i in indices]

            # Generate deterministic ID and readable name
            theme_id = ThemeCluster.generate_theme_id(topic_words)
            name = ThemeCluster.generate_name(topic_words)

            theme = ThemeCluster(
                theme_id=theme_id,
                name=name,
                topic_words=topic_words,
                centroid=centroid,
                document_count=len(indices),
                document_ids=topic_doc_ids,
                metadata={"bertopic_topic_id": topic_id},
            )
            themes[theme_id] = theme

        return themes

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current clustering state.

        Returns:
            Dictionary with theme count, total documents, and per-theme stats.
        """
        if not self._initialized:
            return {
                "initialized": False,
                "n_themes": 0,
                "n_documents": 0,
            }

        total_docs = sum(t.document_count for t in self._themes.values())
        return {
            "initialized": True,
            "n_themes": len(self._themes),
            "n_documents": total_docs,
            "themes": [
                {
                    "theme_id": t.theme_id,
                    "name": t.name,
                    "document_count": t.document_count,
                    "top_words": [w for w, _ in t.topic_words[:5]],
                }
                for t in self._themes.values()
            ],
        }
