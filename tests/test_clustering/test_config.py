"""Tests for clustering configuration."""

import pytest

from src.clustering.config import ClusteringConfig


class TestClusteringConfig:
    """Tests for ClusteringConfig settings."""

    def test_default_values(self):
        """Should have sensible defaults for financial text clustering."""
        config = ClusteringConfig()

        # UMAP defaults
        assert config.umap_n_neighbors == 15
        assert config.umap_n_components == 10
        assert config.umap_min_dist == 0.0
        assert config.umap_metric == "cosine"
        assert config.umap_random_state == 42

        # HDBSCAN defaults
        assert config.hdbscan_min_cluster_size == 10
        assert config.hdbscan_min_samples == 5
        assert config.hdbscan_cluster_selection_method == "eom"
        assert config.hdbscan_prediction_data is True

        # c-TF-IDF defaults
        assert config.top_n_words == 10
        assert config.nr_topics is None

        # Assignment thresholds
        assert config.similarity_threshold_assign == 0.75
        assert config.similarity_threshold_merge == 0.85
        assert config.similarity_threshold_new == 0.30

        # Centroid update
        assert config.centroid_learning_rate == 0.01

        # Model persistence
        assert config.model_save_dir == "models/clustering"

    def test_queue_defaults(self):
        """Should have Redis stream defaults matching the project pattern."""
        config = ClusteringConfig()

        assert config.stream_name == "clustering_queue"
        assert config.consumer_group == "clustering_workers"
        assert config.max_stream_length == 50_000
        assert config.dlq_stream_name == "clustering_queue:dlq"

    def test_worker_defaults(self):
        """Should have worker defaults matching the project pattern."""
        config = ClusteringConfig()

        assert config.worker_batch_timeout == 10.0
        assert config.worker_idle_timeout == 60.0
        assert config.idle_timeout_ms == 30_000
        assert config.max_delivery_attempts == 3

    def test_env_override(self, monkeypatch):
        """Should allow environment variable overrides with CLUSTERING_ prefix."""
        monkeypatch.setenv("CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE", "20")
        monkeypatch.setenv("CLUSTERING_UMAP_N_COMPONENTS", "15")
        monkeypatch.setenv("CLUSTERING_SIMILARITY_THRESHOLD_ASSIGN", "0.80")
        monkeypatch.setenv("CLUSTERING_MODEL_SAVE_DIR", "/tmp/models")

        config = ClusteringConfig()

        assert config.hdbscan_min_cluster_size == 20
        assert config.umap_n_components == 15
        assert config.similarity_threshold_assign == 0.80
        assert config.model_save_dir == "/tmp/models"

    def test_hdbscan_cluster_selection_method_literal(self):
        """Should only accept 'eom' or 'leaf' for cluster selection method."""
        config = ClusteringConfig(hdbscan_cluster_selection_method="leaf")
        assert config.hdbscan_cluster_selection_method == "leaf"

        config = ClusteringConfig(hdbscan_cluster_selection_method="eom")
        assert config.hdbscan_cluster_selection_method == "eom"

        with pytest.raises(ValueError):
            ClusteringConfig(hdbscan_cluster_selection_method="invalid")

    def test_min_cluster_size_bounds(self):
        """min_cluster_size should be >= 2."""
        config = ClusteringConfig(hdbscan_min_cluster_size=2)
        assert config.hdbscan_min_cluster_size == 2

        with pytest.raises(ValueError):
            ClusteringConfig(hdbscan_min_cluster_size=1)

    def test_similarity_threshold_bounds(self):
        """Similarity thresholds should be between 0.0 and 1.0."""
        config = ClusteringConfig(similarity_threshold_assign=0.0)
        assert config.similarity_threshold_assign == 0.0

        config = ClusteringConfig(similarity_threshold_assign=1.0)
        assert config.similarity_threshold_assign == 1.0

        with pytest.raises(ValueError):
            ClusteringConfig(similarity_threshold_assign=-0.1)

        with pytest.raises(ValueError):
            ClusteringConfig(similarity_threshold_assign=1.1)

    def test_centroid_learning_rate_bounds(self):
        """Learning rate should be > 0 and <= 1."""
        config = ClusteringConfig(centroid_learning_rate=0.5)
        assert config.centroid_learning_rate == 0.5

        config = ClusteringConfig(centroid_learning_rate=1.0)
        assert config.centroid_learning_rate == 1.0

        with pytest.raises(ValueError):
            ClusteringConfig(centroid_learning_rate=0.0)

        with pytest.raises(ValueError):
            ClusteringConfig(centroid_learning_rate=1.1)

    def test_umap_n_components_bounds(self):
        """UMAP n_components should be >= 2."""
        config = ClusteringConfig(umap_n_components=2)
        assert config.umap_n_components == 2

        with pytest.raises(ValueError):
            ClusteringConfig(umap_n_components=1)

    def test_nr_topics_optional(self):
        """nr_topics should be None (auto) by default, but accept integers >= 2."""
        config = ClusteringConfig()
        assert config.nr_topics is None

        config = ClusteringConfig(nr_topics=5)
        assert config.nr_topics == 5

        with pytest.raises(ValueError):
            ClusteringConfig(nr_topics=1)

    def test_package_import(self):
        """Should be importable from the clustering package."""
        from src.clustering import ClusteringConfig as CC

        assert CC is ClusteringConfig
