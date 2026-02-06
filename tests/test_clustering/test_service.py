"""Tests for BERTopicService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService


class TestServiceInit:
    """Tests for BERTopicService initialization."""

    def test_default_config(self):
        """Should use default config when none provided."""
        service = BERTopicService()
        assert isinstance(service.config, ClusteringConfig)

    def test_custom_config(self, clustering_config):
        """Should accept custom config."""
        service = BERTopicService(config=clustering_config)
        assert service.config.hdbscan_min_cluster_size == 3

    def test_not_initialized_before_fit(self):
        """Should not be initialized before fit() is called."""
        service = BERTopicService()
        assert not service.is_initialized

    def test_empty_themes_before_fit(self):
        """Should have empty themes before fit() is called."""
        service = BERTopicService()
        assert service.themes == {}

    def test_model_none_before_fit(self):
        """Model should be None before fit() is called."""
        service = BERTopicService()
        assert service.model is None


class TestFit:
    """Tests for BERTopicService.fit() with mocked BERTopic model."""

    def test_themes_discovered(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """fit() should discover non-outlier themes from BERTopic results."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        # 3 topics (0, 1, 2), outlier -1 excluded
        assert len(themes) == 3

    def test_is_initialized_after_fit(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Service should be initialized after successful fit()."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert service.is_initialized

    def test_model_stored_after_fit(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """The BERTopic model should be accessible after fit()."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert service.model is mock_bertopic_model

    def test_theme_has_correct_fields(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Each theme should have all expected ThemeCluster fields."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        for theme in themes.values():
            assert isinstance(theme, ThemeCluster)
            assert theme.theme_id.startswith("theme_")
            assert len(theme.name) > 0
            assert len(theme.topic_words) > 0
            assert isinstance(theme.centroid, np.ndarray)
            assert theme.document_count > 0
            assert len(theme.document_ids) == theme.document_count
            assert "bertopic_topic_id" in theme.metadata

    def test_centroid_computed_correctly(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Centroid should be the mean of document embeddings in the cluster."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        # Topic 0 has indices 0-4
        expected_centroid = np.mean(sample_embeddings[0:5], axis=0)

        # Find the theme with bertopic_topic_id == 0
        topic_0_theme = None
        for theme in themes.values():
            if theme.metadata.get("bertopic_topic_id") == 0:
                topic_0_theme = theme
                break

        assert topic_0_theme is not None
        np.testing.assert_array_almost_equal(topic_0_theme.centroid, expected_centroid)

    def test_outlier_documents_excluded(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Documents with topic -1 should not appear in any theme."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        # doc_014 is the outlier (index 14, topic -1)
        all_assigned_ids = []
        for theme in themes.values():
            all_assigned_ids.extend(theme.document_ids)

        assert "doc_014" not in all_assigned_ids
        # 14 docs assigned (15 total - 1 outlier)
        assert len(all_assigned_ids) == 14

    def test_document_ids_assigned_correctly(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Document IDs should be correctly mapped to their themes."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        # Topic 0 should have doc_000 through doc_004
        topic_0_theme = None
        for theme in themes.values():
            if theme.metadata.get("bertopic_topic_id") == 0:
                topic_0_theme = theme
                break

        assert topic_0_theme is not None
        assert set(topic_0_theme.document_ids) == {"doc_000", "doc_001", "doc_002", "doc_003", "doc_004"}

    def test_topic_words_from_model(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Topic words should come from model.get_topic()."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        # Topic 1 should have memory-related words
        topic_1_theme = None
        for theme in themes.values():
            if theme.metadata.get("bertopic_topic_id") == 1:
                topic_1_theme = theme
                break

        assert topic_1_theme is not None
        words = [w for w, _ in topic_1_theme.topic_words]
        assert "memory" in words
        assert "hbm3e" in words

    def test_fit_transform_called_with_embeddings(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """fit_transform should be called with documents and pre-computed embeddings."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        mock_bertopic_model.fit_transform.assert_called_once()
        call_args = mock_bertopic_model.fit_transform.call_args
        assert call_args[0][0] == sample_documents
        np.testing.assert_array_equal(call_args[1]["embeddings"], sample_embeddings)


class TestFitInputValidation:
    """Tests for fit() input validation."""

    def test_mismatched_documents_and_ids(self, clustering_config, sample_embeddings):
        """Should raise ValueError when documents and IDs have different lengths."""
        service = BERTopicService(config=clustering_config)
        documents = ["doc1", "doc2"]
        ids = ["id1"]  # Mismatched length

        with pytest.raises(ValueError, match="documents.*document_ids.*must have the same length"):
            service.fit(documents, sample_embeddings[:2], ids)

    def test_mismatched_documents_and_embeddings(self, clustering_config):
        """Should raise ValueError when documents and embeddings have different lengths."""
        service = BERTopicService(config=clustering_config)
        documents = ["doc1", "doc2"]
        ids = ["id1", "id2"]
        embeddings = np.zeros((3, 768))  # 3 embeddings for 2 docs

        with pytest.raises(ValueError, match="documents.*embeddings.*must have the same length"):
            service.fit(documents, embeddings, ids)

    def test_empty_input_returns_empty(self, clustering_config):
        """Should return empty dict for empty input."""
        service = BERTopicService(config=clustering_config)
        result = service.fit([], np.empty((0, 768)), [])
        assert result == {}


class TestFitEdgeCases:
    """Tests for edge cases in fit()."""

    def test_all_outliers(self, clustering_config, sample_documents, sample_embeddings, sample_document_ids):
        """When BERTopic assigns all docs to topic -1, should return no themes."""
        service = BERTopicService(config=clustering_config)
        model = MagicMock()
        model.fit_transform.return_value = ([-1] * 15, np.zeros((15, 3)))

        with patch.object(service, "_create_model", return_value=model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert themes == {}
        assert service.is_initialized

    def test_single_cluster(self, clustering_config, sample_documents, sample_embeddings, sample_document_ids):
        """Should handle a single non-outlier cluster."""
        service = BERTopicService(config=clustering_config)
        model = MagicMock()
        # All 15 docs in topic 0
        model.fit_transform.return_value = ([0] * 15, np.random.rand(15, 3))
        model.get_topic.return_value = [("semiconductor", 0.2), ("chip", 0.15)]

        with patch.object(service, "_create_model", return_value=model):
            themes = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert len(themes) == 1
        theme = list(themes.values())[0]
        assert theme.document_count == 15

    def test_internal_error_returns_empty(self, clustering_config, sample_documents, sample_embeddings, sample_document_ids):
        """Internal BERTopic error should return empty dict, not raise."""
        service = BERTopicService(config=clustering_config)
        model = MagicMock()
        model.fit_transform.side_effect = RuntimeError("UMAP failed")

        with patch.object(service, "_create_model", return_value=model):
            result = service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert result == {}
        assert not service.is_initialized

    def test_themes_property_returns_copy(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """themes property should return a copy, not the internal dict."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        themes_a = service.themes
        themes_b = service.themes
        assert themes_a is not themes_b
        assert themes_a == themes_b


class TestGetStats:
    """Tests for get_stats()."""

    def test_stats_before_fit(self):
        """Should return uninitialized stats before fit()."""
        service = BERTopicService()
        stats = service.get_stats()

        assert stats["initialized"] is False
        assert stats["n_themes"] == 0
        assert stats["n_documents"] == 0

    def test_stats_after_fit(
        self,
        clustering_config,
        sample_documents,
        sample_embeddings,
        sample_document_ids,
        mock_bertopic_model,
    ):
        """Should return correct stats after fit()."""
        service = BERTopicService(config=clustering_config)

        with patch.object(service, "_create_model", return_value=mock_bertopic_model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        stats = service.get_stats()
        assert stats["initialized"] is True
        assert stats["n_themes"] == 3
        # 14 docs assigned (15 - 1 outlier)
        assert stats["n_documents"] == 14
        assert len(stats["themes"]) == 3

        # Each theme stat should have expected fields
        for theme_stat in stats["themes"]:
            assert "theme_id" in theme_stat
            assert "name" in theme_stat
            assert "document_count" in theme_stat
            assert "top_words" in theme_stat
            assert len(theme_stat["top_words"]) <= 5
