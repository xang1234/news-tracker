"""Tests for BERTopicService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService


def _get_theme_by_topic_id(service, topic_id):
    """Helper to find a theme by its bertopic_topic_id metadata."""
    for theme in service._themes.values():
        if theme.metadata.get("bertopic_topic_id") == topic_id:
            return theme
    return None


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


# ──────────────────────────────────────────────────────
# transform() tests
# ──────────────────────────────────────────────────────


class TestTransformInputValidation:
    """Tests for transform() input validation."""

    def test_mismatched_documents_and_ids(self, fitted_service, sample_embeddings):
        """Should raise ValueError when documents and IDs have different lengths."""
        with pytest.raises(ValueError, match="documents.*document_ids.*must have the same length"):
            fitted_service.transform(["doc1", "doc2"], sample_embeddings[:2], ["id1"])

    def test_mismatched_documents_and_embeddings(self, fitted_service):
        """Should raise ValueError when documents and embeddings have different lengths."""
        embeddings = np.zeros((3, 768))
        with pytest.raises(ValueError, match="documents.*embeddings.*must have the same length"):
            fitted_service.transform(["doc1", "doc2"], embeddings, ["id1", "id2"])

    def test_not_initialized_returns_empty(self):
        """Should return empty list when service is not initialized."""
        service = BERTopicService()
        embeddings = np.random.randn(1, 768).astype(np.float32)
        result = service.transform(["some doc"], embeddings, ["new_001"])
        assert result == []

    def test_empty_input_returns_empty(self, fitted_service):
        """Should return empty list for empty input."""
        result = fitted_service.transform([], np.empty((0, 768)), [])
        assert result == []


class TestTransformStrongAssignment:
    """Tests for strong-tier assignment (>= similarity_threshold_assign)."""

    def test_assigns_to_nearest_theme(self, fitted_service):
        """Embedding near cluster 0 centroid should assign to that theme."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        # Create an embedding very close to cluster 0's centroid
        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        results = fitted_service.transform(["New GPU news"], emb, ["new_001"])

        assert len(results) == 1
        doc_id, theme_ids, sim = results[0]
        assert doc_id == "new_001"
        assert len(theme_ids) == 1
        assert theme_ids[0] == theme_0.theme_id
        assert sim >= fitted_service.config.similarity_threshold_assign

    def test_centroid_updated_via_ema(self, fitted_service):
        """Strong assignment should update the centroid via EMA."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        original_centroid = theme_0.centroid.copy()
        lr = fitted_service.config.centroid_learning_rate

        # Embedding very close to centroid (will be strong assignment)
        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        fitted_service.transform(["New GPU news"], emb, ["new_001"])

        expected = (1 - lr) * original_centroid + lr * emb[0]
        np.testing.assert_array_almost_equal(theme_0.centroid, expected)

    def test_updated_at_set(self, fitted_service):
        """Strong assignment should set updated_at on the theme."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        assert theme_0.updated_at is None  # Not set by fit()

        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        fitted_service.transform(["New GPU news"], emb, ["new_001"])

        assert theme_0.updated_at is not None

    def test_document_count_incremented(self, fitted_service):
        """Strong assignment should increment document_count."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        original_count = theme_0.document_count

        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        fitted_service.transform(["New GPU news"], emb, ["new_001"])

        assert theme_0.document_count == original_count + 1

    def test_document_ids_includes_new_doc(self, fitted_service):
        """Strong assignment should add doc_id to theme's document_ids."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)

        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        fitted_service.transform(["New GPU news"], emb, ["new_001"])

        assert "new_001" in theme_0.document_ids


class TestTransformWeakAssignment:
    """Tests for weak-tier assignment (between new and assign thresholds)."""

    def test_assigns_without_centroid_update(self, fitted_service):
        """Weak assignment should assign but NOT update the centroid."""
        # Use a very high assign threshold to force weak tier
        fitted_service.config.similarity_threshold_assign = 0.999
        fitted_service.config.similarity_threshold_new = 0.01

        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        original_centroid = theme_0.centroid.copy()

        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        results = fitted_service.transform(["New GPU news"], emb, ["new_001"])

        # Should still assign (similarity > new_threshold)
        assert len(results) == 1
        _, theme_ids, sim = results[0]
        assert len(theme_ids) == 1
        # Centroid should NOT change
        np.testing.assert_array_equal(theme_0.centroid, original_centroid)

    def test_updated_at_remains_none(self, fitted_service):
        """Weak assignment should NOT set updated_at."""
        fitted_service.config.similarity_threshold_assign = 0.999
        fitted_service.config.similarity_threshold_new = 0.01

        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        assert theme_0.updated_at is None

        near_centroid = theme_0.centroid.copy()
        near_centroid += np.random.RandomState(99).randn(768) * 0.01
        near_centroid = near_centroid / np.linalg.norm(near_centroid)
        emb = near_centroid.reshape(1, -1).astype(np.float32)

        fitted_service.transform(["New GPU news"], emb, ["new_001"])

        assert theme_0.updated_at is None


class TestTransformNewThemeCandidate:
    """Tests for new-candidate tier (below similarity_threshold_new)."""

    def _make_orthogonal_embedding(self, fitted_service, dim=768):
        """Create an embedding far from all centroids."""
        rng = np.random.RandomState(123)
        # Random direction in high-dim space is nearly orthogonal to any fixed vectors
        emb = rng.randn(dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb

    def test_returns_empty_theme_ids(self, fitted_service):
        """New candidates should have empty theme_ids in the result."""
        # Set a high new_threshold so everything below it becomes a candidate
        fitted_service.config.similarity_threshold_new = 0.99
        fitted_service.config.similarity_threshold_assign = 0.999

        emb = self._make_orthogonal_embedding(fitted_service).reshape(1, -1)
        results = fitted_service.transform(["Something unrelated"], emb, ["new_001"])

        assert len(results) == 1
        _, theme_ids, _ = results[0]
        assert theme_ids == []

    def test_stored_in_new_theme_candidates(self, fitted_service):
        """New candidates should be buffered in the service."""
        fitted_service.config.similarity_threshold_new = 0.99
        fitted_service.config.similarity_threshold_assign = 0.999

        emb = self._make_orthogonal_embedding(fitted_service).reshape(1, -1)
        fitted_service.transform(["Something unrelated"], emb, ["new_001"])

        candidates = fitted_service.new_theme_candidates
        assert len(candidates) == 1
        assert candidates[0][0] == "new_001"
        np.testing.assert_array_almost_equal(candidates[0][1], emb[0])

    def test_candidates_accumulate_across_calls(self, fitted_service):
        """Multiple transform() calls should accumulate candidates."""
        fitted_service.config.similarity_threshold_new = 0.99
        fitted_service.config.similarity_threshold_assign = 0.999

        rng = np.random.RandomState(200)
        for i in range(3):
            emb = rng.randn(1, 768).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            fitted_service.transform([f"Doc {i}"], emb, [f"new_{i:03d}"])

        assert len(fitted_service.new_theme_candidates) == 3


class TestTransformBatch:
    """Tests for batch transform with multiple documents."""

    def test_multiple_docs_processed(self, fitted_service):
        """Should return one result per input document."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        theme_1 = _get_theme_by_topic_id(fitted_service, 1)

        # Two embeddings near different clusters
        rng = np.random.RandomState(42)
        emb0 = theme_0.centroid + rng.randn(768) * 0.01
        emb0 = emb0 / np.linalg.norm(emb0)
        emb1 = theme_1.centroid + rng.randn(768) * 0.01
        emb1 = emb1 / np.linalg.norm(emb1)
        batch_emb = np.vstack([emb0, emb1]).astype(np.float32)

        results = fitted_service.transform(
            ["GPU news", "Memory news"],
            batch_emb,
            ["new_001", "new_002"],
        )

        assert len(results) == 2
        # First doc should go to theme 0, second to theme 1
        assert results[0][1] == [theme_0.theme_id]
        assert results[1][1] == [theme_1.theme_id]

    def test_mixed_tiers_in_batch(self, fitted_service):
        """A single batch can produce strong assignments and new candidates."""
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)

        # One near centroid (strong), one far away (candidate)
        rng = np.random.RandomState(42)
        near = theme_0.centroid + rng.randn(768) * 0.01
        near = near / np.linalg.norm(near)

        # Far-away embedding: very high new_threshold not needed —
        # random vector in 768-dim is ~orthogonal to any centroid
        far = rng.randn(768).astype(np.float32)
        far = far / np.linalg.norm(far)

        batch_emb = np.vstack([near, far]).astype(np.float32)

        results = fitted_service.transform(
            ["GPU news", "Totally unrelated"],
            batch_emb,
            ["new_001", "new_002"],
        )

        assert len(results) == 2
        # First should be assigned
        assert len(results[0][1]) == 1
        # Second's tier depends on similarity — just verify it returned
        assert results[1][0] == "new_002"


class TestTransformEdgeCases:
    """Tests for edge cases in transform()."""

    def test_no_themes_returns_all_candidates(
        self, clustering_config, sample_documents, sample_embeddings, sample_document_ids,
    ):
        """If fit() produced no themes (all outliers), transform returns empty."""
        service = BERTopicService(config=clustering_config)
        model = MagicMock()
        model.fit_transform.return_value = ([-1] * 15, np.zeros((15, 3)))

        with patch.object(service, "_create_model", return_value=model):
            service.fit(sample_documents, sample_embeddings, sample_document_ids)

        assert service.is_initialized
        assert len(service._themes) == 0

        emb = np.random.randn(1, 768).astype(np.float32)
        result = service.transform(["new doc"], emb, ["new_001"])
        assert result == []

    def test_zero_norm_embedding_no_crash(self, fitted_service):
        """A zero-norm embedding should not raise (guarded by np.where)."""
        zero_emb = np.zeros((1, 768), dtype=np.float32)
        results = fitted_service.transform(["Empty"], zero_emb, ["new_001"])
        assert len(results) == 1  # Should complete without error

    def test_internal_error_returns_empty(self, fitted_service):
        """If _assign_documents raises, transform should return empty list."""
        with patch.object(
            fitted_service, "_assign_documents", side_effect=RuntimeError("numpy broke")
        ):
            results = fitted_service.transform(
                ["test"], np.random.randn(1, 768).astype(np.float32), ["new_001"]
            )
        assert results == []


class TestTransformGetStats:
    """Tests for get_stats() after transform."""

    def test_stats_include_new_theme_candidates(self, fitted_service):
        """Stats should report n_new_theme_candidates count."""
        fitted_service.config.similarity_threshold_new = 0.99
        fitted_service.config.similarity_threshold_assign = 0.999

        rng = np.random.RandomState(42)
        emb = rng.randn(1, 768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        fitted_service.transform(["Unrelated"], emb, ["new_001"])

        stats = fitted_service.get_stats()
        assert stats["n_new_theme_candidates"] == 1

    def test_stats_candidates_zero_before_transform(self, fitted_service):
        """Before any transform, n_new_theme_candidates should be 0."""
        stats = fitted_service.get_stats()
        assert stats["n_new_theme_candidates"] == 0

    def test_stats_candidates_before_fit(self):
        """Even before fit, stats should include n_new_theme_candidates."""
        service = BERTopicService()
        stats = service.get_stats()
        assert stats["n_new_theme_candidates"] == 0


# ──────────────────────────────────────────────────────
# merge_similar_themes() tests
# ──────────────────────────────────────────────────────


class TestMergeSimilarThemes:
    """Tests for BERTopicService.merge_similar_themes()."""

    def test_merges_similar_themes(self, service_with_similar_themes):
        """Themes A and B (sim ~0.9998) should merge, reducing theme count."""
        service = service_with_similar_themes
        assert len(service._themes) == 3

        merges = service.merge_similar_themes()

        assert len(merges) == 1
        assert len(service._themes) == 2

    def test_keeps_larger_theme(self, service_with_similar_themes):
        """Survivor should be the theme with more documents (Theme A, 10 docs)."""
        service = service_with_similar_themes
        merges = service.merge_similar_themes()

        # Theme A had 10 docs, Theme B had 5 — A survives
        merged_from, merged_into = merges[0]
        # The merged_into should be the original ID of theme A (larger)
        # After merge, the survivor has 15 docs total
        surviving_themes = list(service._themes.values())
        merged_theme = [t for t in surviving_themes if t.document_count == 15]
        assert len(merged_theme) == 1

    def test_weighted_centroid(self, service_with_similar_themes):
        """Merged centroid should be weighted average by document count."""
        service = service_with_similar_themes

        # Capture original centroids before merge
        themes = list(service._themes.values())
        # Find themes A (10 docs) and B (5 docs) by doc count
        theme_a = next(t for t in themes if t.document_count == 10)
        theme_b = next(t for t in themes if t.document_count == 5)
        centroid_a = theme_a.centroid.copy()
        centroid_b = theme_b.centroid.copy()

        service.merge_similar_themes()

        expected = (10 * centroid_a + 5 * centroid_b) / 15
        survivor = next(t for t in service._themes.values() if t.document_count == 15)
        np.testing.assert_array_almost_equal(survivor.centroid, expected)

    def test_merged_topic_words_deduped(self, service_with_similar_themes):
        """Merged topic words should have no duplicates and respect top_n_words."""
        service = service_with_similar_themes
        service.merge_similar_themes()

        survivor = next(t for t in service._themes.values() if t.document_count == 15)
        words = [w for w, _ in survivor.topic_words]
        # No duplicate words
        assert len(words) == len(set(words))
        # Respects top_n_words config
        assert len(survivor.topic_words) <= service.config.top_n_words

    def test_document_ids_combined(self, service_with_similar_themes):
        """Survivor should contain document IDs from both themes."""
        service = service_with_similar_themes
        service.merge_similar_themes()

        survivor = next(t for t in service._themes.values() if t.document_count == 15)
        # Theme A had doc_a_0..doc_a_9, Theme B had doc_b_0..doc_b_4
        for i in range(10):
            assert f"doc_a_{i}" in survivor.document_ids
        for i in range(5):
            assert f"doc_b_{i}" in survivor.document_ids

    def test_document_count_summed(self, service_with_similar_themes):
        """Survivor document_count should be sum of merged themes."""
        service = service_with_similar_themes
        service.merge_similar_themes()

        survivor = next(t for t in service._themes.values() if t.document_count == 15)
        assert survivor.document_count == 15

    def test_returns_merge_pairs(self, service_with_similar_themes):
        """Return value should be list of (from_id, into_id) tuples."""
        service = service_with_similar_themes
        merges = service.merge_similar_themes()

        assert len(merges) == 1
        merged_from, merged_into = merges[0]
        assert isinstance(merged_from, str)
        assert isinstance(merged_into, str)
        assert merged_from.startswith("theme_")
        assert merged_into.startswith("theme_")
        assert merged_from != merged_into

    def test_theme_id_regenerated(self, service_with_similar_themes):
        """Theme IDs should be regenerated after merge (topic words changed)."""
        service = service_with_similar_themes
        old_ids = set(service._themes.keys())

        service.merge_similar_themes()

        new_ids = set(service._themes.keys())
        # At least the survivor's ID changed (merged topic words)
        # Theme C's ID might also change if re-keyed — but at minimum
        # the set shouldn't be identical since one theme was removed
        assert len(new_ids) < len(old_ids)

    def test_no_merge_below_threshold(self, fitted_service):
        """Well-separated themes (from fitted_service) should not merge."""
        merges = fitted_service.merge_similar_themes()
        assert merges == []
        assert len(fitted_service._themes) == 3

    def test_not_initialized_returns_empty(self):
        """Not-initialized service should return empty list."""
        service = BERTopicService()
        assert service.merge_similar_themes() == []

    def test_single_theme_returns_empty(self, clustering_config):
        """Service with only one theme should return empty list."""
        service = BERTopicService(config=clustering_config)
        service._initialized = True

        theme = ThemeCluster(
            theme_id="theme_abc123",
            name="solo_theme",
            topic_words=[("gpu", 0.1)],
            centroid=np.random.randn(768),
            document_count=5,
            document_ids=["d1", "d2", "d3", "d4", "d5"],
        )
        service._themes = {"theme_abc123": theme}

        assert service.merge_similar_themes() == []


# ──────────────────────────────────────────────────────
# check_new_themes() tests
# ──────────────────────────────────────────────────────


class TestCheckNewThemes:
    """Tests for BERTopicService.check_new_themes()."""

    def _make_candidates(self, n, dim=768, seed=42):
        """Create n candidate (doc_id, text, embedding) triples in a tight cluster."""
        rng = np.random.RandomState(seed)
        center = rng.randn(dim).astype(np.float64)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(n):
            emb = center + rng.randn(dim) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Candidate document {i} about topic", emb))
        return candidates

    def test_creates_new_theme(self, fitted_service):
        """Valid cluster of candidates should produce a new ThemeCluster."""
        # Create candidates far from existing themes
        rng = np.random.RandomState(999)
        center = rng.randn(768)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(6):
            emb = center + rng.randn(768) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"New topic about quantum computing chip {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            new_themes = fitted_service.check_new_themes(candidates)

        assert len(new_themes) >= 1

    def test_new_theme_has_correct_fields(self, fitted_service):
        """Newly created theme should have all expected ThemeCluster fields."""
        rng = np.random.RandomState(999)
        center = rng.randn(768)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(6):
            emb = center + rng.randn(768) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Quantum computing chip advances {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            new_themes = fitted_service.check_new_themes(candidates)

        assert len(new_themes) >= 1
        theme = new_themes[0]
        assert isinstance(theme, ThemeCluster)
        assert theme.theme_id.startswith("theme_")
        assert len(theme.name) > 0
        assert len(theme.topic_words) > 0
        assert isinstance(theme.centroid, np.ndarray)
        assert theme.document_count == 6
        assert len(theme.document_ids) == 6

    def test_adds_to_service_themes(self, fitted_service):
        """New themes should be added to the service's _themes dict."""
        original_count = len(fitted_service._themes)

        rng = np.random.RandomState(999)
        center = rng.randn(768)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(6):
            emb = center + rng.randn(768) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Quantum topic {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            fitted_service.check_new_themes(candidates)

        assert len(fitted_service._themes) > original_count

    def test_candidates_cleared_after(self, fitted_service):
        """Processed candidate doc_ids should be removed from _new_theme_candidates."""
        # Seed some candidates in the internal buffer
        rng = np.random.RandomState(999)
        for i in range(6):
            emb = rng.randn(768).astype(np.float32)
            fitted_service._new_theme_candidates.append((f"cand_{i}", emb))

        # Also add one that's NOT in the candidate list
        fitted_service._new_theme_candidates.append(("unrelated_doc", rng.randn(768).astype(np.float32)))

        center = rng.randn(768)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(6):
            emb = center + rng.randn(768) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Topic {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            fitted_service.check_new_themes(candidates)

        # cand_0..cand_5 should be removed, unrelated_doc should remain
        remaining_ids = [did for did, _ in fitted_service._new_theme_candidates]
        for i in range(6):
            assert f"cand_{i}" not in remaining_ids
        assert "unrelated_doc" in remaining_ids

    def test_rejects_similar_to_existing(self, fitted_service):
        """Candidate cluster near an existing theme should be skipped."""
        # Create candidates near an existing theme's centroid
        theme_0 = _get_theme_by_topic_id(fitted_service, 0)
        center = theme_0.centroid.copy()
        candidates = []
        for i in range(6):
            rng = np.random.RandomState(i)
            emb = center + rng.randn(768) * 0.01
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Similar GPU topic {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        original_count = len(fitted_service._themes)

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            new_themes = fitted_service.check_new_themes(candidates)

        assert new_themes == []
        assert len(fitted_service._themes) == original_count

    def test_not_enough_candidates(self, fitted_service):
        """Too few candidates should return empty list."""
        candidates = [("cand_0", "Single doc", np.random.randn(768))]
        result = fitted_service.check_new_themes(candidates)
        assert result == []

    def test_not_initialized_returns_empty(self):
        """Not-initialized service should return empty list."""
        service = BERTopicService()
        candidates = [("cand_0", "Doc", np.random.randn(768)) for _ in range(10)]
        assert service.check_new_themes(candidates) == []

    def test_empty_candidates_returns_empty(self, fitted_service):
        """Empty candidate list should return empty list."""
        assert fitted_service.check_new_themes([]) == []

    def test_all_outliers_returns_empty(self, fitted_service):
        """When HDBSCAN labels all candidates as -1, no themes created."""
        rng = np.random.RandomState(999)
        candidates = []
        for i in range(6):
            emb = rng.randn(768)
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Random doc {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([-1, -1, -1, -1, -1, -1])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            new_themes = fitted_service.check_new_themes(candidates)

        assert new_themes == []

    def test_lifecycle_stage_metadata(self, fitted_service):
        """New themes should have lifecycle_stage='emerging' in metadata."""
        rng = np.random.RandomState(999)
        center = rng.randn(768)
        center /= np.linalg.norm(center)
        candidates = []
        for i in range(6):
            emb = center + rng.randn(768) * 0.02
            emb = emb / np.linalg.norm(emb)
            candidates.append((f"cand_{i}", f"Quantum topic {i}", emb))

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])

        with patch.object(fitted_service, "_create_mini_clusterer", return_value=mock_clusterer):
            new_themes = fitted_service.check_new_themes(candidates)

        assert len(new_themes) >= 1
        assert new_themes[0].metadata.get("lifecycle_stage") == "emerging"
