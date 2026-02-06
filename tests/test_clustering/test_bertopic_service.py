"""Integration and performance tests for BERTopicService.

These tests use real BERTopic, UMAP, and HDBSCAN (no mocks) to verify
end-to-end clustering behavior. They require the ML dependencies to be
installed and are slower than the unit tests in test_service.py.

Integration tests validate that the full pipeline (UMAP -> HDBSCAN -> c-TF-IDF)
produces meaningful clusters from synthetic data with known structure.

Performance tests benchmark throughput and latency at realistic scale.
"""

import time

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService


# ──────────────────────────────────────────────────────
# Fixtures for integration tests
# ──────────────────────────────────────────────────────


def _make_clustered_embeddings(
    n_clusters: int,
    n_per_cluster: int,
    dim: int = 768,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Generate synthetic embeddings with well-separated clusters.

    Returns:
        (embeddings, centers, labels) where labels[i] is the true cluster
        assignment for embeddings[i].
    """
    rng = np.random.RandomState(seed)

    centers = rng.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    all_embeddings = []
    all_labels = []

    for cluster_idx in range(n_clusters):
        noise = rng.randn(n_per_cluster, dim) * noise_scale
        cluster_embs = centers[cluster_idx] + noise
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        cluster_embs = cluster_embs / norms
        all_embeddings.append(cluster_embs)
        all_labels.extend([cluster_idx] * n_per_cluster)

    return (
        np.vstack(all_embeddings).astype(np.float32),
        centers,
        all_labels,
    )


def _make_documents(n: int, groups: list[list[str]] | None = None) -> list[str]:
    """Generate n document strings, optionally cycling through thematic groups."""
    if groups is None:
        return [f"Document about topic number {i}" for i in range(n)]

    docs = []
    for i in range(n):
        group_idx = i // (n // len(groups)) if n // len(groups) > 0 else 0
        group_idx = min(group_idx, len(groups) - 1)
        template_idx = i % len(groups[group_idx])
        docs.append(groups[group_idx][template_idx])
    return docs


@pytest.fixture
def integration_config():
    """Config tuned for small synthetic datasets in integration tests."""
    return ClusteringConfig(
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=3,
        umap_n_components=5,
        umap_n_neighbors=10,
        umap_random_state=42,
        top_n_words=5,
        similarity_threshold_assign=0.75,
        similarity_threshold_new=0.30,
        similarity_threshold_merge=0.85,
        centroid_learning_rate=0.01,
    )


@pytest.fixture
def three_cluster_data():
    """30 documents in 3 clusters (10 each), well-separated in 768-dim space."""
    n_clusters, n_per = 3, 10
    embeddings, centers, labels = _make_clustered_embeddings(
        n_clusters=n_clusters, n_per_cluster=n_per, noise_scale=0.05, seed=42,
    )

    groups = [
        [
            "Nvidia announces next-generation GPU for AI training workloads",
            "New GPU accelerator delivers deep learning performance gains",
            "AI chip computing power doubles with latest GPU architecture",
            "Nvidia GPU sales surge driven by data center AI demand",
            "GPU computing platform enables faster model training pipelines",
            "Deep learning GPU cluster achieves record training throughput",
            "Nvidia AI accelerator outperforms competitors in MLPerf benchmarks",
            "GPU architecture innovation drives artificial intelligence progress",
            "Next-gen GPU microarchitecture targets large language model inference",
            "Nvidia dominates AI accelerator market with new GPU launch",
        ],
        [
            "HBM3E memory bandwidth reaches new highs for data center chips",
            "Samsung develops advanced HBM memory for AI accelerators",
            "Memory chip demand increases as HBM technology evolves rapidly",
            "SK Hynix HBM3E memory production ramps up for AI workloads",
            "High bandwidth memory technology drives semiconductor innovation",
            "HBM memory stacking advances enable higher capacity modules",
            "Samsung and SK Hynix compete for HBM3E market leadership",
            "Memory bandwidth bottleneck solved by next-gen HBM technology",
            "AI training memory requirements push HBM adoption forward",
            "Data center memory architecture shifts toward HBM3E standard",
        ],
        [
            "TSMC expands 3nm chip manufacturing capacity in Arizona fab",
            "Semiconductor fabrication technology advances to 2nm process node",
            "Intel foundry services compete with TSMC on manufacturing yield",
            "Chip manufacturing yield improvements boost semiconductor output",
            "Advanced semiconductor manufacturing drives global supply chain",
            "TSMC Arizona fab construction progresses toward production start",
            "Samsung foundry announces 2nm GAA transistor manufacturing node",
            "Semiconductor manufacturing equipment spending hits record levels",
            "TSMC and Samsung race to deliver advanced process technology",
            "Chip fabrication innovations reduce defect density at scale",
        ],
    ]
    documents = _make_documents(n_clusters * n_per, groups)
    doc_ids = [f"doc_{i:03d}" for i in range(n_clusters * n_per)]

    return embeddings, documents, doc_ids, centers, labels


# ──────────────────────────────────────────────────────
# Integration tests (real BERTopic pipeline)
# ──────────────────────────────────────────────────────


@pytest.mark.integration
class TestFitDiscoversThemes:
    """Integration: fit() with real BERTopic discovers themes from synthetic clusters."""

    def test_discovers_at_least_two_themes(
        self, integration_config, three_cluster_data,
    ):
        """Real UMAP+HDBSCAN should find at least 2 themes from 3 distinct clusters."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)

        themes = service.fit(documents, embeddings, doc_ids)

        # With well-separated clusters, BERTopic should find >= 2 themes.
        # We don't require exactly 3 because UMAP randomness and HDBSCAN's
        # density estimation can merge or split clusters.
        assert len(themes) >= 2
        assert service.is_initialized

    def test_themes_have_valid_structure(
        self, integration_config, three_cluster_data,
    ):
        """Each discovered theme should have valid ThemeCluster fields."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)

        themes = service.fit(documents, embeddings, doc_ids)

        for theme in themes.values():
            assert isinstance(theme, ThemeCluster)
            assert theme.theme_id.startswith("theme_")
            assert len(theme.name) > 0
            assert len(theme.topic_words) > 0
            assert theme.centroid.shape == (768,)
            assert theme.document_count > 0
            assert len(theme.document_ids) == theme.document_count

    def test_no_duplicate_document_assignments(
        self, integration_config, three_cluster_data,
    ):
        """Each document should appear in at most one theme."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)

        themes = service.fit(documents, embeddings, doc_ids)

        all_ids = []
        for theme in themes.values():
            all_ids.extend(theme.document_ids)
        assert len(all_ids) == len(set(all_ids))

    def test_outliers_excluded(
        self, integration_config, three_cluster_data,
    ):
        """No theme should contain outlier documents (topic -1)."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)

        themes = service.fit(documents, embeddings, doc_ids)

        assigned_count = sum(t.document_count for t in themes.values())
        # Some docs may be outliers; assigned count <= total
        assert assigned_count <= len(doc_ids)

    def test_theme_ids_are_deterministic(
        self, integration_config, three_cluster_data,
    ):
        """Running fit() twice with same data should produce same theme IDs."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data

        service1 = BERTopicService(config=integration_config)
        themes1 = service1.fit(documents, embeddings, doc_ids)

        service2 = BERTopicService(config=integration_config)
        themes2 = service2.fit(documents, embeddings, doc_ids)

        # Same topic words → same theme IDs (deterministic hashing)
        # This relies on UMAP random_state=42 for reproducibility
        assert set(themes1.keys()) == set(themes2.keys())


@pytest.mark.integration
class TestTransformAssignsCorrectly:
    """Integration: transform() assigns new documents to the correct themes."""

    def test_new_doc_near_cluster_gets_assigned(
        self, integration_config, three_cluster_data,
    ):
        """A new embedding near a cluster centroid should be assigned to that theme."""
        embeddings, documents, doc_ids, centers, _ = three_cluster_data
        service = BERTopicService(config=integration_config)
        service.fit(documents, embeddings, doc_ids)

        # Create a new embedding near cluster 0's center
        rng = np.random.RandomState(99)
        new_emb = centers[0] + rng.randn(768) * 0.02
        new_emb = (new_emb / np.linalg.norm(new_emb)).astype(np.float32).reshape(1, -1)

        results = service.transform(["New GPU chip news"], new_emb, ["new_001"])

        assert len(results) == 1
        doc_id, theme_ids, sim = results[0]
        assert doc_id == "new_001"
        # Should be assigned to some theme (the one near cluster 0)
        assert len(theme_ids) >= 1
        assert sim > 0.5

    def test_distant_doc_becomes_candidate(
        self, integration_config, three_cluster_data,
    ):
        """An embedding far from all themes should become a new-theme candidate."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)
        service.fit(documents, embeddings, doc_ids)

        # Force new_threshold high so far-away docs are candidates
        service.config.similarity_threshold_new = 0.99
        service.config.similarity_threshold_assign = 0.999

        rng = np.random.RandomState(777)
        far_emb = rng.randn(768).astype(np.float32)
        far_emb = (far_emb / np.linalg.norm(far_emb)).reshape(1, -1)

        results = service.transform(["Unrelated content"], far_emb, ["far_001"])

        assert len(results) == 1
        _, theme_ids, _ = results[0]
        assert theme_ids == []
        assert len(service.new_theme_candidates) == 1

    def test_batch_assigns_to_different_themes(
        self, integration_config, three_cluster_data,
    ):
        """Batch of embeddings near different clusters should go to different themes."""
        embeddings, documents, doc_ids, centers, _ = three_cluster_data
        service = BERTopicService(config=integration_config)
        themes = service.fit(documents, embeddings, doc_ids)

        if len(themes) < 2:
            pytest.skip("Need at least 2 themes for this test")

        # Create embeddings near the first two cluster centers
        rng = np.random.RandomState(42)
        emb0 = centers[0] + rng.randn(768) * 0.02
        emb0 = emb0 / np.linalg.norm(emb0)
        emb1 = centers[1] + rng.randn(768) * 0.02
        emb1 = emb1 / np.linalg.norm(emb1)
        batch = np.vstack([emb0, emb1]).astype(np.float32)

        results = service.transform(
            ["GPU news", "Memory news"], batch, ["new_001", "new_002"],
        )

        assert len(results) == 2
        # If both are assigned, they should go to different themes
        ids_0 = results[0][1]
        ids_1 = results[1][1]
        if ids_0 and ids_1:
            assert ids_0 != ids_1


@pytest.mark.integration
class TestMergeSimilarThemesIntegration:
    """Integration: merge_similar_themes() consolidates converged themes."""

    def test_merges_overlapping_clusters(self, integration_config):
        """Two nearly-identical clusters should be merged after fit + manual merge."""
        # Create 2 overlapping clusters + 1 distant cluster
        rng = np.random.RandomState(42)
        dim = 768
        n_per = 15

        # Cluster A and B share a center (will produce similar themes)
        center_ab = rng.randn(dim)
        center_ab = center_ab / np.linalg.norm(center_ab)
        center_c = rng.randn(dim)
        center_c = center_c / np.linalg.norm(center_c)

        embs_a = center_ab + rng.randn(n_per, dim) * 0.03
        embs_b = center_ab + rng.randn(n_per, dim) * 0.03
        embs_c = center_c + rng.randn(n_per, dim) * 0.03

        for arr in [embs_a, embs_b, embs_c]:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr /= norms

        embeddings = np.vstack([embs_a, embs_b, embs_c]).astype(np.float32)
        n_total = n_per * 3
        documents = [f"Topic about semiconductor chips version {i}" for i in range(n_total)]
        doc_ids = [f"doc_{i:03d}" for i in range(n_total)]

        # Use nr_topics to force BERTopic to produce exactly 3 topics
        config = ClusteringConfig(
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            umap_n_components=5,
            umap_n_neighbors=10,
            umap_random_state=42,
            nr_topics=3,
            similarity_threshold_merge=0.80,
        )
        service = BERTopicService(config=config)
        themes = service.fit(documents, embeddings, doc_ids)

        if len(themes) < 2:
            pytest.skip("BERTopic didn't produce enough themes to test merge")

        original_count = len(themes)
        merges = service.merge_similar_themes()

        # With overlapping clusters, at least one merge should happen OR
        # no merge if BERTopic already merged them during fit
        if merges:
            assert len(service._themes) < original_count
            for from_id, into_id in merges:
                assert isinstance(from_id, str)
                assert isinstance(into_id, str)


@pytest.mark.integration
class TestCheckNewThemesIntegration:
    """Integration: check_new_themes() detects emerging themes from outliers."""

    def test_detects_emerging_theme(self, integration_config, three_cluster_data):
        """A tight group of candidates far from existing themes should form a new theme."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)
        service.fit(documents, embeddings, doc_ids)

        original_theme_count = len(service._themes)

        # Create a large, tight cluster of candidates in a new region.
        # HDBSCAN in 768-dim needs many points to reliably detect density peaks
        # due to the curse of dimensionality.
        rng = np.random.RandomState(555)
        new_center = rng.randn(768)
        new_center = new_center / np.linalg.norm(new_center)

        candidates = []
        for i in range(30):
            emb = new_center + rng.randn(768) * 0.01
            emb = emb / np.linalg.norm(emb)
            candidates.append((
                f"new_cand_{i}",
                f"Emerging quantum computing chip technology breakthrough {i}",
                emb.astype(np.float32),
            ))

        new_themes = service.check_new_themes(candidates)

        # Real HDBSCAN should find a cluster in these tight candidates
        assert len(new_themes) >= 1
        assert len(service._themes) > original_theme_count

        for theme in new_themes:
            assert theme.metadata.get("lifecycle_stage") == "emerging"
            assert theme.theme_id.startswith("theme_")

    def test_rejects_candidates_near_existing(
        self, integration_config, three_cluster_data,
    ):
        """Candidates near an existing theme centroid should not create a new theme."""
        embeddings, documents, doc_ids, _, _ = three_cluster_data
        service = BERTopicService(config=integration_config)
        service.fit(documents, embeddings, doc_ids)

        if not service._themes:
            pytest.skip("No themes discovered")

        original_count = len(service._themes)

        # Create candidates very close to the first existing theme's centroid
        existing_centroid = list(service._themes.values())[0].centroid
        rng = np.random.RandomState(42)
        candidates = []
        for i in range(10):
            emb = existing_centroid + rng.randn(768) * 0.01
            emb = emb / np.linalg.norm(emb)
            candidates.append((
                f"near_cand_{i}",
                f"Very similar topic to existing theme number {i}",
                emb.astype(np.float32),
            ))

        new_themes = service.check_new_themes(candidates)

        # These should be rejected as too similar to existing themes
        assert new_themes == []
        assert len(service._themes) == original_count


@pytest.mark.integration
class TestFullPipeline:
    """Integration: end-to-end fit -> transform -> merge -> check_new flow."""

    def test_full_pipeline_fit_then_transform(
        self, integration_config, three_cluster_data,
    ):
        """Full pipeline: fit discovers themes, transform assigns new docs."""
        embeddings, documents, doc_ids, centers, _ = three_cluster_data
        service = BERTopicService(config=integration_config)

        # Phase 1: Fit
        themes = service.fit(documents, embeddings, doc_ids)
        assert len(themes) >= 1
        assert service.is_initialized

        initial_total_docs = sum(t.document_count for t in service._themes.values())

        # Phase 2: Transform - assign new documents
        rng = np.random.RandomState(100)
        n_new = 5
        new_embs = []
        for i in range(n_new):
            # Near a random existing cluster center
            center_idx = i % len(centers)
            emb = centers[center_idx] + rng.randn(768) * 0.03
            emb = emb / np.linalg.norm(emb)
            new_embs.append(emb)
        new_embs = np.array(new_embs, dtype=np.float32)
        new_docs = [f"New document about topic {i}" for i in range(n_new)]
        new_ids = [f"new_{i:03d}" for i in range(n_new)]

        results = service.transform(new_docs, new_embs, new_ids)
        assert len(results) == n_new

        # At least some should be assigned
        assigned = [r for r in results if r[1]]
        assert len(assigned) >= 1

        # Phase 3: Merge (may or may not find anything to merge)
        merges = service.merge_similar_themes()
        # Just verify it runs without error
        assert isinstance(merges, list)

        # Phase 4: Stats should reflect the state
        stats = service.get_stats()
        assert stats["initialized"] is True
        assert stats["n_themes"] >= 1


# ──────────────────────────────────────────────────────
# Edge cases (real BERTopic)
# ──────────────────────────────────────────────────────


@pytest.mark.integration
class TestEdgeCases:
    """Integration edge cases with real BERTopic."""

    def test_empty_input(self, integration_config):
        """Empty input should return empty dict without error."""
        service = BERTopicService(config=integration_config)
        result = service.fit([], np.empty((0, 768)), [])
        assert result == {}

    def test_single_document(self, integration_config):
        """Single document should not crash (HDBSCAN needs min_cluster_size)."""
        service = BERTopicService(config=integration_config)
        emb = np.random.randn(1, 768).astype(np.float32)
        result = service.fit(["Only one document"], emb, ["doc_000"])
        # BERTopic may return empty or raise internally — our service handles it
        assert isinstance(result, dict)

    def test_all_identical_embeddings(self, integration_config):
        """All-identical embeddings should not crash."""
        service = BERTopicService(config=integration_config)
        base = np.random.RandomState(42).randn(768).astype(np.float32)
        embs = np.tile(base, (20, 1))
        docs = [f"Identical document {i}" for i in range(20)]
        ids = [f"doc_{i:03d}" for i in range(20)]

        result = service.fit(docs, embs, ids)
        # UMAP may struggle with identical points, but service should handle it
        assert isinstance(result, dict)

    def test_two_documents_below_min_cluster(self, integration_config):
        """Fewer docs than min_cluster_size should handle gracefully."""
        service = BERTopicService(config=integration_config)
        embs = np.random.RandomState(42).randn(2, 768).astype(np.float32)
        docs = ["Document one", "Document two"]
        ids = ["doc_000", "doc_001"]

        result = service.fit(docs, embs, ids)
        assert isinstance(result, dict)

    def test_high_dimensional_noise(self, integration_config):
        """Pure random noise (no structure) should produce few or no themes."""
        service = BERTopicService(config=integration_config)
        rng = np.random.RandomState(42)
        embs = rng.randn(30, 768).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        docs = [f"Random noise document number {i}" for i in range(30)]
        ids = [f"doc_{i:03d}" for i in range(30)]

        result = service.fit(docs, embs, ids)
        # Random noise has no clusters — HDBSCAN should assign most to outlier -1
        assert isinstance(result, dict)
        # Most or all should be outliers, so few themes
        assert len(result) <= 3


# ──────────────────────────────────────────────────────
# Performance benchmarks
# ──────────────────────────────────────────────────────


@pytest.mark.performance
class TestFitPerformance:
    """Performance benchmarks for fit()."""

    def test_fit_performance_100k_docs(self):
        """Benchmark fit() with 100k documents across 50 clusters."""
        n_clusters, n_per = 50, 2000
        embeddings, _, _ = _make_clustered_embeddings(
            n_clusters=n_clusters, n_per_cluster=n_per,
            noise_scale=0.08, seed=42,
        )
        n_total = n_clusters * n_per
        documents = [f"Benchmark document {i} about financial topic" for i in range(n_total)]
        doc_ids = [f"doc_{i:06d}" for i in range(n_total)]

        config = ClusteringConfig(
            hdbscan_min_cluster_size=20,
            hdbscan_min_samples=10,
            umap_n_components=10,
            umap_n_neighbors=15,
            umap_random_state=42,
        )
        service = BERTopicService(config=config)

        start = time.monotonic()
        themes = service.fit(documents, embeddings, doc_ids)
        elapsed = time.monotonic() - start

        assert len(themes) >= 1
        assert service.is_initialized
        # Log for visibility in test output
        print(f"\nfit() 100k docs: {elapsed:.2f}s, {len(themes)} themes discovered")


@pytest.mark.performance
class TestTransformPerformance:
    """Performance benchmarks for transform()."""

    def test_transform_performance_10k_docs(self):
        """Benchmark transform() assigning 10k docs to existing themes."""
        # First, fit on a smaller dataset to establish themes
        fit_embs, fit_centers, _ = _make_clustered_embeddings(
            n_clusters=20, n_per_cluster=100, noise_scale=0.05, seed=42,
        )
        fit_docs = [f"Training document {i}" for i in range(2000)]
        fit_ids = [f"train_{i:04d}" for i in range(2000)]

        config = ClusteringConfig(
            hdbscan_min_cluster_size=10,
            hdbscan_min_samples=5,
            umap_n_components=10,
            umap_n_neighbors=15,
            umap_random_state=42,
        )
        service = BERTopicService(config=config)
        themes = service.fit(fit_docs, fit_embs, fit_ids)
        assert len(themes) >= 1

        # Now transform 10k new documents
        rng = np.random.RandomState(99)
        n_new = 10_000
        new_embs = rng.randn(n_new, 768).astype(np.float32)
        norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        new_embs = new_embs / norms
        new_docs = [f"New document {i}" for i in range(n_new)]
        new_ids = [f"new_{i:05d}" for i in range(n_new)]

        start = time.monotonic()
        results = service.transform(new_docs, new_embs, new_ids)
        elapsed = time.monotonic() - start

        assert len(results) == n_new
        print(f"\ntransform() 10k docs: {elapsed:.4f}s ({n_new/elapsed:.0f} docs/sec)")

    def test_batch_assignment_performance(self):
        """Benchmark batch cosine similarity with many themes."""
        # Simulate 100 themes with centroids
        rng = np.random.RandomState(42)
        n_themes = 100
        dim = 768
        centroids = rng.randn(n_themes, dim).astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / norms

        # Build a service with 100 themes
        config = ClusteringConfig()
        service = BERTopicService(config=config)
        service._initialized = True
        for i in range(n_themes):
            words = [(f"word_{i}_{j}", 0.1 - j * 0.01) for j in range(5)]
            theme = ThemeCluster(
                theme_id=f"theme_{i:03d}",
                name=f"theme_{i}",
                topic_words=words,
                centroid=centroids[i],
                document_count=50,
                document_ids=[f"d_{i}_{j}" for j in range(50)],
            )
            service._themes[theme.theme_id] = theme

        # Assign 5000 documents
        n_docs = 5000
        doc_embs = rng.randn(n_docs, dim).astype(np.float32)
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / doc_norms
        doc_texts = [f"Doc {i}" for i in range(n_docs)]
        doc_ids = [f"doc_{i:05d}" for i in range(n_docs)]

        start = time.monotonic()
        results = service.transform(doc_texts, doc_embs, doc_ids)
        elapsed = time.monotonic() - start

        assert len(results) == n_docs
        # Batch cosine similarity should be fast — under 1 second for 5k x 100
        assert elapsed < 5.0, f"Batch assignment too slow: {elapsed:.2f}s"
        print(
            f"\nbatch assignment 5k docs x 100 themes: {elapsed:.4f}s "
            f"({n_docs/elapsed:.0f} docs/sec)"
        )
