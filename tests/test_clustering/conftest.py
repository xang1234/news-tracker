"""Shared fixtures for clustering tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService


@pytest.fixture
def clustering_config():
    """Small config values for fast test clustering."""
    return ClusteringConfig(
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=2,
        umap_n_components=5,
        umap_n_neighbors=5,
        top_n_words=5,
    )


@pytest.fixture
def sample_documents():
    """15 financial texts in 3 thematic groups (5 each)."""
    return [
        # Group 0: GPU/AI (indices 0-4)
        "Nvidia announces next-generation GPU architecture for AI training",
        "New GPU accelerator delivers breakthrough deep learning performance",
        "AI chip GPU computing power doubles with latest architecture release",
        "Nvidia GPU sales surge driven by AI data center demand",
        "GPU computing platform enables faster AI model training",
        # Group 1: Memory (indices 5-9)
        "HBM3E memory bandwidth reaches new highs for data center chips",
        "Samsung develops advanced HBM memory for AI accelerators",
        "Memory chip demand increases as HBM technology evolves",
        "SK Hynix HBM3E memory production ramps up for AI workloads",
        "High bandwidth memory technology drives semiconductor innovation",
        # Group 2: Manufacturing (indices 10-14)
        "TSMC expands 3nm chip manufacturing capacity in Arizona",
        "Semiconductor fabrication technology advances to 2nm process node",
        "Intel foundry manufacturing services compete with TSMC production",
        "Chip manufacturing yield improvements boost semiconductor output",
        "Advanced semiconductor manufacturing drives global supply chain",
    ]


@pytest.fixture
def sample_embeddings():
    """
    15x768 synthetic embeddings with 3 separable clusters.

    Each cluster is centered around a distinct random vector with small
    per-document noise, producing well-separated groups for testing.
    """
    rng = np.random.RandomState(42)
    dim = 768
    n_per_group = 5

    # Create 3 distinct cluster centers
    centers = rng.randn(3, dim)
    # Normalize centers for cosine-like separation
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    embeddings = []
    for center in centers:
        # Small noise around each center
        noise = rng.randn(n_per_group, dim) * 0.05
        group = center + noise
        # Normalize each embedding
        norms = np.linalg.norm(group, axis=1, keepdims=True)
        group = group / norms
        embeddings.append(group)

    return np.vstack(embeddings).astype(np.float32)


@pytest.fixture
def sample_document_ids():
    """Document IDs for the 15 sample documents."""
    return [f"doc_{i:03d}" for i in range(15)]


@pytest.fixture
def mock_bertopic_model():
    """
    Mock BERTopic model that returns 3 topics + outliers.

    Topic assignments:
    - Indices 0-4: topic 0 (GPU/AI)
    - Indices 5-9: topic 1 (Memory)
    - Indices 10-13: topic 2 (Manufacturing)
    - Index 14: topic -1 (outlier)
    """
    model = MagicMock()

    # fit_transform returns (topics, probabilities)
    topics = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, -1]
    probs = np.random.rand(15, 3)
    model.fit_transform.return_value = (topics, probs)

    # get_topic returns word/score pairs for each topic
    topic_words = {
        0: [
            ("gpu", 0.15),
            ("nvidia", 0.12),
            ("architecture", 0.10),
            ("ai", 0.08),
            ("training", 0.06),
        ],
        1: [
            ("memory", 0.18),
            ("hbm3e", 0.14),
            ("bandwidth", 0.11),
            ("hbm", 0.09),
            ("chip", 0.07),
        ],
        2: [
            ("manufacturing", 0.16),
            ("tsmc", 0.13),
            ("fabrication", 0.10),
            ("semiconductor", 0.08),
            ("process", 0.06),
        ],
    }
    model.get_topic.side_effect = lambda tid: topic_words.get(tid, [])

    return model


@pytest.fixture
def fitted_service(
    clustering_config,
    sample_documents,
    sample_embeddings,
    sample_document_ids,
    mock_bertopic_model,
):
    """
    BERTopicService with 3 themes pre-fitted via mock model.

    Themes are built from sample_embeddings with 3 well-separated clusters:
    - Cluster 0 (GPU/AI): doc_000..doc_004, centroid ≈ center[0]
    - Cluster 1 (Memory): doc_005..doc_009, centroid ≈ center[1]
    - Cluster 2 (Manufacturing): doc_010..doc_013, centroid ≈ center[2]
    - doc_014 is an outlier (excluded)
    """
    service = BERTopicService(config=clustering_config)

    with patch.object(service, "_create_model", return_value=mock_bertopic_model):
        service.fit(sample_documents, sample_embeddings, sample_document_ids)

    return service


@pytest.fixture
def service_with_similar_themes(clustering_config):
    """
    BERTopicService with 3 manually-constructed themes, two of which are
    very similar (cosine sim ~0.9998, well above the 0.85 merge threshold).

    - Theme A (10 docs): base centroid
    - Theme B (5 docs): centroid perturbed ~0.02 from A (nearly identical)
    - Theme C (8 docs): orthogonal centroid (distant from both A and B)
    """
    rng = np.random.RandomState(42)
    dim = 768

    # Theme A: random unit vector
    centroid_a = rng.randn(dim).astype(np.float64)
    centroid_a /= np.linalg.norm(centroid_a)

    # Theme B: tiny perturbation of A (cosine sim ~0.9998)
    perturbation = rng.randn(dim) * 0.02
    centroid_b = centroid_a + perturbation
    centroid_b /= np.linalg.norm(centroid_b)

    # Theme C: orthogonal direction (distant from A and B)
    centroid_c = rng.randn(dim).astype(np.float64)
    # Remove component along A to make more orthogonal
    centroid_c -= np.dot(centroid_c, centroid_a) * centroid_a
    centroid_c /= np.linalg.norm(centroid_c)

    words_a = [("gpu", 0.15), ("nvidia", 0.12), ("ai", 0.10), ("chip", 0.08), ("training", 0.06)]
    words_b = [("gpu", 0.14), ("accelerator", 0.11), ("nvidia", 0.10), ("compute", 0.07), ("deep", 0.05)]
    words_c = [("memory", 0.18), ("hbm3e", 0.14), ("bandwidth", 0.11), ("samsung", 0.09), ("dram", 0.07)]

    theme_a = ThemeCluster(
        theme_id=ThemeCluster.generate_theme_id(words_a),
        name=ThemeCluster.generate_name(words_a),
        topic_words=words_a,
        centroid=centroid_a,
        document_count=10,
        document_ids=[f"doc_a_{i}" for i in range(10)],
        metadata={"bertopic_topic_id": 0},
    )
    theme_b = ThemeCluster(
        theme_id=ThemeCluster.generate_theme_id(words_b),
        name=ThemeCluster.generate_name(words_b),
        topic_words=words_b,
        centroid=centroid_b,
        document_count=5,
        document_ids=[f"doc_b_{i}" for i in range(5)],
        metadata={"bertopic_topic_id": 1},
    )
    theme_c = ThemeCluster(
        theme_id=ThemeCluster.generate_theme_id(words_c),
        name=ThemeCluster.generate_name(words_c),
        topic_words=words_c,
        centroid=centroid_c,
        document_count=8,
        document_ids=[f"doc_c_{i}" for i in range(8)],
        metadata={"bertopic_topic_id": 2},
    )

    service = BERTopicService(config=clustering_config)
    service._initialized = True
    service._themes = {
        theme_a.theme_id: theme_a,
        theme_b.theme_id: theme_b,
        theme_c.theme_id: theme_c,
    }

    return service
