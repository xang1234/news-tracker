"""
BERTopic clustering service configuration.

Provides Pydantic settings for the BERTopic-based document clustering service
including UMAP dimensionality reduction, HDBSCAN clustering, c-TF-IDF topic
representation, and theme assignment thresholds.

Parameter Tuning Guide:
    - hdbscan_min_cluster_size: Lower (5-10) = more granular themes,
      higher (20-50) = broader themes. Start with 10 for financial news.
    - hdbscan_min_samples: Lower = more clusters, higher = more conservative.
      Keep <= min_cluster_size.
    - umap_n_components: 5-15 works well. Higher preserves more information
      but is slower. 10 is a good default for 768-dim FinBERT embeddings.
    - similarity_threshold_assign: Higher = fewer assignments to existing themes,
      lower = more. 0.75 balances precision/recall for financial topics.
    - similarity_threshold_merge: 0.85 is conservative (only merge near-duplicates),
      0.75 is more aggressive (merge related themes).
    - centroid_learning_rate: Controls EMA update speed for theme centroids.
      Lower = more stable themes, higher = faster adaptation to new content.
"""

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClusteringConfig(BaseSettings):
    """
    Configuration for the BERTopic clustering service.

    All settings can be overridden via environment variables prefixed with CLUSTERING_.

    Example:
        CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE=20
        CLUSTERING_UMAP_N_COMPONENTS=15
        CLUSTERING_SIMILARITY_THRESHOLD_ASSIGN=0.80
    """

    model_config = SettingsConfigDict(
        env_prefix="CLUSTERING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # UMAP parameters (dimensionality reduction)
    umap_n_neighbors: int = Field(
        default=15,
        ge=2,
        le=200,
        description="Local neighborhood size for UMAP. Larger = more global structure.",
    )
    umap_n_components: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Target dimensions after reduction. 5-15 works well for text.",
    )
    umap_min_dist: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum distance between points. 0.0 = tight clusters preferred.",
    )
    umap_metric: str = Field(
        default="cosine",
        description="Distance metric for UMAP. Cosine is standard for text embeddings.",
    )
    umap_random_state: int = Field(
        default=42,
        description="Random seed for reproducible dimensionality reduction.",
    )

    # HDBSCAN parameters (density-based clustering)
    hdbscan_min_cluster_size: int = Field(
        default=10,
        ge=2,
        le=500,
        description="Minimum documents per theme. Lower = more granular themes.",
    )
    hdbscan_min_samples: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Core point threshold. Lower = more clusters, higher = more conservative.",
    )
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = Field(
        default="eom",
        description="Cluster extraction method. 'eom' = variable sizes, 'leaf' = uniform.",
    )
    hdbscan_prediction_data: bool = Field(
        default=True,
        description="Enable soft clustering for approximate_predict on new documents.",
    )

    # c-TF-IDF parameters (topic representation)
    top_n_words: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of representative keywords per topic.",
    )
    nr_topics: Optional[int] = Field(
        default=None,
        ge=2,
        description="Target number of topics. None = automatic via HDBSCAN.",
    )

    # Assignment thresholds
    similarity_threshold_assign: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to assign a document to an existing theme.",
    )
    similarity_threshold_merge: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity between theme centroids to trigger merge.",
    )
    similarity_threshold_new: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Below this similarity, flag document as a potential new theme.",
    )

    # Centroid update
    centroid_learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="EMA update rate for theme centroids. Lower = more stable themes.",
    )

    # Redis stream configuration (clustering queue)
    stream_name: str = Field(
        default="clustering_queue",
        description="Redis stream name for clustering jobs.",
    )
    consumer_group: str = Field(
        default="clustering_workers",
        description="Consumer group name for clustering workers.",
    )
    max_stream_length: int = Field(
        default=50_000,
        description="Maximum stream length before trimming.",
    )
    dlq_stream_name: str = Field(
        default="clustering_queue:dlq",
        description="Dead letter queue stream name.",
    )

    # Model persistence
    model_save_dir: str = Field(
        default="models/clustering",
        description="Directory for persisting trained BERTopic models.",
    )

    # Worker configuration
    worker_batch_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout in seconds for batch accumulation before clustering.",
    )
    worker_idle_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for idle worker shutdown.",
    )

    # Queue reclaim configuration
    idle_timeout_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=300_000,
        description="Idle time before reclaiming pending messages (ms).",
    )
    max_delivery_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max delivery attempts before moving to DLQ.",
    )
