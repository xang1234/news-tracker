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
from datetime import datetime, timezone
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
        self._new_theme_candidates: list[tuple[str, np.ndarray]] = []

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

    @property
    def new_theme_candidates(self) -> list[tuple[str, np.ndarray]]:
        """Return a copy of new theme candidate (doc_id, embedding) pairs."""
        return list(self._new_theme_candidates)

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

    def transform(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        document_ids: list[str],
    ) -> list[tuple[str, list[str], float]]:
        """
        Assign new documents to existing themes via cosine similarity.

        Uses three-tier assignment against theme centroids:
        - Strong (>= similarity_threshold_assign): assign + EMA centroid update
        - Weak (>= similarity_threshold_new, < assign): assign, no centroid update
        - New candidate (< similarity_threshold_new): buffered for future theme creation

        Args:
            documents: List of document text strings (unused in similarity, kept for API consistency).
            embeddings: Pre-computed embedding matrix of shape (n_docs, embedding_dim).
            document_ids: List of document IDs corresponding to each document.

        Returns:
            List of (doc_id, [theme_ids], max_similarity) tuples.
            Empty list if not initialized, no themes, or on internal error.

        Raises:
            ValueError: If input lengths don't match.
        """
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
            return []

        if not self._initialized or not self._themes:
            return []

        try:
            return self._assign_documents(embeddings, document_ids)
        except Exception:
            logger.exception("Transform failed")
            return []

    def _assign_documents(
        self,
        embeddings: np.ndarray,
        document_ids: list[str],
    ) -> list[tuple[str, list[str], float]]:
        """
        Core assignment logic: batch cosine similarity + three-tier routing.

        Args:
            embeddings: Document embedding matrix (n_docs, dim).
            document_ids: Document ID list.

        Returns:
            List of (doc_id, [theme_ids], max_similarity) tuples.
        """
        theme_ids_ordered = list(self._themes.keys())
        themes_ordered = [self._themes[tid] for tid in theme_ids_ordered]

        # Build centroid matrix (n_themes, dim)
        centroid_matrix = np.vstack([t.centroid for t in themes_ordered])

        # Batch cosine similarity via normalized dot product
        # Normalize embeddings
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1.0, emb_norms)
        emb_normalized = embeddings / emb_norms

        # Normalize centroids
        cen_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        cen_norms = np.where(cen_norms == 0, 1.0, cen_norms)
        cen_normalized = centroid_matrix / cen_norms

        # Similarity matrix: (n_docs, n_themes)
        sim_matrix = emb_normalized @ cen_normalized.T

        # Per-document assignment
        results: list[tuple[str, list[str], float]] = []
        now = datetime.now(timezone.utc)
        lr = self.config.centroid_learning_rate
        assign_threshold = self.config.similarity_threshold_assign
        new_threshold = self.config.similarity_threshold_new

        for i, doc_id in enumerate(document_ids):
            max_idx = int(np.argmax(sim_matrix[i]))
            max_sim = float(sim_matrix[i, max_idx])

            if max_sim >= assign_threshold:
                # Strong assignment: assign + EMA centroid update
                theme = themes_ordered[max_idx]
                theme.document_ids.append(doc_id)
                theme.document_count += 1
                theme.centroid = (1 - lr) * theme.centroid + lr * embeddings[i]
                theme.updated_at = now
                results.append((doc_id, [theme_ids_ordered[max_idx]], max_sim))

            elif max_sim >= new_threshold:
                # Weak assignment: assign, no centroid update
                theme = themes_ordered[max_idx]
                theme.document_ids.append(doc_id)
                theme.document_count += 1
                results.append((doc_id, [theme_ids_ordered[max_idx]], max_sim))

            else:
                # New theme candidate: buffer for future theme creation
                self._new_theme_candidates.append(
                    (doc_id, embeddings[i].copy())
                )
                results.append((doc_id, [], max_sim))

        return results

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

    def merge_similar_themes(self) -> list[tuple[str, str]]:
        """
        Merge themes whose centroids exceed the similarity threshold.

        Uses greedy highest-similarity-first strategy with a merged_set to
        prevent chain merges. The larger theme (by document_count) survives;
        the absorbed theme is removed. Survivor gets weighted centroid,
        combined topic words, and merged document IDs.

        Returns:
            List of (merged_from_id, merged_into_id) tuples using the
            original theme IDs (before re-keying). Empty list if no merges
            occurred or service is not initialized.
        """
        if not self._initialized or len(self._themes) < 2:
            return []

        theme_ids_ordered = list(self._themes.keys())
        themes_ordered = [self._themes[tid] for tid in theme_ids_ordered]
        n_themes = len(themes_ordered)

        # Build centroid matrix and compute pairwise cosine similarity
        centroid_matrix = np.vstack([t.centroid for t in themes_ordered])
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = centroid_matrix / norms
        sim_matrix = normalized @ normalized.T

        # Collect upper-triangle pairs above merge threshold
        merge_threshold = self.config.similarity_threshold_merge
        pairs: list[tuple[float, int, int]] = []
        for i in range(n_themes):
            for j in range(i + 1, n_themes):
                if sim_matrix[i, j] >= merge_threshold:
                    pairs.append((float(sim_matrix[i, j]), i, j))

        if not pairs:
            return []

        # Sort descending by similarity (greedy highest-first)
        pairs.sort(key=lambda x: x[0], reverse=True)

        merged_set: set[int] = set()
        merge_results: list[tuple[str, str]] = []
        now = datetime.now(timezone.utc)

        for sim, idx_a, idx_b in pairs:
            if idx_a in merged_set or idx_b in merged_set:
                continue

            theme_a = themes_ordered[idx_a]
            theme_b = themes_ordered[idx_b]

            # Survivor = larger document_count (ties: lower index)
            if theme_a.document_count >= theme_b.document_count:
                survivor, absorbed = theme_a, theme_b
                absorbed_idx = idx_b
            else:
                survivor, absorbed = theme_b, theme_a
                absorbed_idx = idx_a

            original_survivor_id = survivor.theme_id
            original_absorbed_id = absorbed.theme_id
            old_centroid = survivor.centroid.copy()

            # Weighted centroid
            n1, n2 = survivor.document_count, absorbed.document_count
            survivor.centroid = (n1 * survivor.centroid + n2 * absorbed.centroid) / (
                n1 + n2
            )

            # Merge topic words: combine, dedup by word keeping max score
            word_scores: dict[str, float] = {}
            for word, score in survivor.topic_words:
                word_scores[word] = max(word_scores.get(word, 0.0), score)
            for word, score in absorbed.topic_words:
                word_scores[word] = max(word_scores.get(word, 0.0), score)
            merged_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            survivor.topic_words = merged_words[: self.config.top_n_words]

            # Combine document IDs and counts
            survivor.document_ids.extend(absorbed.document_ids)
            survivor.document_count = n1 + n2
            survivor.updated_at = now

            # Track merge in metadata
            prev_merges = survivor.metadata.get("merged_from", [])
            prev_merges.append(original_absorbed_id)
            survivor.metadata["merged_from"] = prev_merges

            # Log merge details
            centroid_shift = float(np.linalg.norm(survivor.centroid - old_centroid))
            logger.info(
                f"Merged theme {original_absorbed_id} into {original_survivor_id}: "
                f"similarity={sim:.4f}, centroid_shift={centroid_shift:.6f}, "
                f"combined_docs={survivor.document_count}"
            )

            merged_set.add(absorbed_idx)
            merge_results.append((original_absorbed_id, original_survivor_id))

        # Re-key _themes: regenerate IDs on survivors, remove absorbed
        new_themes: dict[str, ThemeCluster] = {}
        for idx, theme in enumerate(themes_ordered):
            if idx in merged_set:
                continue
            # Regenerate theme_id and name from (possibly updated) topic words
            theme.theme_id = ThemeCluster.generate_theme_id(theme.topic_words)
            theme.name = ThemeCluster.generate_name(theme.topic_words)
            new_themes[theme.theme_id] = theme

        self._themes = new_themes

        logger.info(
            f"Merge complete: {len(merge_results)} merges, "
            f"{len(self._themes)} themes remaining"
        )

        return merge_results

    def check_new_themes(
        self,
        candidates: list[tuple[str, str, np.ndarray]],
    ) -> list[ThemeCluster]:
        """
        Detect emerging themes from outlier candidate documents.

        Runs lightweight HDBSCAN on candidate embeddings (no UMAP — pool is
        too small for meaningful dimensionality reduction). New clusters are
        checked against existing themes to avoid duplicates, then added to
        the service's theme dictionary.

        Args:
            candidates: List of (doc_id, text, embedding) triples. Text is
                needed for TF-IDF keyword extraction since _new_theme_candidates
                only stores (doc_id, embedding).

        Returns:
            List of newly created ThemeCluster objects. Empty list if not
            initialized, too few candidates, or no valid clusters found.
        """
        if not self._initialized or not candidates:
            return []

        min_candidates = max(3, self.config.hdbscan_min_cluster_size // 2)
        if len(candidates) < min_candidates:
            return []

        doc_ids = [c[0] for c in candidates]
        texts = [c[1] for c in candidates]
        embeddings = np.vstack([c[2] for c in candidates])

        try:
            clusterer = self._create_mini_clusterer(min_size=min_candidates)
            labels = clusterer.fit_predict(embeddings)
        except Exception:
            logger.exception("Mini-clustering failed")
            return []

        # Build existing centroid matrix for overlap checking
        existing_centroids = None
        if self._themes:
            existing_centroids = np.vstack(
                [t.centroid for t in self._themes.values()]
            )
            cen_norms = np.linalg.norm(existing_centroids, axis=1, keepdims=True)
            cen_norms = np.where(cen_norms == 0, 1.0, cen_norms)
            existing_centroids_normalized = existing_centroids / cen_norms

        new_themes: list[ThemeCluster] = []
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            if label == -1:
                continue

            # Gather cluster members
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            cluster_embeddings = embeddings[indices]
            cluster_texts = [texts[i] for i in indices]
            cluster_doc_ids = [doc_ids[i] for i in indices]

            # Compute centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Overlap check against existing themes
            if existing_centroids is not None:
                cen_norm = np.linalg.norm(centroid)
                if cen_norm == 0:
                    continue
                centroid_normalized = centroid / cen_norm
                similarities = existing_centroids_normalized @ centroid_normalized
                max_sim = float(np.max(similarities))
                if max_sim >= self.config.similarity_threshold_new:
                    logger.debug(
                        f"Skipping candidate cluster (label={label}): "
                        f"max_sim={max_sim:.4f} >= threshold "
                        f"{self.config.similarity_threshold_new}"
                    )
                    continue

            # Extract keywords via TF-IDF
            topic_words = self._extract_keywords_tfidf(cluster_texts)

            # Create ThemeCluster
            theme_id = ThemeCluster.generate_theme_id(topic_words)
            name = ThemeCluster.generate_name(topic_words)

            theme = ThemeCluster(
                theme_id=theme_id,
                name=name,
                topic_words=topic_words,
                centroid=centroid,
                document_count=len(indices),
                document_ids=cluster_doc_ids,
                metadata={"lifecycle_stage": "emerging"},
            )

            self._themes[theme_id] = theme
            new_themes.append(theme)

            top_words = [w for w, _ in topic_words[:3]]
            logger.info(
                f"New theme detected: {theme_id} ({name}), "
                f"{len(indices)} docs, keywords={top_words}"
            )

        # Clear ALL passed-in candidate doc_ids from _new_theme_candidates
        passed_doc_ids = set(doc_ids)
        self._new_theme_candidates = [
            (did, emb)
            for did, emb in self._new_theme_candidates
            if did not in passed_doc_ids
        ]

        logger.info(
            f"New theme check complete: {len(new_themes)} themes created "
            f"from {len(candidates)} candidates"
        )

        return new_themes

    def _extract_keywords_tfidf(
        self, texts: list[str]
    ) -> list[tuple[str, float]]:
        """
        Extract top keywords from texts using TF-IDF scoring.

        Lightweight alternative to full BERTopic for small candidate clusters.
        Uses sklearn CountVectorizer + TfidfTransformer to score n-grams,
        then returns the top-ranked keywords.

        Args:
            texts: List of document text strings.

        Returns:
            List of (word, score) tuples sorted by TF-IDF score descending.
            Returns [("unknown", 0.0)] on extraction failure.
        """
        try:
            from sklearn.feature_extraction.text import (
                CountVectorizer,
                TfidfTransformer,
            )

            vectorizer = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=1000,
            )
            counts = vectorizer.fit_transform(texts)
            tfidf = TfidfTransformer().fit_transform(counts)

            # Sum TF-IDF scores across all documents per term
            scores = np.asarray(tfidf.sum(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()

            # Sort by score descending, take top_n_words
            top_indices = scores.argsort()[::-1][: self.config.top_n_words]
            return [(feature_names[i], float(scores[i])) for i in top_indices]

        except Exception:
            logger.exception("TF-IDF keyword extraction failed")
            return [("unknown", 0.0)]

    def _create_mini_clusterer(self, min_size: int) -> Any:
        """
        Create a lightweight HDBSCAN clusterer for small candidate pools.

        Follows the deferred-import pattern of _create_model() for testability
        and to avoid loading hdbscan at module level.

        Args:
            min_size: Minimum cluster size for HDBSCAN.

        Returns:
            Configured HDBSCAN instance.
        """
        from hdbscan import HDBSCAN

        return HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(1, min_size // 2),
            metric="euclidean",
        )

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
                "n_new_theme_candidates": len(self._new_theme_candidates),
            }

        total_docs = sum(t.document_count for t in self._themes.values())
        return {
            "initialized": True,
            "n_themes": len(self._themes),
            "n_documents": total_docs,
            "n_new_theme_candidates": len(self._new_theme_candidates),
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
