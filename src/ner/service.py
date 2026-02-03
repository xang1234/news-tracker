"""
NER Service for financial entity extraction.

Provides domain-specific named entity recognition for semiconductor
and financial news content using spaCy with custom EntityRuler patterns
and optional coreference resolution via fastcoref.

Architecture:
- Lazy model loading (models loaded on first extract() call)
- EntityRuler patterns run before statistical NER
- Optional fastcoref for coreference resolution
- Fuzzy matching for entity variations using rapidfuzz
- Optional embedding-based semantic similarity for theme linking
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rapidfuzz import fuzz

from src.config.tickers import COMPANY_TO_TICKER, SEMICONDUCTOR_TICKERS
from src.ner.config import NERConfig
from src.ner.schemas import EntityType, FinancialEntity

if TYPE_CHECKING:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc

    from src.embedding.service import EmbeddingService

logger = logging.getLogger(__name__)


class NERService:
    """
    Named Entity Recognition service for financial text.

    Extracts company names, tickers, products, technologies, and metrics
    from text using a combination of:
    - spaCy's statistical NER
    - EntityRuler pattern matching
    - Fuzzy matching for entity variations
    - Coreference resolution (optional)
    - Embedding-based semantic similarity for theme linking (optional)

    Usage:
        >>> service = NERService()
        >>> entities = service.extract_sync("Nvidia announced HBM3E support for H200")
        >>> for e in entities:
        ...     print(f"{e.type}: {e.text} -> {e.normalized}")
        COMPANY: Nvidia -> NVIDIA
        TECHNOLOGY: HBM3E -> HBM3E
        PRODUCT: H200 -> H200

    For semantic theme linking:
        >>> from src.embedding.service import EmbeddingService
        >>> embedding_svc = EmbeddingService()
        >>> service = NERService(embedding_service=embedding_svc)
        >>> scores = await service.link_entities_to_theme_semantic(
        ...     entities, ["AI accelerator", "deep learning"]
        ... )

    Note:
        Models are loaded lazily on first extract() call to avoid
        memory overhead when NER is disabled.
    """

    def __init__(
        self,
        config: NERConfig | None = None,
        embedding_service: "EmbeddingService | None" = None,
    ):
        """
        Initialize NER service.

        Args:
            config: NER configuration. If None, uses default config.
            embedding_service: Optional embedding service for semantic theme linking.
                When provided, enables embedding-based similarity scoring
                in link_entities_to_theme().
        """
        self.config = config or NERConfig()
        self._nlp: "Language | None" = None
        self._coref_model: Any = None
        self._embedding_service = embedding_service
        self._initialized = False

        # Build company name lookup (lowercase -> normalized name, ticker)
        self._company_lookup: dict[str, tuple[str, str]] = {}
        for name, ticker in COMPANY_TO_TICKER.items():
            if ticker in SEMICONDUCTOR_TICKERS:
                self._company_lookup[name.lower()] = (name.upper(), ticker)

    @property
    def is_initialized(self) -> bool:
        """Check if models are loaded."""
        return self._initialized

    @property
    def has_embedding_service(self) -> bool:
        """Check if embedding service is available for semantic linking."""
        return self._embedding_service is not None

    def _initialize(self) -> None:
        """
        Load spaCy model and configure pipeline.

        This is called lazily on first extract() call.
        """
        if self._initialized:
            return

        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy is required for NER. Install with: pip install spacy"
            ) from e

        # Try to load the configured model
        model_name = self.config.spacy_model
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self._nlp = spacy.load(model_name)
        except OSError:
            # Try fallback model
            fallback = self.config.fallback_model
            logger.warning(
                f"Model '{model_name}' not found, trying fallback: {fallback}"
            )
            try:
                self._nlp = spacy.load(fallback)
            except OSError as e:
                raise OSError(
                    f"Neither '{model_name}' nor '{fallback}' could be loaded. "
                    f"Install with: python -m spacy download {model_name}"
                ) from e

        # Add EntityRuler before NER component
        self._load_patterns()

        # Load coreference model if enabled
        if self.config.enable_coreference:
            self._load_coref_model()

        self._initialized = True
        logger.info(f"NER service initialized with model: {self._nlp.meta['name']}")

    def _load_patterns(self) -> None:
        """Load EntityRuler patterns from JSONL files."""
        if self._nlp is None:
            return

        patterns_dir = self.config.patterns_dir
        if not patterns_dir.exists():
            logger.warning(f"Patterns directory not found: {patterns_dir}")
            return

        # Collect all patterns from JSONL files
        all_patterns: list[dict[str, Any]] = []

        for pattern_file in patterns_dir.glob("*.jsonl"):
            try:
                import json
                with open(pattern_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pattern = json.loads(line)
                            all_patterns.append(pattern)
                logger.debug(f"Loaded patterns from: {pattern_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load patterns from {pattern_file}: {e}")

        if all_patterns:
            # Add EntityRuler before the NER component
            try:
                ruler = self._nlp.add_pipe(
                    "entity_ruler",
                    before="ner",
                    config={"overwrite_ents": False},
                )
                ruler.add_patterns(all_patterns)
                logger.info(f"Loaded {len(all_patterns)} EntityRuler patterns")
            except ValueError:
                # NER component might not exist in the pipeline
                ruler = self._nlp.add_pipe(
                    "entity_ruler",
                    last=True,
                    config={"overwrite_ents": False},
                )
                ruler.add_patterns(all_patterns)
                logger.info(
                    f"Loaded {len(all_patterns)} EntityRuler patterns (appended)"
                )

    def _load_coref_model(self) -> None:
        """Load fastcoref model for coreference resolution."""
        try:
            from fastcoref import FCoref

            logger.info("Loading fastcoref model...")
            self._coref_model = FCoref(device="cpu")
            logger.info("Coreference model loaded")
        except ImportError:
            logger.warning(
                "fastcoref not installed. Coreference resolution disabled. "
                "Install with: pip install fastcoref"
            )
            self._coref_model = None
        except Exception as e:
            logger.warning(f"Failed to load coref model: {e}")
            self._coref_model = None

    def extract_sync(self, text: str) -> list[FinancialEntity]:
        """
        Extract entities from text (synchronous).

        Args:
            text: Text to extract entities from.

        Returns:
            List of extracted FinancialEntity objects.
        """
        if not self._initialized:
            self._initialize()

        if not text or not text.strip():
            return []

        # Truncate if too long
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]
            logger.debug(f"Text truncated to {self.config.max_text_length} chars")

        # Process with spaCy
        doc = self._nlp(text)

        # Extract entities from spaCy NER + EntityRuler
        entities = self._extract_from_doc(doc, text)

        # Add cashtag entities (pattern-based)
        entities.extend(self._extract_cashtags(text))

        # Apply coreference resolution if available
        if self._coref_model is not None:
            entities = self._resolve_coreferences(text, entities)

        # Fuzzy match company names not caught by NER
        entities.extend(self._fuzzy_match_companies(text, entities))

        # Deduplicate overlapping entities
        entities = self._deduplicate_entities(entities)

        # Filter by confidence threshold
        entities = [
            e for e in entities if e.confidence >= self.config.confidence_threshold
        ]

        # Filter by configured entity types
        entities = [e for e in entities if e.type in self.config.extract_types]

        return entities

    async def extract(self, text: str) -> list[FinancialEntity]:
        """
        Extract entities from text (async wrapper).

        Note: spaCy processing is CPU-bound, so this is a simple
        wrapper around extract_sync. For true parallelism, use
        extract_batch with multiple texts.

        Args:
            text: Text to extract entities from.

        Returns:
            List of extracted FinancialEntity objects.
        """
        return self.extract_sync(text)

    async def extract_batch(
        self, texts: list[str]
    ) -> list[list[FinancialEntity]]:
        """
        Extract entities from multiple texts.

        Uses spaCy's nlp.pipe() for efficient batch processing.

        Args:
            texts: List of texts to process.

        Returns:
            List of entity lists, one per input text.
        """
        if not self._initialized:
            self._initialize()

        if not texts:
            return []

        results: list[list[FinancialEntity]] = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            # Truncate long texts
            batch = [
                t[: self.config.max_text_length] if len(t) > self.config.max_text_length else t
                for t in batch
            ]

            # Process batch with spaCy pipe
            docs = list(self._nlp.pipe(batch))

            for doc, text in zip(docs, batch):
                entities = self._extract_from_doc(doc, text)
                entities.extend(self._extract_cashtags(text))

                if self._coref_model is not None:
                    entities = self._resolve_coreferences(text, entities)

                entities.extend(self._fuzzy_match_companies(text, entities))
                entities = self._deduplicate_entities(entities)
                entities = [
                    e for e in entities
                    if e.confidence >= self.config.confidence_threshold
                    and e.type in self.config.extract_types
                ]
                results.append(entities)

        return results

    def extract_tickers(self, text: str) -> list[str]:
        """
        Convenience method to extract just ticker symbols.

        Args:
            text: Text to extract tickers from.

        Returns:
            List of unique ticker symbols.
        """
        entities = self.extract_sync(text)
        tickers: set[str] = set()

        for entity in entities:
            if entity.type == "TICKER":
                tickers.add(entity.normalized)
            elif entity.type == "COMPANY" and "ticker" in entity.metadata:
                tickers.add(entity.metadata["ticker"])

        return sorted(tickers)

    def link_entities_to_theme(
        self,
        entities: list[FinancialEntity],
        theme_keywords: list[str],
    ) -> dict[str, float]:
        """
        Calculate relevance scores for entities to a theme (synchronous).

        Uses substring matching combined with domain heuristics.
        For more accurate semantic matching, use link_entities_to_theme_semantic().

        Useful for disambiguation when multiple interpretations
        are possible (e.g., "Apple" the company vs. the fruit).

        Args:
            entities: List of extracted entities.
            theme_keywords: Keywords defining the theme context.

        Returns:
            Dict mapping entity normalized names to relevance scores (0.0-1.0).
        """
        theme_text = " ".join(theme_keywords).lower()
        scores: dict[str, float] = {}

        for entity in entities:
            score = 0.0

            # Check if entity text appears in theme
            if entity.text.lower() in theme_text:
                score += 0.5

            # Check metadata for theme-related info
            if entity.metadata.get("ticker") in SEMICONDUCTOR_TICKERS:
                score += 0.3

            # Check if entity type aligns with theme
            if entity.type in ("TECHNOLOGY", "PRODUCT"):
                score += 0.2

            scores[entity.normalized] = min(1.0, score)

        return scores

    async def link_entities_to_theme_semantic(
        self,
        entities: list[FinancialEntity],
        theme_keywords: list[str],
    ) -> dict[str, float]:
        """
        Calculate relevance scores using embedding-based semantic similarity.

        Uses cosine similarity between entity text and theme keywords
        in embedding space for more robust disambiguation. Combines
        semantic similarity with domain-specific heuristics.

        Requires an EmbeddingService to be injected at construction time.
        Falls back to link_entities_to_theme() if no embedding service
        is available.

        Scoring formula:
            score = (semantic_base_score * cosine_similarity) + domain_bonus
        where domain_bonus adds:
            +0.2 for semiconductor tickers
            +0.1 for TECHNOLOGY/PRODUCT entity types

        Args:
            entities: List of extracted entities.
            theme_keywords: Keywords defining the theme context.

        Returns:
            Dict mapping entity normalized names to relevance scores (0.0-1.0).

        Example:
            >>> # "GPU" semantically matches "graphics processing unit"
            >>> scores = await service.link_entities_to_theme_semantic(
            ...     entities, ["graphics processing unit", "AI accelerator"]
            ... )
            >>> scores["GPU"]  # High score due to semantic similarity
            0.82
        """
        if self._embedding_service is None:
            logger.warning(
                "No embedding service available, falling back to substring matching"
            )
            return self.link_entities_to_theme(entities, theme_keywords)

        if not entities:
            return {}

        scores: dict[str, float] = {}

        # Embed theme keywords as a single context
        theme_text = " ".join(theme_keywords)
        try:
            theme_embedding = await self._embedding_service.embed_minilm(theme_text)
            theme_vec = np.array(theme_embedding)
        except Exception as e:
            logger.warning(f"Failed to embed theme keywords: {e}")
            return self.link_entities_to_theme(entities, theme_keywords)

        # Embed each entity and compute cosine similarity
        for entity in entities:
            # Build entity context string (text + type for disambiguation)
            entity_context = f"{entity.text} {entity.type.lower()}"

            try:
                entity_embedding = await self._embedding_service.embed_minilm(entity_context)
                entity_vec = np.array(entity_embedding)

                # Compute cosine similarity
                similarity = self._cosine_similarity(entity_vec, theme_vec)

                # Apply similarity threshold
                if similarity < self.config.semantic_similarity_threshold:
                    similarity = 0.0

                # Build score: semantic similarity + domain heuristics
                semantic_score = self.config.semantic_base_score * similarity

                # Add domain-specific bonuses
                domain_bonus = 0.0
                if entity.metadata.get("ticker") in SEMICONDUCTOR_TICKERS:
                    domain_bonus += 0.2
                if entity.type in ("TECHNOLOGY", "PRODUCT"):
                    domain_bonus += 0.1

                scores[entity.normalized] = min(1.0, semantic_score + domain_bonus)

            except Exception as e:
                logger.warning(f"Failed to embed entity '{entity.text}': {e}")
                # Fall back to simple matching for this entity
                fallback = self.link_entities_to_theme([entity], theme_keywords)
                scores.update(fallback)

        return scores

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity in range [-1, 1], typically [0, 1] for embeddings.
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _extract_from_doc(
        self, doc: "Doc", original_text: str
    ) -> list[FinancialEntity]:
        """Extract entities from a spaCy Doc."""
        entities: list[FinancialEntity] = []

        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type is None:
                continue

            # Normalize the entity
            normalized = self._normalize_entity(ent.text, entity_type)
            confidence = self._calculate_confidence(ent)

            # Add ticker metadata for company entities
            metadata: dict[str, Any] = {}
            if entity_type == "COMPANY":
                ticker = self._get_ticker_for_company(ent.text)
                if ticker:
                    metadata["ticker"] = ticker

            entities.append(
                FinancialEntity(
                    text=ent.text,
                    type=entity_type,
                    normalized=normalized,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return entities

    def _extract_cashtags(self, text: str) -> list[FinancialEntity]:
        """Extract $TICKER cashtags from text."""
        entities: list[FinancialEntity] = []

        # Match $TICKER pattern (1-5 uppercase letters)
        pattern = r'\$([A-Z]{1,5})\b'

        for match in re.finditer(pattern, text.upper()):
            ticker = match.group(1)
            if ticker in SEMICONDUCTOR_TICKERS:
                # Find the actual position in original text
                # (case-insensitive search)
                for m in re.finditer(rf'\$({ticker})\b', text, re.IGNORECASE):
                    entities.append(
                        FinancialEntity(
                            text=m.group(0),
                            type="TICKER",
                            normalized=ticker,
                            start=m.start(),
                            end=m.end(),
                            confidence=1.0,
                            metadata={"source": "cashtag"},
                        )
                    )
                    break  # Only match first occurrence

        return entities

    def _resolve_coreferences(
        self, text: str, entities: list[FinancialEntity]
    ) -> list[FinancialEntity]:
        """
        Resolve coreferences to link pronouns to entities.

        Uses fastcoref to find mentions like "the company" and
        link them to their antecedents (e.g., "Nvidia").
        """
        if self._coref_model is None:
            return entities

        try:
            # Get coreference clusters
            result = self._coref_model.predict(texts=[text])
            if not result or not result[0]:
                return entities

            clusters = result[0].get_clusters(as_strings=False)

            # Build a map of coreferent spans
            # For each mention, find if it refers to a known entity
            for cluster in clusters:
                # Find if any span in the cluster matches a known entity
                entity_ref: FinancialEntity | None = None
                for start, end in cluster:
                    for entity in entities:
                        if entity.start <= start and entity.end >= end:
                            entity_ref = entity
                            break
                    if entity_ref:
                        break

                # If we found an entity reference, add entries for other mentions
                if entity_ref:
                    for start, end in cluster:
                        mention_text = text[start:end]
                        # Skip if this is the original entity
                        if start == entity_ref.start and end == entity_ref.end:
                            continue
                        # Skip if already covered by another entity
                        if any(e.start == start and e.end == end for e in entities):
                            continue

                        # Add a new entity for this coreference
                        entities.append(
                            FinancialEntity(
                                text=mention_text,
                                type=entity_ref.type,
                                normalized=entity_ref.normalized,
                                start=start,
                                end=end,
                                confidence=entity_ref.confidence * 0.8,  # Slightly lower confidence
                                metadata={
                                    **entity_ref.metadata,
                                    "coreference_of": entity_ref.text,
                                },
                            )
                        )

        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")

        return entities

    def _fuzzy_match_companies(
        self, text: str, existing_entities: list[FinancialEntity]
    ) -> list[FinancialEntity]:
        """Find company names via fuzzy matching."""
        entities: list[FinancialEntity] = []
        text_lower = text.lower()

        # Get spans already covered by entities
        covered_spans = {(e.start, e.end) for e in existing_entities}

        # Check each known company name
        for company_name, (normalized, ticker) in self._company_lookup.items():
            # Skip if we already found this company
            if any(e.normalized == normalized or e.metadata.get("ticker") == ticker
                   for e in existing_entities):
                continue

            # Try exact match first
            idx = text_lower.find(company_name)
            if idx != -1:
                end_idx = idx + len(company_name)
                if (idx, end_idx) not in covered_spans:
                    entities.append(
                        FinancialEntity(
                            text=text[idx:end_idx],
                            type="COMPANY",
                            normalized=normalized,
                            start=idx,
                            end=end_idx,
                            confidence=0.95,
                            metadata={"ticker": ticker, "source": "lookup"},
                        )
                    )
                continue

            # Fuzzy match for variations
            words = text_lower.split()
            for i, word in enumerate(words):
                if len(word) < 4:
                    continue
                score = fuzz.ratio(company_name, word)
                if score >= self.config.fuzzy_threshold:
                    # Find word position in original text
                    pattern = rf'\b{re.escape(word)}\b'
                    for match in re.finditer(pattern, text_lower):
                        if (match.start(), match.end()) not in covered_spans:
                            entities.append(
                                FinancialEntity(
                                    text=text[match.start():match.end()],
                                    type="COMPANY",
                                    normalized=normalized,
                                    start=match.start(),
                                    end=match.end(),
                                    confidence=score / 100.0,
                                    metadata={"ticker": ticker, "source": "fuzzy"},
                                )
                            )
                            break

        return entities

    def _deduplicate_entities(
        self, entities: list[FinancialEntity]
    ) -> list[FinancialEntity]:
        """
        Remove duplicate and overlapping entities.

        When entities overlap, keeps the longer (more specific) one,
        unless the shorter one has significantly higher confidence.
        """
        if len(entities) <= 1:
            return entities

        # Sort by start position, then by length (longer first)
        entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

        result: list[FinancialEntity] = []
        for entity in entities:
            # Check if this entity overlaps with any existing
            overlapping = [e for e in result if entity.overlaps(e)]

            if not overlapping:
                result.append(entity)
            else:
                # Compare with overlapping entities
                for other in overlapping:
                    # Keep the longer entity unless confidence difference is large
                    entity_len = entity.end - entity.start
                    other_len = other.end - other.start

                    if entity_len > other_len and entity.confidence >= other.confidence - 0.2:
                        result.remove(other)
                        result.append(entity)
                        break
                    # If new entity has much higher confidence, prefer it
                    elif entity.confidence > other.confidence + 0.3:
                        result.remove(other)
                        result.append(entity)
                        break

        # Also deduplicate by normalized form + type
        seen: set[tuple[str, str]] = set()
        unique: list[FinancialEntity] = []
        for entity in result:
            key = (entity.normalized, entity.type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _map_spacy_label(self, label: str) -> EntityType | None:
        """Map spaCy entity labels to our entity types."""
        mapping: dict[str, EntityType] = {
            # Standard spaCy labels
            "ORG": "COMPANY",
            "COMPANY": "COMPANY",
            "PRODUCT": "PRODUCT",
            "TECHNOLOGY": "TECHNOLOGY",
            "METRIC": "METRIC",
            "TICKER": "TICKER",
            # Financial-specific labels (if using financial model)
            "MONEY": "METRIC",
            "PERCENT": "METRIC",
            "QUANTITY": "METRIC",
            "CARDINAL": "METRIC",
        }
        return mapping.get(label)

    def _normalize_entity(self, text: str, entity_type: EntityType) -> str:
        """Normalize entity text based on type."""
        text = text.strip()

        if entity_type == "TICKER":
            return text.upper().lstrip("$")

        if entity_type == "COMPANY":
            # Check if we have a canonical name
            lower_text = text.lower()
            if lower_text in self._company_lookup:
                return self._company_lookup[lower_text][0]
            return text.upper()

        if entity_type in ("TECHNOLOGY", "PRODUCT"):
            # Keep original case for these
            return text

        return text

    def _calculate_confidence(self, ent: Any) -> float:
        """Calculate confidence score for a spaCy entity."""
        # spaCy doesn't provide per-entity confidence directly
        # We use heuristics based on entity source and context

        # EntityRuler patterns get high confidence
        if hasattr(ent, "_") and hasattr(ent._, "ruler_pattern"):
            return 0.95

        # Known companies get high confidence
        if ent.label_ in ("ORG", "COMPANY"):
            if ent.text.lower() in self._company_lookup:
                return 0.95
            return 0.8

        # Products and technologies from our patterns
        if ent.label_ in ("PRODUCT", "TECHNOLOGY", "TICKER"):
            return 0.9

        return 0.7

    def _get_ticker_for_company(self, company_name: str) -> str | None:
        """Look up ticker symbol for a company name."""
        lower_name = company_name.lower()
        if lower_name in self._company_lookup:
            return self._company_lookup[lower_name][1]
        return None
