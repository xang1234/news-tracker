"""
Document deduplication using MinHash LSH.

Identifies near-duplicate documents across platforms using Locality
Sensitive Hashing. This catches:
- Copy-paste campaigns on social media
- Syndicated news articles from wire services
- Reposts with minor modifications

Uses the datasketch library for efficient MinHash LSH implementation.
"""

import logging
import re
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from src.config.settings import get_settings
from src.ingestion.schemas import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class DuplicateResult:
    """Result of duplicate detection."""

    is_duplicate: bool
    similar_doc_ids: list[str] = field(default_factory=list)
    similarity_score: float | None = None


class DeduplicationIndex:
    """
    MinHash LSH index for near-duplicate detection.

    Uses MinHash to create compact signatures of documents, then LSH
    to efficiently find similar documents without pairwise comparison.

    Parameters:
        - threshold: Jaccard similarity threshold (default 0.85)
        - num_perm: Number of permutations for MinHash (default 128)

    Higher num_perm = more accurate but more memory/compute.
    Higher threshold = stricter duplicate detection.

    Note: For production with high document volume, consider using
    Redis-backed LSH (datasketch's MinHashLSH with redis storage).
    """

    def __init__(
        self,
        threshold: float | None = None,
        num_perm: int = 128,
    ):
        """
        Initialize deduplication index.

        Args:
            threshold: Similarity threshold for duplicates (0.0-1.0)
            num_perm: Number of permutations for MinHash
        """
        settings = get_settings()
        self.threshold = threshold or settings.duplicate_threshold
        self.num_perm = num_perm

        # Initialize LSH index
        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # Track inserted documents
        self._doc_count = 0

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return self._doc_count

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for text.

        Uses word-level shingling (3-grams) for better semantic matching.

        Args:
            text: Document text

        Returns:
            MinHash signature
        """
        m = MinHash(num_perm=self.num_perm)

        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()

        # Use 3-word shingles (n-grams)
        # This captures phrase-level similarity
        shingle_size = 3
        for i in range(len(words) - shingle_size + 1):
            shingle = ' '.join(words[i : i + shingle_size])
            m.update(shingle.encode('utf-8'))

        # Also add individual words for short texts
        if len(words) < shingle_size * 2:
            for word in words:
                m.update(word.encode('utf-8'))

        return m

    def check_duplicate(self, doc: NormalizedDocument) -> DuplicateResult:
        """
        Check if document is a duplicate of existing documents.

        Does NOT add the document to the index. Use insert() to add.

        Args:
            doc: Document to check

        Returns:
            DuplicateResult with duplicate status and similar doc IDs
        """
        minhash = self._create_minhash(doc.content)

        # Query for similar documents
        similar_ids = self._lsh.query(minhash)

        if similar_ids:
            return DuplicateResult(
                is_duplicate=True,
                similar_doc_ids=list(similar_ids),
            )

        return DuplicateResult(is_duplicate=False)

    def insert(self, doc: NormalizedDocument) -> bool:
        """
        Insert document into the index.

        Args:
            doc: Document to insert

        Returns:
            True if inserted, False if duplicate detected
        """
        # Check for duplicates first
        result = self.check_duplicate(doc)
        if result.is_duplicate:
            logger.debug(
                f"Duplicate detected: {doc.id} similar to {result.similar_doc_ids}"
            )
            return False

        # Create MinHash and insert
        minhash = self._create_minhash(doc.content)

        try:
            self._lsh.insert(doc.id, minhash)
            self._doc_count += 1
            return True
        except ValueError:
            # Document ID already exists
            logger.debug(f"Document {doc.id} already in index")
            return False

    def insert_if_unique(self, doc: NormalizedDocument) -> DuplicateResult:
        """
        Check and insert document in one operation.

        Args:
            doc: Document to check and potentially insert

        Returns:
            DuplicateResult (with is_duplicate=False if inserted)
        """
        minhash = self._create_minhash(doc.content)

        # Query for similar documents
        similar_ids = self._lsh.query(minhash)

        if similar_ids:
            return DuplicateResult(
                is_duplicate=True,
                similar_doc_ids=list(similar_ids),
            )

        # Insert if unique
        try:
            self._lsh.insert(doc.id, minhash)
            self._doc_count += 1
        except ValueError:
            # Already exists
            pass

        return DuplicateResult(is_duplicate=False)

    def clear(self) -> None:
        """Clear the index."""
        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._doc_count = 0
        logger.info("Deduplication index cleared")


class Deduplicator:
    """
    Document deduplicator with index management.

    Provides high-level API for deduplicating document streams.

    Usage:
        dedup = Deduplicator()

        for doc in documents:
            if dedup.is_duplicate(doc):
                continue  # Skip duplicate
            dedup.add(doc)
            # Process unique document
    """

    def __init__(
        self,
        threshold: float | None = None,
        num_perm: int = 128,
        max_index_size: int = 100_000,
    ):
        """
        Initialize deduplicator.

        Args:
            threshold: Similarity threshold for duplicates
            num_perm: Number of permutations for MinHash
            max_index_size: Maximum documents in index before rotation
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.max_index_size = max_index_size

        self._index = DeduplicationIndex(
            threshold=threshold,
            num_perm=num_perm,
        )

        # Track statistics
        self._total_processed = 0
        self._duplicates_found = 0

    @property
    def stats(self) -> dict[str, int]:
        """Get deduplication statistics."""
        return {
            "total_processed": self._total_processed,
            "duplicates_found": self._duplicates_found,
            "index_size": self._index.size,
        }

    def is_duplicate(self, doc: NormalizedDocument) -> bool:
        """
        Check if document is a duplicate.

        Args:
            doc: Document to check

        Returns:
            True if duplicate, False if unique
        """
        self._total_processed += 1
        result = self._index.check_duplicate(doc)

        if result.is_duplicate:
            self._duplicates_found += 1
            logger.debug(
                f"Duplicate: {doc.id} similar to {result.similar_doc_ids[:3]}"
            )

        return result.is_duplicate

    def add(self, doc: NormalizedDocument) -> bool:
        """
        Add document to the index.

        Should be called after is_duplicate() returns False.

        Args:
            doc: Document to add

        Returns:
            True if added successfully
        """
        # Check if index needs rotation
        if self._index.size >= self.max_index_size:
            logger.info(
                f"Index size ({self._index.size}) reached limit, rotating"
            )
            self._rotate_index()

        return self._index.insert(doc)

    def process(self, doc: NormalizedDocument) -> bool:
        """
        Check and add document in one call.

        Args:
            doc: Document to process

        Returns:
            True if document is unique and was added
        """
        self._total_processed += 1

        # Check if index needs rotation before inserting
        if self._index.size >= self.max_index_size:
            logger.info(
                f"Index size ({self._index.size}) reached limit, rotating"
            )
            self._rotate_index()

        result = self._index.insert_if_unique(doc)

        if result.is_duplicate:
            self._duplicates_found += 1
            return False

        return True

    def process_batch(
        self,
        docs: list[NormalizedDocument],
    ) -> list[NormalizedDocument]:
        """
        Process batch of documents, returning only unique ones.

        Args:
            docs: Documents to process

        Returns:
            List of unique documents
        """
        unique = []
        for doc in docs:
            if self.process(doc):
                unique.append(doc)

        logger.info(
            f"Dedup batch: {len(docs)} input, "
            f"{len(unique)} unique, "
            f"{len(docs) - len(unique)} duplicates"
        )

        return unique

    def _rotate_index(self) -> None:
        """
        Rotate the index by clearing old entries.

        In production, you might want to:
        - Keep a sliding window of recent documents
        - Persist old index to disk
        - Use Redis-backed LSH
        """
        old_size = self._index.size
        self._index.clear()
        logger.info(f"Index rotated: cleared {old_size} documents")

    def clear(self) -> None:
        """Clear index and reset statistics."""
        self._index.clear()
        self._total_processed = 0
        self._duplicates_found = 0
