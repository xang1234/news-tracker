"""
Document deduplication using MinHash LSH.

Identifies near-duplicate documents across platforms using Locality
Sensitive Hashing. This catches:
- Copy-paste campaigns on social media
- Syndicated news articles from wire services
- Reposts with minor modifications

Uses the datasketch library for efficient MinHash LSH implementation.
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field

import redis.asyncio as redis
from asyncpg.exceptions import UniqueViolationError
from datasketch import MinHash, MinHashLSH

from src.config.settings import get_settings
from src.ingestion.schemas import NormalizedDocument
from src.storage.database import Database
from src.storage.repository import DocumentRepository

logger = logging.getLogger(__name__)

_COMPARE_AND_DELETE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


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
        self.threshold = settings.duplicate_threshold if threshold is None else threshold
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
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        words = text.split()

        # Use 3-word shingles (n-grams)
        # This captures phrase-level similarity
        shingle_size = 3
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i : i + shingle_size])
            m.update(shingle.encode("utf-8"))

        # Also add individual words for short texts
        if len(words) < shingle_size * 2:
            for word in words:
                m.update(word.encode("utf-8"))

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
            logger.debug(f"Duplicate detected: {doc.id} similar to {result.similar_doc_ids}")
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
            logger.debug(f"Duplicate: {doc.id} similar to {result.similar_doc_ids[:3]}")

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
            logger.info(f"Index size ({self._index.size}) reached limit, rotating")
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
            logger.info(f"Index size ({self._index.size}) reached limit, rotating")
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


@dataclass
class SharedDuplicateResult:
    """Result of shared duplicate detection plus optional storage."""

    is_duplicate: bool
    canonical_document_id: str | None = None
    duplicate_type: str | None = None
    similar_doc_ids: list[str] = field(default_factory=list)
    similarity_score: float | None = None
    stored: bool = False


class SharedDeduplicator:
    """Redis-backed, persisted duplicate suppression shared across workers."""

    def __init__(
        self,
        database: Database,
        *,
        redis_url: str | None = None,
        redis_client: redis.Redis | None = None,
        threshold: float | None = None,
        num_perm: int = 128,
        bands: int = 32,
        lock_ttl_seconds: int = 30,
        redis_prefix: str = "dedupe:lsh",
    ) -> None:
        settings = get_settings()
        self._database = database
        self._redis_url = redis_url or str(settings.redis_url)
        self._redis = redis_client
        self._owns_redis = redis_client is None
        self.threshold = settings.duplicate_threshold if threshold is None else threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = max(1, num_perm // bands)
        self.lock_ttl_seconds = lock_ttl_seconds
        self.redis_prefix = redis_prefix
        self._warm_key = f"{self.redis_prefix}:warm"
        self._warm_lock_key = f"{self.redis_prefix}:warm_lock"
        self._total_processed = 0
        self._duplicates_found = 0
        self._indexed_documents = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_processed": self._total_processed,
            "duplicates_found": self._duplicates_found,
            "index_size": self._indexed_documents,
        }

    async def connect(self) -> None:
        """Initialize the Redis client when the service starts."""
        if self._redis is not None:
            return
        self._redis = redis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def close(self) -> None:
        """Close the owned Redis client."""
        if self._redis is not None and self._owns_redis:
            await self._redis.close()
            self._redis = None

    async def store_if_unique(
        self,
        doc: NormalizedDocument,
        *,
        repository: DocumentRepository,
    ) -> SharedDuplicateResult:
        """Check a document against shared state and persist it if unique."""
        self._total_processed += 1
        await self.connect()
        await self._ensure_index_warm(repository)

        exact_fingerprint = self.build_exact_fingerprint(doc)
        signature = self.create_signature(doc.content)
        bucket_keys = self._bucket_keys(signature)
        lock_tokens = await self._acquire_locks(bucket_keys)

        try:
            async with self._database.transaction() as conn:
                exact_match = await repository.get_dedup_signature_by_exact_fingerprint(
                    exact_fingerprint,
                    conn=conn,
                )
                if exact_match is not None:
                    self._duplicates_found += 1
                    return SharedDuplicateResult(
                        is_duplicate=True,
                        canonical_document_id=str(exact_match["canonical_document_id"]),
                        duplicate_type="exact",
                        similar_doc_ids=[str(exact_match["document_id"])],
                    )

                candidate_ids = await self._candidate_document_ids(signature)
                candidate_rows = await repository.fetch_dedup_signatures(
                    sorted(candidate_ids),
                    conn=conn,
                )
                near_match = self._find_nearest_match(signature, candidate_rows)
                if near_match is not None:
                    self._duplicates_found += 1
                    return SharedDuplicateResult(
                        is_duplicate=True,
                        canonical_document_id=near_match["canonical_document_id"],
                        duplicate_type="near",
                        similar_doc_ids=[near_match["document_id"]],
                        similarity_score=near_match["similarity_score"],
                    )

                await repository.insert(doc, conn=conn)
                await repository.upsert_dedup_signature(
                    document_id=doc.id,
                    canonical_document_id=doc.id,
                    exact_fingerprint=exact_fingerprint,
                    minhash_signature=self.serialize_signature(signature),
                    conn=conn,
                )
            await self._index_signature(doc.id, signature)
            self._indexed_documents += 1
            return SharedDuplicateResult(is_duplicate=False, stored=True)
        except UniqueViolationError:
            exact_match = await repository.get_dedup_signature_by_exact_fingerprint(exact_fingerprint)
            self._duplicates_found += 1
            canonical_document_id = (
                str(exact_match["canonical_document_id"]) if exact_match is not None else None
            )
            similar_doc_ids = [str(exact_match["document_id"])] if exact_match is not None else []
            return SharedDuplicateResult(
                is_duplicate=True,
                canonical_document_id=canonical_document_id,
                duplicate_type="exact",
                similar_doc_ids=similar_doc_ids,
            )
        finally:
            await self._release_locks(lock_tokens)

    async def _ensure_index_warm(self, repository: DocumentRepository) -> None:
        """Hydrate Redis bucket sets from persisted signatures on first use."""
        redis_client = self._redis_client
        if await redis_client.exists(self._warm_key):
            return

        warm_token = f"warm-{uuid.uuid4().hex}"
        acquired = await redis_client.set(
            self._warm_lock_key,
            warm_token,
            nx=True,
            ex=self.lock_ttl_seconds,
        )
        if not acquired:
            for _ in range(20):
                if await redis_client.exists(self._warm_key):
                    return
                await asyncio.sleep(0.1)
            rows = await repository.list_dedup_signatures()
            self._indexed_documents = len(rows)
            for row in rows:
                await self._index_signature(
                    str(row["document_id"]),
                    self.deserialize_signature(row["minhash_signature"]),
                )
            return

        try:
            rows = await repository.list_dedup_signatures()
            self._indexed_documents = len(rows)
            for row in rows:
                await self._index_signature(
                    str(row["document_id"]),
                    self.deserialize_signature(row["minhash_signature"]),
                )
            await redis_client.set(self._warm_key, "1", ex=86_400)
        finally:
            await self._release_lock(self._warm_lock_key, warm_token)

    async def _candidate_document_ids(self, signature: list[int]) -> set[str]:
        redis_client = self._redis_client
        candidate_ids: set[str] = set()
        for bucket_key in self._bucket_keys(signature):
            candidate_ids.update(await redis_client.smembers(bucket_key))
        return candidate_ids

    async def _index_signature(self, document_id: str, signature: list[int]) -> None:
        redis_client = self._redis_client
        for bucket_key in self._bucket_keys(signature):
            await redis_client.sadd(bucket_key, document_id)

    async def _acquire_locks(self, bucket_keys: list[str]) -> dict[str, str]:
        redis_client = self._redis_client
        lock_tokens: dict[str, str] = {}
        for bucket_key in sorted(set(bucket_keys)):
            lock_key = f"{bucket_key}:lock"
            token = uuid.uuid4().hex
            acquired = False
            for _ in range(20):
                acquired = bool(
                    await redis_client.set(
                        lock_key,
                        token,
                        nx=True,
                        ex=self.lock_ttl_seconds,
                    )
                )
                if acquired:
                    lock_tokens[lock_key] = token
                    break
                await asyncio.sleep(0.05)
            if not acquired:
                await self._release_locks(lock_tokens)
                raise RuntimeError(f"Failed to acquire dedupe lock for {lock_key}")
        return lock_tokens

    async def _release_locks(self, lock_tokens: dict[str, str]) -> None:
        if self._redis is None:
            return
        for lock_key, token in lock_tokens.items():
            await self._release_lock(lock_key, token)

    async def _release_lock(self, lock_key: str, token: str) -> int:
        """Release a Redis lock only if we still own it."""
        if self._redis is None:
            return 0
        return int(await self._redis.eval(_COMPARE_AND_DELETE_LUA, 1, lock_key, token))

    def build_exact_fingerprint(self, doc: NormalizedDocument) -> str:
        """Build a deterministic exact fingerprint across workers and restarts."""
        payload = {
            "content_type": doc.content_type,
            "title": self._normalize_for_exact(doc.title or ""),
            "content": self._normalize_for_exact(doc.content),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def create_signature(self, text: str) -> list[int]:
        """Create a MinHash signature ready for JSON persistence."""
        minhash = MinHash(num_perm=self.num_perm)
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        words = text.split()
        shingle_size = 3
        for index in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[index : index + shingle_size])
            minhash.update(shingle.encode("utf-8"))
        if len(words) < shingle_size * 2:
            for word in words:
                minhash.update(word.encode("utf-8"))
        return [int(value) for value in minhash.hashvalues]

    def serialize_signature(self, signature: list[int]) -> list[str]:
        """Convert signature integers to JSON-safe strings."""
        return [str(value) for value in signature]

    def deserialize_signature(self, value: object) -> list[int]:
        """Parse a stored signature row back into integers."""
        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value
        if not isinstance(parsed, list):
            return []
        return [int(item) for item in parsed]

    def _bucket_keys(self, signature: list[int]) -> list[str]:
        """Compute deterministic Redis bucket keys for a signature."""
        keys: list[str] = []
        for band_index in range(self.bands):
            start = band_index * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            if not band:
                continue
            band_digest = hashlib.sha1(
                ":".join(str(value) for value in band).encode("utf-8")
            ).hexdigest()
            keys.append(f"{self.redis_prefix}:band:{band_index}:{band_digest}")
        return keys

    def _find_nearest_match(
        self,
        signature: list[int],
        candidate_rows: list[dict[str, object]],
    ) -> dict[str, object] | None:
        best_match: dict[str, object] | None = None
        best_score = 0.0
        for row in candidate_rows:
            candidate_signature = self.deserialize_signature(row["minhash_signature"])
            if not candidate_signature:
                continue
            similarity = self._estimate_similarity(signature, candidate_signature)
            if similarity < self.threshold or similarity < best_score:
                continue
            best_score = similarity
            best_match = {
                "document_id": str(row["document_id"]),
                "canonical_document_id": str(row["canonical_document_id"]),
                "similarity_score": similarity,
            }
        return best_match

    @staticmethod
    def _estimate_similarity(left: list[int], right: list[int]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        matches = sum(
            1
            for left_value, right_value in zip(left, right, strict=False)
            if left_value == right_value
        )
        return matches / len(left)

    @staticmethod
    def _normalize_for_exact(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @property
    def _redis_client(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("SharedDeduplicator is not connected to Redis")
        return self._redis
