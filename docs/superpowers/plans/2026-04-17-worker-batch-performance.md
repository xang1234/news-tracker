# Worker Batch Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove obvious hot-path round-trips in the embedding and sentiment workers by bulk-loading documents once per batch and using true batched sentiment inference for document-level jobs.

**Architecture:** Add one bulk-read API to `DocumentRepository`, then switch both workers to use it instead of calling `get_by_id()` inside loops. Keep sentiment behavior stable by splitting each batch into two lanes: entity-sentiment documents stay on the existing `analyze_with_entities()` path, while document-level-only jobs flow through the already-implemented `SentimentService.analyze_batch()` API.

**Tech Stack:** Python 3.12, async repository layer, Redis Streams workers, FinBERT sentiment service, pytest, AsyncMock

---

### Task 1: Add a Bulk Document Fetch API

**Files:**
- Create: `tests/test_storage/test_repository_bulk_fetch.py`
- Modify: `src/storage/repository.py`
- Test: `tests/test_storage/test_repository_bulk_fetch.py`

- [ ] **Step 1: Write the failing repository test**

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.storage.repository import DocumentRepository


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    return db


@pytest.fixture
def repo(mock_db):
    return DocumentRepository(mock_db)


def _row(doc_id: str) -> dict:
    now = datetime.now(UTC)
    return {
        "id": doc_id,
        "platform": Platform.NEWS.value,
        "url": f"https://example.com/{doc_id}",
        "timestamp": now,
        "author_id": "author-1",
        "author_name": "Author",
        "content": f"content-{doc_id}",
        "content_type": "article",
        "title": f"title-{doc_id}",
        "engagement": {"likes": 1, "shares": 0, "comments": 0},
        "tickers": [],
        "entities_mentioned": [],
        "keywords": [],
        "embedding": None,
        "embedding_minilm": None,
        "theme_ids": [],
        "raw_data": None,
        "sentiment": None,
        "events_extracted": [],
        "urls_mentioned": [],
        "spam_score": None,
        "bot_probability": None,
        "authority_score": None,
        "created_at": now,
        "updated_at": now,
        "fetched_at": now,
    }


@pytest.mark.asyncio
async def test_get_by_ids_uses_any_query_and_preserves_requested_order(repo, mock_db):
    mock_db.fetch.return_value = [_row("doc_2"), _row("doc_1")]

    docs = await repo.get_by_ids(["doc_1", "missing", "doc_2"])

    assert [doc.id for doc in docs] == ["doc_1", "doc_2"]
    sql, ids = mock_db.fetch.call_args.args
    assert "WHERE id = ANY($1::text[])" in sql
    assert ids == ["doc_1", "missing", "doc_2"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/test_storage/test_repository_bulk_fetch.py::test_get_by_ids_uses_any_query_and_preserves_requested_order`

Expected: FAIL with `AttributeError: 'DocumentRepository' object has no attribute 'get_by_ids'`

- [ ] **Step 3: Add `DocumentRepository.get_by_ids()`**

```python
async def get_by_ids(self, doc_ids: list[str]) -> list[NormalizedDocument]:
    """
    Fetch multiple documents by ID while preserving the requested order.

    Missing IDs are ignored in the returned list.
    """
    if not doc_ids:
        return []

    sql = "SELECT * FROM documents WHERE id = ANY($1::text[])"
    rows = await self._db.fetch(sql, doc_ids)

    by_id = {row["id"]: self._row_to_document(row) for row in rows}
    return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]
```

- [ ] **Step 4: Run the repository test again**

Run: `uv run pytest -q tests/test_storage/test_repository_bulk_fetch.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_storage/test_repository_bulk_fetch.py src/storage/repository.py
git commit -m "feat: add bulk document repository fetch"
```

### Task 2: Switch the Embedding Worker to One Repository Call Per Batch

**Files:**
- Modify: `src/embedding/worker.py`
- Modify: `tests/test_embedding/test_worker.py`
- Test: `tests/test_embedding/test_worker.py`

- [ ] **Step 1: Write the failing worker test**

Add this test to `tests/test_embedding/test_worker.py`:

```python
@pytest.mark.asyncio
async def test_fetch_documents_uses_bulk_repository_call(
    mock_database,
    mock_embedding_service_for_worker,
    sample_documents,
):
    worker = EmbeddingWorker(
        database=mock_database,
        embedding_service=mock_embedding_service_for_worker,
    )

    mock_repo = AsyncMock(spec=DocumentRepository)
    mock_repo.get_by_ids = AsyncMock(return_value=sample_documents[:2])
    mock_repo.get_by_id = AsyncMock()
    worker._repository = mock_repo

    docs = await worker._fetch_documents(["doc_0", "doc_1"])

    assert [doc.id for doc in docs] == ["doc_0", "doc_1"]
    mock_repo.get_by_ids.assert_awaited_once_with(["doc_0", "doc_1"])
    mock_repo.get_by_id.assert_not_awaited()
```

- [ ] **Step 2: Run the embedding-worker test to verify it fails**

Run: `uv run pytest -q tests/test_embedding/test_worker.py -k bulk_repository_call`

Expected: FAIL because `_fetch_documents()` still calls `get_by_id()`

- [ ] **Step 3: Update the embedding worker to use `get_by_ids()`**

Replace the current `_fetch_documents()` body in `src/embedding/worker.py` with:

```python
async def _fetch_documents(self, doc_ids: list[str]) -> list[Any]:
    """Fetch documents from database by IDs."""
    if self._repository is None:
        return []
    return await self._repository.get_by_ids(doc_ids)
```

- [ ] **Step 4: Run the focused worker tests**

Run: `uv run pytest -q tests/test_embedding/test_worker.py -k "bulk_repository_call or process_batch_success or document_not_found"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/worker.py tests/test_embedding/test_worker.py
git commit -m "perf: batch document loads in embedding worker"
```

### Task 3: Batch Sentiment Worker Fetches and Document-Level Inference

**Files:**
- Create: `tests/test_sentiment/test_worker.py`
- Modify: `src/sentiment/worker.py`
- Test: `tests/test_sentiment/test_worker.py`

- [ ] **Step 1: Write the failing sentiment-worker tests**

Create `tests/test_sentiment/test_worker.py` with:

```python
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.sentiment.config import SentimentConfig
from src.sentiment.queue import SentimentJob, SentimentQueue
from src.sentiment.worker import SentimentWorker
from src.storage.database import Database
from src.storage.repository import DocumentRepository


@pytest.fixture
def sample_documents():
    now = datetime.now(UTC)
    return [
        NormalizedDocument(
            id="doc_plain",
            platform=Platform.NEWS,
            url="https://example.com/plain",
            timestamp=now,
            author_id="author-1",
            author_name="Author",
            content="Revenue beat expectations.",
            content_type="article",
            title="Plain doc",
            engagement=EngagementMetrics(likes=1, shares=0, comments=0),
        ),
        NormalizedDocument(
            id="doc_entity",
            platform=Platform.NEWS,
            url="https://example.com/entity",
            timestamp=now,
            author_id="author-2",
            author_name="Author",
            content="NVIDIA beat estimates and AMD guided higher.",
            content_type="article",
            title="Entity doc",
            engagement=EngagementMetrics(likes=1, shares=0, comments=0),
            entities_mentioned=[
                {"text": "NVIDIA", "normalized": "NVIDIA", "type": "COMPANY", "start": 0, "end": 6}
            ],
        ),
    ]


@pytest.fixture
def mock_queue():
    queue = AsyncMock(spec=SentimentQueue)
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    return queue


@pytest.fixture
def mock_database():
    db = AsyncMock(spec=Database)
    return db


@pytest.mark.asyncio
async def test_fetch_documents_uses_bulk_repository_call(
    mock_queue,
    mock_database,
    sample_documents,
):
    worker = SentimentWorker(queue=mock_queue, database=mock_database)
    repo = AsyncMock(spec=DocumentRepository)
    repo.get_by_ids = AsyncMock(return_value=sample_documents)
    repo.get_by_id = AsyncMock()
    worker._repository = repo

    docs = await worker._fetch_documents(["doc_plain", "doc_entity"])

    assert [doc.id for doc in docs] == ["doc_plain", "doc_entity"]
    repo.get_by_ids.assert_awaited_once_with(["doc_plain", "doc_entity"])
    repo.get_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_batch_uses_analyze_batch_for_document_level_jobs(
    mock_queue,
    mock_database,
    sample_documents,
):
    config = SentimentConfig(batch_size=8, enable_entity_sentiment=True)
    service = AsyncMock()
    service.analyze_batch = AsyncMock(
        return_value=[
            {"label": "positive", "confidence": 0.9, "scores": {"positive": 0.9}, "model": "finbert"}
        ]
    )
    service.analyze_with_entities = AsyncMock(
        return_value={
            "label": "positive",
            "confidence": 0.8,
            "scores": {"positive": 0.8},
            "model": "finbert",
            "entity_sentiments": [],
        }
    )

    worker = SentimentWorker(
        queue=mock_queue,
        database=mock_database,
        sentiment_service=service,
        config=config,
    )

    repo = AsyncMock(spec=DocumentRepository)
    repo.get_by_ids = AsyncMock(return_value=sample_documents)
    repo.update_sentiment = AsyncMock(return_value=True)
    worker._repository = repo

    batch = [
        SentimentJob(message_id="msg-1", document_id="doc_plain"),
        SentimentJob(message_id="msg-2", document_id="doc_entity"),
    ]

    await worker._process_batch(batch)

    service.analyze_batch.assert_awaited_once()
    batch_texts = service.analyze_batch.await_args.args[0]
    assert batch_texts == ["Plain doc. Revenue beat expectations."]
    service.analyze_with_entities.assert_awaited_once()
    assert repo.update_sentiment.await_count == 2
    assert mock_queue.ack.await_count == 2
```

- [ ] **Step 2: Run the new sentiment-worker tests to verify they fail**

Run: `uv run pytest -q tests/test_sentiment/test_worker.py`

Expected: FAIL because `_fetch_documents()` still calls `get_by_id()` and `_process_batch()` still calls `analyze()` once per document

- [ ] **Step 3: Switch the sentiment worker to `get_by_ids()`**

Replace `_fetch_documents()` in `src/sentiment/worker.py` with:

```python
async def _fetch_documents(self, doc_ids: list[str]) -> list[Any]:
    """Fetch documents from database by IDs."""
    if not self._repository:
        return []
    return await self._repository.get_by_ids(doc_ids)
```

- [ ] **Step 4: Refactor `_process_batch()` into entity and document-only lanes**

Update the body of `_process_batch()` so the document-level branch batches first:

```python
plain_jobs: list[tuple[SentimentJob, Any, str]] = []
entity_jobs: list[tuple[SentimentJob, Any, str]] = []

for job in batch:
    doc = doc_map.get(job.document_id)
    if doc is None:
        skipped += 1
        await self._queue.ack(job.message_id)
        continue
    if doc.sentiment is not None:
        skipped += 1
        await self._queue.ack(job.message_id)
        continue
    if not doc.content or not doc.content.strip():
        skipped += 1
        await self._queue.ack(job.message_id)
        continue

    text = f"{doc.title}. {doc.content}" if doc.title else doc.content
    if doc.entities_mentioned and self._config.enable_entity_sentiment:
        entity_jobs.append((job, doc, text))
    else:
        plain_jobs.append((job, doc, text))

if plain_jobs:
    batch_results = await self._sentiment_service.analyze_batch(
        [text for _, _, text in plain_jobs]
    )
    for (job, doc, _text), result in zip(plain_jobs, batch_results, strict=True):
        success = await self._repository.update_sentiment(job.document_id, result)
        if success:
            processed += 1
        else:
            errors += 1
        await self._queue.ack(job.message_id)

for job, doc, text in entity_jobs:
    result = await self._sentiment_service.analyze_with_entities(
        text, doc.entities_mentioned
    )
    success = await self._repository.update_sentiment(job.document_id, result)
    if success:
        processed += 1
    else:
        errors += 1
    await self._queue.ack(job.message_id)
```

Keep the existing warning/error logging and metrics calls while moving the control flow to these two lanes. Do not change queue semantics: every handled job still ends in `ack()`, and exceptions still `nack()` the job.

- [ ] **Step 5: Run the focused sentiment tests**

Run: `uv run pytest -q tests/test_sentiment/test_worker.py tests/test_sentiment/test_service.py -k "worker or analyze_batch"`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sentiment/worker.py tests/test_sentiment/test_worker.py
git commit -m "perf: batch document-level sentiment inference"
```

### Task 4: Final Verification and Hand-Off

**Files:**
- Modify: none
- Test: `tests/test_storage/test_repository_bulk_fetch.py`
- Test: `tests/test_embedding/test_worker.py`
- Test: `tests/test_sentiment/test_worker.py`
- Test: `tests/test_sentiment/test_service.py`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
uv run pytest -q \
  tests/test_storage/test_repository_bulk_fetch.py \
  tests/test_embedding/test_worker.py \
  tests/test_sentiment/test_worker.py \
  tests/test_sentiment/test_service.py
```

Expected: PASS

- [ ] **Step 2: Run lint on touched files**

Run:

```bash
uv run ruff check \
  src/storage/repository.py \
  src/embedding/worker.py \
  src/sentiment/worker.py \
  tests/test_storage/test_repository_bulk_fetch.py \
  tests/test_embedding/test_worker.py \
  tests/test_sentiment/test_worker.py
```

Expected: PASS

- [ ] **Step 3: Capture before/after behavior notes**

Record these measurements in the PR description or commit notes:

```text
- Embedding worker document loads: 1 repository call per batch instead of N
- Sentiment worker document loads: 1 repository call per batch instead of N
- Sentiment inference: 1 analyze_batch() call for document-level jobs per batch
```

- [ ] **Step 4: Final sync**

```bash
git status
bd sync
git pull --rebase
git push
git status
```

Expected: clean working tree and branch up to date with origin
