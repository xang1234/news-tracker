"""Tests for ClaimRetrievalService orchestration.

Fakes the embedding service and repository so we exercise the wiring —
query → embed (MiniLM) → search → wrap; and the backfill path
list_unembedded → embed_batch → store_embedding — without a model or DB.
"""

from __future__ import annotations

from typing import Any

from src.claims.schemas import EvidenceClaim, make_claim_key
from src.embedding.service import ModelType
from src.retrieval.config import ClaimRetrievalConfig
from src.retrieval.schemas import ClaimRetrievalFilter, RetrievedClaim
from src.retrieval.service import ClaimRetrievalService


def _claim(claim_id: str = "claim_x", **overrides: Any) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", "supplies_to", "NVIDIA")
    base: dict[str, Any] = {
        "claim_id": claim_id,
        "claim_key": key,
        "lane": "narrative",
        "source_id": claim_id,
        "predicate": "supplies_to",
        "subject_text": "TSMC",
        "object_text": "NVIDIA",
        "contract_version": "v1",
    }
    base.update(overrides)
    return EvidenceClaim(**base)


class _FakeEmbedding:
    def __init__(self) -> None:
        self.embed_calls: list[tuple[str, ModelType]] = []
        self.batch_calls: list[tuple[list[str], ModelType]] = []

    async def embed(self, text: str, model_type: ModelType = ModelType.FINBERT) -> list[float]:
        self.embed_calls.append((text, model_type))
        return [0.1, 0.2, 0.3]

    async def embed_batch(
        self,
        texts: list[str],
        model_type: ModelType = ModelType.FINBERT,
        show_progress: bool = False,
    ) -> list[list[float]]:
        self.batch_calls.append((texts, model_type))
        return [[float(i)] for i, _ in enumerate(texts)]


class _FakeRepo:
    def __init__(self, *, search_result=None, unembedded=None) -> None:
        self._search_result = search_result or []
        self._unembedded = unembedded or []
        self.search_args: dict[str, Any] | None = None
        self.stored: list[tuple[str, list[float]]] = []
        self.store_returns = True

    async def search(self, query_embedding, *, limit, threshold, filters):
        self.search_args = {
            "query_embedding": query_embedding,
            "limit": limit,
            "threshold": threshold,
            "filters": filters,
        }
        return self._search_result

    async def list_unembedded(self, limit):
        return self._unembedded[:limit]

    async def store_embedding(self, claim_id, embedding):
        self.stored.append((claim_id, embedding))
        return self.store_returns


class _FakeClaims:
    def __init__(self, claims: dict[str, EvidenceClaim] | None = None) -> None:
        self._claims = claims or {}

    async def get_claim(self, claim_id):
        return self._claims.get(claim_id)


def _service(embedding=None, repo=None, claims=None, config=None) -> ClaimRetrievalService:
    svc = ClaimRetrievalService(
        database=object(),
        embedding_service=embedding or _FakeEmbedding(),
        config=config or ClaimRetrievalConfig(),
    )
    if repo is not None:
        svc._repo = repo
    if claims is not None:
        svc._claims = claims
    return svc


class TestRetrieve:
    async def test_embeds_query_with_minilm_and_wraps_results(self) -> None:
        emb = _FakeEmbedding()
        repo = _FakeRepo(search_result=[(_claim("claim_x"), 0.91), (_claim("claim_y"), 0.7)])
        svc = _service(embedding=emb, repo=repo)

        results = await svc.retrieve("who supplies NVIDIA?")

        assert all(isinstance(r, RetrievedClaim) for r in results)
        assert [(r.claim.claim_id, r.score) for r in results] == [
            ("claim_x", 0.91),
            ("claim_y", 0.7),
        ]
        # Query must be embedded with MiniLM (the 384-dim column model).
        assert emb.embed_calls == [("who supplies NVIDIA?", ModelType.MINILM)]

    async def test_uses_config_default_limit_and_threshold(self) -> None:
        repo = _FakeRepo()
        svc = _service(
            repo=repo, config=ClaimRetrievalConfig(default_limit=5, similarity_threshold=0.42)
        )

        await svc.retrieve("q")

        assert repo.search_args["limit"] == 5
        assert repo.search_args["threshold"] == 0.42

    async def test_explicit_limit_and_filters_passed_through(self) -> None:
        repo = _FakeRepo()
        svc = _service(repo=repo)
        flt = ClaimRetrievalFilter(theme_id="theme_ai")

        await svc.retrieve("q", limit=3, filters=flt)

        assert repo.search_args["limit"] == 3
        assert repo.search_args["filters"] is flt


class TestIndexPending:
    async def test_embeds_batch_and_stores_each(self) -> None:
        emb = _FakeEmbedding()
        repo = _FakeRepo(unembedded=[_claim("claim_a"), _claim("claim_b")])
        svc = _service(embedding=emb, repo=repo)

        count = await svc.index_pending()

        assert count == 2
        assert [cid for cid, _ in repo.stored] == ["claim_a", "claim_b"]
        # Batch embedding uses MiniLM over the composed claim text.
        texts, model = emb.batch_calls[0]
        assert model == ModelType.MINILM
        assert texts == ["TSMC supplies to NVIDIA", "TSMC supplies to NVIDIA"]

    async def test_returns_zero_when_nothing_pending(self) -> None:
        emb = _FakeEmbedding()
        repo = _FakeRepo(unembedded=[])
        svc = _service(embedding=emb, repo=repo)

        assert await svc.index_pending() == 0
        assert emb.batch_calls == []  # no model call when there's nothing to do

    async def test_counts_only_successful_stores(self) -> None:
        repo = _FakeRepo(unembedded=[_claim("claim_a")])
        repo.store_returns = False
        svc = _service(repo=repo)

        assert await svc.index_pending() == 0


class TestIndexClaim:
    async def test_embeds_and_stores_single_claim(self) -> None:
        emb = _FakeEmbedding()
        repo = _FakeRepo()
        claims = _FakeClaims({"claim_x": _claim("claim_x")})
        svc = _service(embedding=emb, repo=repo, claims=claims)

        ok = await svc.index_claim("claim_x")

        assert ok is True
        assert repo.stored == [("claim_x", [0.1, 0.2, 0.3])]
        assert emb.embed_calls == [("TSMC supplies to NVIDIA", ModelType.MINILM)]

    async def test_returns_false_for_unknown_claim(self) -> None:
        repo = _FakeRepo()
        svc = _service(repo=repo, claims=_FakeClaims({}))

        assert await svc.index_claim("ghost") is False
        assert repo.stored == []
