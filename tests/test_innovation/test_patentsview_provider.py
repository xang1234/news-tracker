"""Tests for USPTO PatentsView/ODP patent ingestion helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from src.innovation.patents import (
    PatentQuery,
    PatentRecord,
    PatentsViewProvider,
    StalePatentSnapshotError,
    deduplicate_patent_families,
    load_patentsview_bulk_snapshot,
)


class FakeHTTPClient:
    def __init__(self, *payloads: dict[str, Any]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return httpx.Response(200, json=self._payloads.pop(0))


def _odp_patent(
    patent_number: str,
    *,
    application_number: str,
    title: str,
    filing_date: str = "2025-01-04",
    grant_date: str | None = "2026-04-02",
    family_id: str | None = None,
) -> dict[str, Any]:
    return {
        "patentNumber": patent_number,
        "applicationNumberText": application_number,
        "applicationMetaData": {
            "inventionTitle": title,
            "filingDate": filing_date,
            "grantDate": grant_date,
            "applicantNameBag": [{"applicantNameText": "NVIDIA Corporation"}],
            "cpcClassificationBag": [{"cpcClassificationText": "G06N 3/04"}],
            "ipcClassificationBag": [{"ipcClassificationText": "G06N"}],
        },
        "familyId": family_id,
        "patentTermAdjustmentData": {"applicationTypeCategory": "Utility"},
    }


@pytest.mark.asyncio
async def test_odp_search_builds_patent_query_and_follows_offset_pagination() -> None:
    client = FakeHTTPClient(
        {
            "patentFileWrapperDataBag": [
                _odp_patent("US123", application_number="18111111", title="GPU scheduler"),
                _odp_patent("US124", application_number="18111112", title="GPU interconnect"),
            ],
            "count": 3,
        },
        {
            "patentFileWrapperDataBag": [
                _odp_patent("US125", application_number="18111113", title="GPU memory fabric"),
            ],
            "count": 3,
        },
    )
    provider = PatentsViewProvider(client, api_key="free-odp-key", page_size=2)

    records = await provider.fetch_patents(
        PatentQuery(
            assignees=["NVIDIA Corporation"],
            cpc_classes=["G06N"],
            ipc_classes=["G06N"],
            keywords=["GPU"],
            start=date(2025, 1, 1),
            end=date(2026, 6, 1),
        ),
        fetched_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
    )

    assert [record.patent_id for record in records] == ["US123", "US124", "US125"]
    assert records[0].source_attribution == "uspto_odp_patentsview_transition"
    assert records[0].assignees == ["NVIDIA Corporation"]
    assert records[0].cpc_classes == ["G06N 3/04"]
    assert client.calls[0]["url"] == "https://api.uspto.gov/api/v1/patent/applications/search"
    assert client.calls[0]["headers"] == {"X-API-KEY": "free-odp-key"}
    assert client.calls[0]["params"]["offset"] == "0"
    assert client.calls[1]["params"]["offset"] == "2"
    query_text = client.calls[0]["params"]["q"]
    assert "applicationMetaData.applicantNameBag.applicantNameText:NVIDIA*" in query_text
    assert "applicationMetaData.cpcClassificationBag.cpcClassificationText:G06N*" in query_text
    assert "applicationMetaData.ipcClassificationBag.ipcClassificationText:G06N*" in query_text
    assert "GPU" in query_text
    assert "applicationMetaData.filingDate:[2025-01-01 TO 2026-06-01]" in query_text


@pytest.mark.asyncio
async def test_odp_search_continues_when_total_count_is_absent() -> None:
    client = FakeHTTPClient(
        {
            "patentFileWrapperDataBag": [
                _odp_patent("US123", application_number="18111111", title="GPU scheduler"),
            ],
        },
        {"patentFileWrapperDataBag": []},
    )
    provider = PatentsViewProvider(client, api_key="free-odp-key", page_size=1)

    records = await provider.fetch_patents(PatentQuery(keywords=["GPU"]))

    assert [record.patent_id for record in records] == ["US123"]
    assert [call["params"]["offset"] for call in client.calls] == ["0", "1"]


def test_bulk_snapshot_loader_rejects_stale_patentsview_exports_by_default() -> None:
    rows = [
        {
            "patent_number": "US999",
            "application_number": "17123456",
            "invention_title": "Chiplet package",
            "filing_date": "2023-02-01",
            "grant_date": "2024-03-15",
            "assignee_organization": "Advanced Micro Devices, Inc.",
            "cpc_group_id": "H01L 23/00",
            "ipc_class": "H01L",
            "family_id": "fam-999",
            "source_url": "https://data.uspto.gov/bulkdata/datasets/pvannual/2024.csv",
        }
    ]

    with pytest.raises(StalePatentSnapshotError, match="PatentsView bulk snapshot is stale"):
        load_patentsview_bulk_snapshot(
            rows,
            snapshot_date=date(2025, 1, 1),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
            max_age_days=120,
        )

    records = load_patentsview_bulk_snapshot(
        rows,
        snapshot_date=date(2025, 1, 1),
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        max_age_days=120,
        allow_stale=True,
    )
    assert records[0].metadata["snapshot_stale"] is True
    assert records[0].metadata["snapshot_age_days"] == 516
    assert records[0].source_attribution == "uspto_patentsview_bulk"


def test_deduplicate_patent_families_prefers_grants_over_duplicate_applications() -> None:
    application = PatentRecord(
        patent_id="",
        application_id="18111111",
        patent_family_id="fam-1",
        title="AI accelerator scheduling",
        abstract="",
        assignees=["NVIDIA Corporation"],
        cpc_classes=["G06N"],
        ipc_classes=[],
        application_date=date(2025, 1, 4),
        grant_date=None,
        source_url="https://data.uspto.gov/application",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    grant = PatentRecord(
        patent_id="US123",
        application_id="18111111",
        patent_family_id="fam-1",
        title="AI accelerator scheduling",
        abstract="",
        assignees=["NVIDIA Corporation"],
        cpc_classes=["G06N"],
        ipc_classes=[],
        application_date=date(2025, 1, 4),
        grant_date=date(2026, 4, 2),
        source_url="https://data.uspto.gov/grant",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    other_family = PatentRecord(
        patent_id="US999",
        application_id="17123456",
        patent_family_id="fam-2",
        title="Chiplet package",
        abstract="",
        assignees=["Advanced Micro Devices, Inc."],
        cpc_classes=["H01L"],
        ipc_classes=[],
        application_date=date(2024, 2, 1),
        grant_date=date(2025, 8, 1),
        source_url="https://data.uspto.gov/grant/999",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    deduped = deduplicate_patent_families([application, grant, other_family])

    assert [record.patent_id for record in deduped] == ["US123", "US999"]
