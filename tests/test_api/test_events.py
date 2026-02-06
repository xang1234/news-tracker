"""Tests for event API endpoints."""

from datetime import datetime, timezone

import pytest

from tests.test_api.conftest import _make_theme


def _make_event_row(
    event_id: str = "evt-1",
    doc_id: str = "doc-1",
    event_type: str = "capacity_expansion",
    actor: str = "TSMC",
    action: str = "is expanding",
    object: str = "fab capacity",
    time_ref: str = "Q3 2026",
    quantity: str | None = None,
    tickers: list[str] | None = None,
    confidence: float = 0.8,
    created_at: datetime | None = None,
) -> dict:
    """Build a mock event row matching repository output."""
    return {
        "event_id": event_id,
        "doc_id": doc_id,
        "event_type": event_type,
        "actor": actor,
        "action": action,
        "object": object,
        "time_ref": time_ref,
        "quantity": quantity,
        "tickers": tickers or ["TSM"],
        "confidence": confidence,
        "span_start": 0,
        "span_end": 50,
        "extractor_version": "1.0",
        "created_at": created_at or datetime(2026, 2, 5, 12, 0, 0, tzinfo=timezone.utc),
    }


class TestGetThemeEvents:
    """Tests for GET /themes/{theme_id}/events."""

    def test_happy_path(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM", "NVDA"])
        mock_theme_repo.get_by_id.return_value = theme

        events = [
            _make_event_row(event_id="e1", tickers=["TSM"], actor="TSMC"),
            _make_event_row(
                event_id="e2", tickers=["NVDA"], event_type="product_launch",
                actor="NVIDIA", action="launched", object="H200 GPU",
            ),
        ]
        mock_doc_repo.get_events_by_tickers.return_value = events

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["theme_id"] == "theme_abc123"
        assert data["total"] == 2
        assert len(data["events"]) == 2
        assert "latency_ms" in data

    def test_404_theme_not_found(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = None

        resp = client.get("/themes/nonexistent/events")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_empty_events(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme
        mock_doc_repo.get_events_by_tickers.return_value = []

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["events"] == []
        assert data["event_counts"] == {}
        assert data["investment_signal"] is None

    def test_event_type_filter(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme
        mock_doc_repo.get_events_by_tickers.return_value = [
            _make_event_row(event_type="capacity_expansion", tickers=["TSM"]),
        ]

        resp = client.get("/themes/theme_abc123/events?event_type=capacity_expansion")

        assert resp.status_code == 200
        # Verify the filter was passed through to the repo
        call_kwargs = mock_doc_repo.get_events_by_tickers.call_args
        assert call_kwargs.kwargs.get("event_type") == "capacity_expansion" or \
               call_kwargs[1].get("event_type") == "capacity_expansion"

    def test_invalid_event_type(self, client, mock_theme_repo):
        theme = _make_theme()
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123/events?event_type=invalid_type")

        assert resp.status_code == 422
        assert "Invalid event_type" in resp.json()["detail"]

    def test_days_param(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme
        mock_doc_repo.get_events_by_tickers.return_value = []

        resp = client.get("/themes/theme_abc123/events?days=30")

        assert resp.status_code == 200
        # Verify since was passed (30 days back)
        call_kwargs = mock_doc_repo.get_events_by_tickers.call_args
        since_arg = call_kwargs.kwargs.get("since") or call_kwargs[1].get("since")
        assert since_arg is not None

    def test_limit_param(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme
        mock_doc_repo.get_events_by_tickers.return_value = []

        resp = client.get("/themes/theme_abc123/events?limit=5")

        assert resp.status_code == 200
        # Over-fetch: limit * 3 = 15
        call_kwargs = mock_doc_repo.get_events_by_tickers.call_args
        limit_arg = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit_arg == 15

    def test_deduplication(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme

        # Two events with same composite key from different docs
        events = [
            _make_event_row(
                event_id="e1", doc_id="d1", actor="TSMC", action="is expanding",
                object="fab capacity", time_ref="Q3 2026", tickers=["TSM"],
            ),
            _make_event_row(
                event_id="e2", doc_id="d2", actor="TSMC", action="is expanding",
                object="fab capacity", time_ref="Q3 2026", tickers=["TSM"],
            ),
        ]
        mock_doc_repo.get_events_by_tickers.return_value = events

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["events"][0]["source_doc_ids"]) == 2

    def test_investment_signal(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme

        events = [
            _make_event_row(
                event_id="e1", doc_id="d1", actor="TSMC",
                event_type="capacity_expansion", tickers=["TSM"],
            ),
            _make_event_row(
                event_id="e2", doc_id="d2", actor="Intel",
                event_type="capacity_expansion", tickers=["TSM"],
            ),
        ]
        mock_doc_repo.get_events_by_tickers.return_value = events

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["investment_signal"] == "supply_increasing"

    def test_event_counts(self, client, mock_theme_repo, mock_doc_repo):
        theme = _make_theme(top_tickers=["TSM", "NVDA"])
        mock_theme_repo.get_by_id.return_value = theme

        events = [
            _make_event_row(
                event_id="e1", doc_id="d1", event_type="capacity_expansion",
                actor="TSMC", tickers=["TSM"],
            ),
            _make_event_row(
                event_id="e2", doc_id="d2", event_type="product_launch",
                actor="NVIDIA", tickers=["NVDA"],
            ),
            _make_event_row(
                event_id="e3", doc_id="d3", event_type="product_launch",
                actor="AMD", tickers=["NVDA"],
            ),
        ]
        mock_doc_repo.get_events_by_tickers.return_value = events

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["event_counts"]["capacity_expansion"] == 1
        assert data["event_counts"]["product_launch"] == 2

    def test_no_tickers_theme(self, client, mock_theme_repo, mock_doc_repo):
        """Theme with empty top_tickers should return empty events."""
        theme = _make_theme(top_tickers=[])
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["events"] == []
        # Repo should not have been called
        mock_doc_repo.get_events_by_tickers.assert_not_called()

    def test_response_field_validation(self, client, mock_theme_repo, mock_doc_repo):
        """Verify all expected fields are present in event items."""
        theme = _make_theme(top_tickers=["TSM"])
        mock_theme_repo.get_by_id.return_value = theme

        events = [_make_event_row(tickers=["TSM"])]
        mock_doc_repo.get_events_by_tickers.return_value = events

        resp = client.get("/themes/theme_abc123/events")

        assert resp.status_code == 200
        event = resp.json()["events"][0]

        expected_fields = {
            "event_id", "doc_id", "event_type", "actor", "action",
            "object", "time_ref", "quantity", "tickers", "confidence",
            "source_doc_ids", "created_at",
        }
        assert expected_fields.issubset(set(event.keys()))

    def test_days_validation_min(self, client, mock_theme_repo):
        """days < 1 should be rejected by FastAPI validation."""
        theme = _make_theme()
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123/events?days=0")

        assert resp.status_code == 422

    def test_days_validation_max(self, client, mock_theme_repo):
        """days > 90 should be rejected by FastAPI validation."""
        theme = _make_theme()
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123/events?days=91")

        assert resp.status_code == 422

    def test_limit_validation_max(self, client, mock_theme_repo):
        """limit > 200 should be rejected by FastAPI validation."""
        theme = _make_theme()
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123/events?limit=201")

        assert resp.status_code == 422
