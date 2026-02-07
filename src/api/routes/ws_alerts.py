"""WebSocket endpoint for real-time alert streaming.

Clients connect to ``/ws/alerts`` with optional query parameters for
filtering. Alerts are pushed as JSON messages via Redis pub/sub fan-out.

Auth is via ``api_key`` query parameter since browsers cannot set
custom headers on WebSocket upgrade requests.
"""

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.alerts.broadcaster import AlertBroadcaster
from src.alerts.schemas import VALID_SEVERITIES
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level broadcaster reference, set during app lifespan
_broadcaster: AlertBroadcaster | None = None


def set_broadcaster(broadcaster: AlertBroadcaster) -> None:
    """Set the module-level broadcaster (called during app startup)."""
    global _broadcaster
    _broadcaster = broadcaster


def get_broadcaster() -> AlertBroadcaster | None:
    """Get the current broadcaster instance."""
    return _broadcaster


def _validate_api_key(api_key: str | None) -> bool:
    """Validate an API key for WebSocket connections.

    Same logic as ``verify_api_key`` but adapted for query-param auth
    (no FastAPI dependency injection in WebSocket handlers).

    Args:
        api_key: Key from query parameter.

    Returns:
        True if valid or dev mode (no keys configured).
    """
    settings = get_settings()

    # Dev mode: no keys configured → allow all
    if not settings.api_keys:
        return True

    if api_key is None:
        return False

    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]
    return bool(valid_keys) and api_key in valid_keys


@router.websocket("/ws/alerts")
async def ws_alerts(
    ws: WebSocket,
    severity: str | None = Query(default=None),
    theme_id: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
) -> None:
    """WebSocket endpoint for real-time alert streaming.

    Query parameters:
        severity: Filter by alert severity (critical, warning, info).
        theme_id: Filter by theme identifier.
        api_key: API key for authentication.
    """
    settings = get_settings()

    # Check feature flag
    if not settings.ws_alerts_enabled:
        await ws.close(code=1008, reason="WebSocket alerts not enabled")
        return

    # Validate auth
    if not _validate_api_key(api_key):
        await ws.close(code=1008, reason="Invalid or missing API key")
        return

    # Validate severity filter
    if severity and severity not in VALID_SEVERITIES:
        await ws.close(code=1008, reason=f"Invalid severity: {severity}")
        return

    broadcaster = _broadcaster
    if broadcaster is None:
        await ws.close(code=1011, reason="Broadcaster not available")
        return

    # Accept the connection
    await ws.accept()

    # Register with broadcaster
    if not broadcaster.connect(ws, severity=severity, theme_id=theme_id):
        await ws.close(code=1008, reason="Max connections reached")
        return

    try:
        # Keep connection alive — listen for client messages
        while True:
            try:
                raw = await ws.receive_text()
                # Accept client-initiated pings
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                except (json.JSONDecodeError, TypeError):
                    pass
            except WebSocketDisconnect:
                break
    finally:
        broadcaster.disconnect(ws)
