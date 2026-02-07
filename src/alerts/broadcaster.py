"""WebSocket alert broadcaster using Redis pub/sub for fan-out.

Subscribes to the ``alerts:broadcast`` Redis channel and pushes matching
alerts to connected WebSocket clients based on their filter preferences.
Each API server process runs its own subscriber, so this scales across
multiple uvicorn workers.

Pattern: Background subscriber task + per-client filter matching.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

from src.alerts.schemas import Alert

logger = logging.getLogger(__name__)

CHANNEL_NAME = "alerts:broadcast"


@dataclass
class ClientConnection:
    """A connected WebSocket client with optional filter preferences."""

    ws: WebSocket
    severity: str | None = None
    theme_id: str | None = None
    connected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AlertBroadcaster:
    """Manages WebSocket connections and Redis pub/sub subscription.

    Lifecycle:
        1. ``start(redis_client)`` — subscribe to Redis channel, spawn listener
        2. ``connect(ws, ...)`` / ``disconnect(ws)`` — manage clients
        3. ``stop()`` — cancel background tasks, close pub/sub
    """

    def __init__(
        self,
        max_connections: int = 100,
        heartbeat_interval: int = 30,
    ) -> None:
        self._max_connections = max_connections
        self._heartbeat_interval = heartbeat_interval
        self._clients: dict[WebSocket, ClientConnection] = {}
        self._subscriber_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._pubsub: Any | None = None
        self._running = False

    @property
    def active_connections(self) -> int:
        """Number of currently connected WebSocket clients."""
        return len(self._clients)

    def connect(
        self,
        ws: WebSocket,
        severity: str | None = None,
        theme_id: str | None = None,
    ) -> bool:
        """Register a new WebSocket client.

        Args:
            ws: WebSocket connection.
            severity: Optional severity filter (only receive matching alerts).
            theme_id: Optional theme_id filter.

        Returns:
            True if registered, False if max connections reached.
        """
        if len(self._clients) >= self._max_connections:
            return False

        self._clients[ws] = ClientConnection(
            ws=ws,
            severity=severity,
            theme_id=theme_id,
        )
        logger.info(
            "WebSocket client connected (total=%d, severity=%s, theme_id=%s)",
            len(self._clients), severity, theme_id,
        )
        return True

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket client."""
        removed = self._clients.pop(ws, None)
        if removed:
            logger.info(
                "WebSocket client disconnected (total=%d)", len(self._clients),
            )

    async def start(self, redis_client: Any) -> None:
        """Start the Redis subscriber and heartbeat background tasks.

        Args:
            redis_client: An async Redis client instance.
        """
        if self._running:
            return

        self._running = True

        try:
            self._pubsub = redis_client.pubsub()
            await self._pubsub.subscribe(CHANNEL_NAME)
            self._subscriber_task = asyncio.create_task(
                self._listen(), name="alert-broadcaster-listener",
            )
            self._heartbeat_task = asyncio.create_task(
                self._send_heartbeats(), name="alert-broadcaster-heartbeat",
            )
            logger.info(
                "AlertBroadcaster started (channel=%s, heartbeat=%ds)",
                CHANNEL_NAME, self._heartbeat_interval,
            )
        except Exception as e:
            self._running = False
            logger.error("Failed to start AlertBroadcaster: %s", e)

    async def stop(self) -> None:
        """Stop the subscriber and heartbeat tasks, close pub/sub."""
        self._running = False

        if self._subscriber_task is not None:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
            self._subscriber_task = None

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe(CHANNEL_NAME)
                await self._pubsub.close()
            except Exception as e:
                logger.warning("Error closing pub/sub: %s", e)
            self._pubsub = None

        self._clients.clear()
        logger.info("AlertBroadcaster stopped")

    @staticmethod
    async def publish(redis_client: Any, alert: Alert) -> bool:
        """Publish an alert to the Redis broadcast channel.

        Args:
            redis_client: An async Redis client.
            alert: Alert to broadcast.

        Returns:
            True if published successfully.
        """
        try:
            payload = json.dumps({
                "type": "alert",
                "data": alert.to_dict(),
            })
            await redis_client.publish(CHANNEL_NAME, payload)
            return True
        except Exception as e:
            logger.warning("Failed to publish alert to broadcast channel: %s", e)
            return False

    async def _listen(self) -> None:
        """Background task: read messages from Redis pub/sub and dispatch."""
        try:
            while self._running:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0,
                    )
                    if message is not None and message["type"] == "message":
                        await self._dispatch_message(message["data"])
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("Error reading pub/sub message: %s", e)
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    async def _dispatch_message(self, raw_data: str | bytes) -> None:
        """Parse a pub/sub message and send to matching clients."""
        try:
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")
            payload = json.loads(raw_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Invalid broadcast message: %s", e)
            return

        alert_data = payload.get("data", {})
        alert_severity = alert_data.get("severity")
        alert_theme_id = alert_data.get("theme_id")

        disconnected: list[WebSocket] = []

        for ws, client in list(self._clients.items()):
            if not self._matches_filters(client, alert_severity, alert_theme_id):
                continue

            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    @staticmethod
    def _matches_filters(
        client: ClientConnection,
        alert_severity: str | None,
        alert_theme_id: str | None,
    ) -> bool:
        """Check if an alert matches a client's filter preferences.

        Args:
            client: Connected client with optional filters.
            alert_severity: Severity of the incoming alert.
            alert_theme_id: Theme ID of the incoming alert.

        Returns:
            True if the alert should be sent to this client.
        """
        if client.severity and client.severity != alert_severity:
            return False
        if client.theme_id and client.theme_id != alert_theme_id:
            return False
        return True

    async def _send_heartbeats(self) -> None:
        """Background task: send periodic heartbeat pings to all clients."""
        try:
            while self._running:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._clients:
                    continue

                heartbeat = json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                disconnected: list[WebSocket] = []
                for ws in list(self._clients):
                    try:
                        await ws.send_text(heartbeat)
                    except Exception:
                        disconnected.append(ws)

                for ws in disconnected:
                    self.disconnect(ws)
        except asyncio.CancelledError:
            pass
