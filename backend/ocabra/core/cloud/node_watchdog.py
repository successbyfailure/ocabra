"""Self-destruct for cloud nodes that lost their hub.

Runs **on the pod**. The hub's reconcile loop already kills nodes on TTL, idle
and budget — but only while the hub is alive and can reach the provider. If
the hub goes down (or deletes the node's peer), the pod would keep billing
forever. So the node watches its own outgoing tunnel: if no hub has been
connected for ``OCABRA_CLOUD_SELF_DESTRUCT_MINUTES``, it terminates itself
using the pod-scoped ``RUNPOD_API_KEY`` RunPod injects (falling back to
``stop`` if that key may not terminate — the docs don't say).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from ocabra.core.federation import FederationManager

logger = structlog.get_logger(__name__)

CHECK_INTERVAL_S = 60
RUNPOD_API = "https://api.runpod.io/v2"


class NodeSelfDestruct:
    def __init__(self, fm: FederationManager, minutes: int) -> None:
        self.fm = fm
        self.minutes = minutes
        self.pod_id = os.environ.get("RUNPOD_POD_ID", "")
        self.api_key = os.environ.get("RUNPOD_API_KEY", "")
        self._last_connected = time.monotonic()
        self._task: asyncio.Task | None = None

    @classmethod
    def from_env(cls, fm: FederationManager) -> NodeSelfDestruct | None:
        minutes = int(getattr(fm.settings, "ocabra_cloud_self_destruct_minutes", 0) or 0)
        if (
            minutes <= 0
            or not os.environ.get("RUNPOD_POD_ID")
            or not os.environ.get("RUNPOD_API_KEY")
        ):
            return None
        return cls(fm, minutes)

    def start(self) -> None:
        logger.info("cloud_self_destruct_armed", minutes=self.minutes, pod_id=self.pod_id)
        self._task = asyncio.create_task(self._loop(), name="cloud-self-destruct")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def connected(self) -> bool:
        return any(
            p.connection_mode == "tunnel_out" and p.tunnel_connected
            for p in self.fm.get_all_peers()
        )

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_S)
            if self.connected():
                self._last_connected = time.monotonic()
                continue
            lost_for = time.monotonic() - self._last_connected
            if lost_for >= self.minutes * 60:
                logger.warning(
                    "cloud_self_destruct",
                    pod_id=self.pod_id,
                    minutes_without_hub=round(lost_for / 60),
                )
                await self.destroy()

    async def destroy(self) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(base_url=RUNPOD_API, headers=headers, timeout=20) as client:
            for action in ("terminate", "stop"):
                try:
                    resp = await client.post(f"/pods/{self.pod_id}/action", json={"action": action})
                    if resp.status_code < 400:
                        logger.warning("cloud_self_destruct_done", action=action)
                        return True
                    logger.warning(
                        "cloud_self_destruct_refused", action=action, status=resp.status_code
                    )
                except httpx.HTTPError as exc:
                    logger.warning("cloud_self_destruct_error", action=action, error=str(exc))
        return False
