"""Tests for the Ollama server updater endpoint.

Regression driver (2026-08-29): the compose service for Ollama was switched
from ``image: ollama/ollama:latest`` to ``build: Dockerfile.ollama`` (to
layer ffmpeg on top for the 0.32+ multimodal path). ``compose pull`` then
silently no-ops for that service — the local image ``ocabra-ollama`` isn't
in any registry, but compose returns exit 0 anyway — so ``force-recreate``
brings back the stale local image and the updater reports "done" without
actually updating. The updater must detect ``build:`` and use
``compose build --pull`` instead.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.api.internal import ollama_admin


# ── _service_uses_build_key ─────────────────────────────────


class TestServiceUsesBuildKey:
    @pytest.mark.asyncio
    async def test_returns_true_when_service_has_build(self):
        payload = {
            "services": {
                "ollama": {
                    "build": {"context": ".", "dockerfile": "Dockerfile.ollama"},
                    "image": "ocabra-ollama:latest",
                }
            }
        }
        with patch.object(
            ollama_admin,
            "_run_compose",
            new=AsyncMock(return_value=(0, json.dumps(payload), "")),
        ):
            assert (
                await ollama_admin._service_uses_build_key(("-f", "compose.yml"), "ollama")
                is True
            )

    @pytest.mark.asyncio
    async def test_returns_false_when_service_only_has_image(self):
        payload = {"services": {"ollama": {"image": "ollama/ollama:latest"}}}
        with patch.object(
            ollama_admin,
            "_run_compose",
            new=AsyncMock(return_value=(0, json.dumps(payload), "")),
        ):
            assert (
                await ollama_admin._service_uses_build_key(("-f", "compose.yml"), "ollama")
                is False
            )

    @pytest.mark.asyncio
    async def test_returns_false_on_compose_failure(self):
        """Any parse or subprocess failure must degrade to ``False`` so the
        caller falls back to plain pull (previous behaviour) instead of
        blowing up the update flow."""
        with patch.object(
            ollama_admin,
            "_run_compose",
            new=AsyncMock(return_value=(1, "", "boom")),
        ):
            assert (
                await ollama_admin._service_uses_build_key(("-f", "compose.yml"), "ollama")
                is False
            )

    @pytest.mark.asyncio
    async def test_returns_false_on_unparseable_json(self):
        with patch.object(
            ollama_admin,
            "_run_compose",
            new=AsyncMock(return_value=(0, "not json at all", "")),
        ):
            assert (
                await ollama_admin._service_uses_build_key(("-f", "compose.yml"), "ollama")
                is False
            )

    @pytest.mark.asyncio
    async def test_returns_false_when_service_absent(self):
        payload = {"services": {"api": {"image": "foo"}}}
        with patch.object(
            ollama_admin,
            "_run_compose",
            new=AsyncMock(return_value=(0, json.dumps(payload), "")),
        ):
            assert (
                await ollama_admin._service_uses_build_key(("-f", "compose.yml"), "ollama")
                is False
            )


# ── _run_server_update dispatch ──────────────────────────────


class TestRunServerUpdateDispatch:
    """Verify the updater picks ``build --pull`` vs ``pull`` based on the
    resolved compose config. Mocks all IO — the real update flow needs
    docker + the compose file, which are not available in unit tests.
    """

    @staticmethod
    def _reset_state() -> None:
        ollama_admin._update_state.status = "idle"
        ollama_admin._update_state.detail = None
        ollama_admin._update_state.from_version = None
        ollama_admin._update_state.to_version = None
        ollama_admin._update_state.started_at = None
        ollama_admin._update_state.finished_at = None

    @pytest.mark.asyncio
    async def test_uses_build_pull_when_service_has_build(self):
        self._reset_state()
        calls: list[tuple[str, ...]] = []

        async def fake_run_compose(*args: str) -> tuple[int, str, str]:
            calls.append(args)
            # First call = ``config --format=json`` from _service_uses_build_key
            if "config" in args:
                return (
                    0,
                    json.dumps({"services": {"ollama": {"build": {}}}}),
                    "",
                )
            return (0, "", "")

        registry_mock = AsyncMock()
        registry_mock.get_version = AsyncMock(return_value="0.32.0")
        with (
            patch.object(ollama_admin, "_run_compose", side_effect=fake_run_compose),
            patch.object(ollama_admin, "_registry", registry_mock),
        ):
            await ollama_admin._run_server_update()

        # Expect: config-probe, then ``build --pull ollama``, then ``up``.
        cmds = [a for a in calls if "config" not in a]
        assert any("build" in c and "--pull" in c and "ollama" in c for c in cmds), (
            f"Expected `compose build --pull ollama`; got {cmds}"
        )
        assert not any("pull" in c and "build" not in c for c in cmds), (
            f"Should NOT have plain `compose pull`; got {cmds}"
        )
        assert ollama_admin._update_state.status == "done"

    @pytest.mark.asyncio
    async def test_uses_plain_pull_when_service_only_has_image(self):
        self._reset_state()
        calls: list[tuple[str, ...]] = []

        async def fake_run_compose(*args: str) -> tuple[int, str, str]:
            calls.append(args)
            if "config" in args:
                return (
                    0,
                    json.dumps({"services": {"ollama": {"image": "ollama/ollama:latest"}}}),
                    "",
                )
            return (0, "", "")

        registry_mock = AsyncMock()
        registry_mock.get_version = AsyncMock(return_value="0.32.0")
        with (
            patch.object(ollama_admin, "_run_compose", side_effect=fake_run_compose),
            patch.object(ollama_admin, "_registry", registry_mock),
        ):
            await ollama_admin._run_server_update()

        cmds = [a for a in calls if "config" not in a]
        assert any(
            "pull" in c and "build" not in c and "ollama" in c for c in cmds
        ), f"Expected `compose pull ollama`; got {cmds}"
        assert not any("build" in c for c in cmds), (
            f"Should NOT have `compose build` for a plain-image service; got {cmds}"
        )
        assert ollama_admin._update_state.status == "done"

    @pytest.mark.asyncio
    async def test_reports_failure_when_build_returns_nonzero(self):
        """Regression against the silent-success bug: a failed pull/build must
        surface as ``status=error`` — before, the wrong compose command ran
        successfully-but-uselessly and the updater reported ``done`` even
        though nothing actually happened.
        """
        self._reset_state()

        async def fake_run_compose(*args: str) -> tuple[int, str, str]:
            if "config" in args:
                return (
                    0,
                    json.dumps({"services": {"ollama": {"build": {}}}}),
                    "",
                )
            if "build" in args:
                return (1, "", "dockerfile parse error")
            return (0, "", "")

        registry_mock = AsyncMock()
        registry_mock.get_version = AsyncMock(return_value="0.32.0")
        with (
            patch.object(ollama_admin, "_run_compose", side_effect=fake_run_compose),
            patch.object(ollama_admin, "_registry", registry_mock),
        ):
            await ollama_admin._run_server_update()

        assert ollama_admin._update_state.status == "error"
        assert "dockerfile parse error" in (ollama_admin._update_state.detail or "")
