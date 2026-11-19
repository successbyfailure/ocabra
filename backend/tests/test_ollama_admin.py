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


def _compose(fake):
    """Patch the Docker layer so ``docker compose <args>`` calls ``fake(*args)``.

    The compose argv is built by the docker agent's operations
    (ocabra.docker_agent.ops); these tests run them locally (no agent) and
    look at what reaches ``docker compose``.
    """

    async def fake_exec(argv, timeout=120.0):  # noqa: ASYNC109
        assert argv[:2] == ["docker", "compose"], argv
        code, out, err = await fake(*argv[2:])
        return {"code": code, "stdout": out, "stderr": err}

    return patch("ocabra.docker_agent.ops._exec", new=fake_exec)


def _compose_returns(result):
    async def fake(*_args):
        return result

    return _compose(fake)


@pytest.fixture(autouse=True)
def _no_docker_agent(monkeypatch):
    monkeypatch.setattr("ocabra.core.docker_ops.settings.docker_agent_url", "")


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
        with _compose_returns((0, json.dumps(payload), "")):
            assert await ollama_admin._service_uses_build_key() is True

    @pytest.mark.asyncio
    async def test_returns_false_when_service_only_has_image(self):
        payload = {"services": {"ollama": {"image": "ollama/ollama:latest"}}}
        with _compose_returns((0, json.dumps(payload), "")):
            assert await ollama_admin._service_uses_build_key() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_compose_failure(self):
        """Any parse or subprocess failure must degrade to ``False`` so the
        caller falls back to plain pull (previous behaviour) instead of
        blowing up the update flow."""
        with _compose_returns((1, "", "boom")):
            assert await ollama_admin._service_uses_build_key() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_unparseable_json(self):
        with _compose_returns((0, "not json at all", "")):
            assert await ollama_admin._service_uses_build_key() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_service_absent(self):
        payload = {"services": {"api": {"image": "foo"}}}
        with _compose_returns((0, json.dumps(payload), "")):
            assert await ollama_admin._service_uses_build_key() is False


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
            _compose(fake_run_compose),
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
            _compose(fake_run_compose),
            patch.object(ollama_admin, "_registry", registry_mock),
        ):
            await ollama_admin._run_server_update()

        cmds = [a for a in calls if "config" not in a]
        assert any("pull" in c and "build" not in c and "ollama" in c for c in cmds), (
            f"Expected `compose pull ollama`; got {cmds}"
        )
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
            _compose(fake_run_compose),
            patch.object(ollama_admin, "_registry", registry_mock),
        ):
            await ollama_admin._run_server_update()

        assert ollama_admin._update_state.status == "error"
        assert "dockerfile parse error" in (ollama_admin._update_state.detail or "")


# ── Concurrent POSTs ────────────────────────────────────────


class TestConcurrentStart:
    """Two POSTs racing used to both see ``idle`` (the task sets ``pulling``
    only once it runs) and the update ran twice; the second must get 409."""

    @staticmethod
    def _reset() -> None:
        ollama_admin._update_state.status = "idle"
        ollama_admin._apply_state.status = "idle"

    @staticmethod
    def _fake_spawn(spawned: list[str]):
        def fake_spawn(coro, name=None):
            coro.close()
            spawned.append(name)

        return fake_spawn

    async def test_second_update_post_gets_409(self, monkeypatch):
        self._reset()
        spawned: list[str] = []
        monkeypatch.setattr(ollama_admin, "spawn", self._fake_spawn(spawned))
        try:
            first = await ollama_admin.start_server_update(_user=None)
            with pytest.raises(ollama_admin.HTTPException) as exc:
                await ollama_admin.start_server_update(_user=None)
            assert first.status == "pulling"
            assert exc.value.status_code == 409
            assert spawned == ["ollama-server-update"]
            # Nor can the settings be applied while the update holds the container.
            with pytest.raises(ollama_admin.HTTPException):
                await ollama_admin.apply_server_settings(_user=None)
        finally:
            self._reset()

    async def test_second_apply_post_gets_409(self, monkeypatch):
        self._reset()
        spawned: list[str] = []
        monkeypatch.setattr(ollama_admin, "spawn", self._fake_spawn(spawned))
        try:
            await ollama_admin.apply_server_settings(_user=None)
            for start in (ollama_admin.apply_server_settings, ollama_admin.start_server_update):
                with pytest.raises(ollama_admin.HTTPException) as exc:
                    await start(_user=None)
                assert exc.value.status_code == 409
            assert spawned == ["ollama-apply-settings"]
        finally:
            self._reset()
