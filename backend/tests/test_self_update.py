"""oCabra updating itself (core/self_update.py)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ocabra.config import settings
from ocabra.core import self_update as su


def test_newest_and_is_newer():
    assert su.newest(["latest", "0.95.6", "v0.96.0", "0.9.10", "sha-abc"]) == "0.96.0"
    assert su.newest(["latest"]) is None
    assert su.is_newer("0.96.0", "0.95.6")
    assert not su.is_newer("0.95.6", "0.95.6")
    assert su.is_newer("0.10.0", "0.9.99")  # numeric, not lexical
    assert not su.is_newer(None, "0.95.6")


def _inspect(image: str) -> str:
    return json.dumps(
        [
            {
                "Name": "/ocabra-api-1",
                "Config": {
                    "Image": image,
                    "Labels": {
                        "com.docker.compose.project": "ocabra",
                        "com.docker.compose.project.working_dir": "/docker/ocabra",
                        "com.docker.compose.project.config_files": "/docker/ocabra/docker-compose.yml",
                    },
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_a_stack_on_release_images_can_update_itself(monkeypatch):
    monkeypatch.setattr(su.os.path, "exists", lambda p: True)
    monkeypatch.delenv("OCABRA_NODE_IMAGE", raising=False)

    async def fake_exec(argv, timeout=120.0):  # noqa: ASYNC109
        assert argv[:2] == ["docker", "inspect"], argv
        return {"code": 0, "stdout": _inspect("ghcr.io/successbyfailure/ocabra-api:0.95.6"), "stderr": ""}

    monkeypatch.setattr("ocabra.docker_agent.ops._exec", fake_exec)
    monkeypatch.setattr("ocabra.core.docker_ops.settings.docker_agent_url", "")
    dep = await su.detect_deployment()
    assert dep.kind == "compose" and dep.can_apply
    assert dep.working_dir == "/docker/ocabra" and dep.project == "ocabra"


@pytest.mark.asyncio
async def test_a_stack_building_its_own_images_only_gets_told(monkeypatch):
    monkeypatch.setattr(su.os.path, "exists", lambda p: True)
    monkeypatch.delenv("OCABRA_NODE_IMAGE", raising=False)

    async def fake_exec(argv, timeout=120.0):  # noqa: ASYNC109
        assert argv[:2] == ["docker", "inspect"], argv
        return {"code": 0, "stdout": _inspect("ocabra-api:slim"), "stderr": ""}

    monkeypatch.setattr("ocabra.docker_agent.ops._exec", fake_exec)
    monkeypatch.setattr("ocabra.core.docker_ops.settings.docker_agent_url", "")
    dep = await su.detect_deployment()
    assert dep.kind == "local_build" and not dep.can_apply
    assert "OCABRA_API_IMAGE" in dep.reason


@pytest.mark.asyncio
async def test_the_node_image_is_replaced_from_the_hub(monkeypatch):
    monkeypatch.setenv("OCABRA_NODE_IMAGE", "1")
    dep = await su.detect_deployment()
    assert dep.kind == "node" and not dep.can_apply


def test_the_updater_script_is_valid_shell_and_does_the_steps():
    dep = su.Deployment(
        "compose",
        project="ocabra",
        working_dir="/docker/ocabra",
        config_files=["/docker/ocabra/docker-compose.yml"],
        api_image="ghcr.io/successbyfailure/ocabra-api:0.95.6",
    )
    script = su.updater_script(dep, "0.96.0", "0.95.6")
    subprocess.run(["sh", "-n"], input=script, text=True, check=True)
    assert "git merge --ff-only --quiet v0.96.0" in script
    assert "OCABRA_API_IMAGE ghcr.io/successbyfailure/ocabra-api:0.96.0" in script
    assert "OCABRA_FRONTEND_IMAGE ghcr.io/successbyfailure/ocabra-frontend:0.96.0" in script
    assert (
        "docker compose -p ocabra -f /docker/ocabra/docker-compose.yml pull $services" in script
    )
    assert '"0.96.0"' in script  # waits for /health to report the new version
    assert "rollback" in script


def _app(inflight: int = 0):
    mm = SimpleNamespace(inflight_snapshot=lambda: {"m": inflight} if inflight else {})
    return SimpleNamespace(state=SimpleNamespace(model_manager=mm))


@pytest.mark.asyncio
async def test_apply_refuses_while_busy(monkeypatch):
    upd = su.SelfUpdater(_app(inflight=2))
    upd._deployment = su.Deployment("compose", project="ocabra", working_dir="/x", api_image="i")

    async def deployment(refresh=False):
        return upd._deployment

    async def busy(app):
        return ["2 peticiones en curso"]

    monkeypatch.setattr(upd, "deployment", deployment)
    monkeypatch.setattr(su, "busy_reasons", busy)
    with pytest.raises(RuntimeError, match="Ocupado"):
        await upd.apply("99.0.0")


@pytest.mark.asyncio
async def test_auto_waits_for_the_idle_window_then_applies(monkeypatch):
    monkeypatch.setattr(settings, "self_update_policy", "auto")
    monkeypatch.setattr(settings, "self_update_idle_minutes", 10)
    upd = su.SelfUpdater(_app())
    upd._deployment = su.Deployment("compose", project="ocabra", working_dir="/x", api_image="i")
    applied: list[str] = []

    async def check(force=False):
        return "99.0.0"

    async def busy(app):
        return []

    async def apply(version=None, *, force=False):
        applied.append(version)

    monkeypatch.setattr(upd, "check", check)
    monkeypatch.setattr(su, "busy_reasons", busy)
    monkeypatch.setattr(upd, "apply", apply)
    clock = [1000.0]
    monkeypatch.setattr(su.time, "time", lambda: clock[0])
    await upd.tick()
    assert upd.state.phase == "waiting_idle" and not applied
    clock[0] += 11 * 60
    await upd.tick()
    assert applied == ["99.0.0"]


@pytest.mark.asyncio
async def test_notify_never_applies(monkeypatch):
    monkeypatch.setattr(settings, "self_update_policy", "notify")
    upd = su.SelfUpdater(_app())

    async def check(force=False):
        return "99.0.0"

    async def apply(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("applied with notify")

    monkeypatch.setattr(upd, "check", check)
    monkeypatch.setattr(upd, "apply", apply)
    await upd.tick()


@pytest.mark.asyncio
async def test_a_finished_updater_closes_the_updating_phase(monkeypatch):
    upd = su.SelfUpdater(_app())
    upd._deployment = su.Deployment("compose", project="ocabra", working_dir="/x", api_image="i")
    upd.state.phase = "updating"
    upd.state.target = "0.98.1"

    async def status():
        return {"running": False, "exit_code": 4, "finished_at": "", "log": ["ROLLBACK"]}

    monkeypatch.setattr(upd, "_updater_status", status)
    snap = await upd.snapshot()
    assert snap["phase"] == "failed" and "se volvió" in snap["message"]


def test_the_updater_keeps_compose_errors_in_its_log():
    dep = su.Deployment(
        "compose", project="ocabra", working_dir="/x", config_files=[], api_image="i"
    )
    assert "exec 2>&1" in su.updater_script(dep, "1.0.1", "1.0.0")


@pytest.mark.asyncio
async def test_auto_does_not_retry_a_release_that_failed(monkeypatch):
    monkeypatch.setattr(settings, "self_update_policy", "auto")
    monkeypatch.setattr(settings, "self_update_idle_minutes", 1)
    upd = su.SelfUpdater(_app())
    upd._deployment = su.Deployment("compose", project="ocabra", working_dir="/x", api_image="i")
    upd.state.failed_target = "99.0.0"
    latest = ["99.0.0"]
    applied: list[str] = []

    async def check(force=False):
        return latest[0]

    async def busy(app):
        return []

    async def status():
        return None

    async def apply(version=None, *, force=False):
        applied.append(version)

    monkeypatch.setattr(upd, "check", check)
    monkeypatch.setattr(upd, "_updater_status", status)
    monkeypatch.setattr(su, "busy_reasons", busy)
    monkeypatch.setattr(upd, "apply", apply)
    clock = [1000.0]
    monkeypatch.setattr(su.time, "time", lambda: clock[0])
    for _ in range(3):
        await upd.tick()
        clock[0] += 120
    assert applied == []
    latest[0] = "99.0.1"  # a newer release is tried again
    await upd.tick()
    clock[0] += 120
    await upd.tick()
    assert applied == ["99.0.1"]


def _run_with_agent(tmp_path, compose_services: str) -> str:
    """Run the updater script's ``with_agent`` with a fake ``docker``."""
    import re
    import stat

    dep = su.Deployment(
        "compose",
        project="ocabra",
        working_dir="/docker/ocabra",
        config_files=["/docker/ocabra/docker-compose.yml"],
        api_image="ghcr.io/successbyfailure/ocabra-api:0.106.0",
    )
    script = su.updater_script(dep, "0.107.0", "0.106.0")
    function = re.search(r"^with_agent\(\) \{\n.*?^\}\n", script, re.S | re.M).group(0)
    fake = tmp_path / "docker"
    fake.write_text(f"#!/bin/sh\nprintf '{compose_services}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    out = subprocess.run(
        ["sh", "-c", function + "with_agent"],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_the_updater_also_recreates_the_docker_agent_when_the_compose_has_it(tmp_path):
    """The agent uses the api image and launches the next updater: left on the
    old image after an update, it would keep the old code for good."""
    assert _run_with_agent(tmp_path, "api\\nfrontend\\ndocker-agent\\n") == (
        "api frontend docker-agent"
    )


def test_the_updater_leaves_the_agent_out_when_the_compose_does_not_define_it(tmp_path):
    """A rollback to a version without the agent must not ask compose for it."""
    assert _run_with_agent(tmp_path, "api\\nfrontend\\n") == "api frontend"
