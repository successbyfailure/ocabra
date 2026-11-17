"""The docker agent (SEG-03): only oCabra's own operations, validated, behind a token.

Before, the api ran the Docker CLI with the socket mounted, so whoever
controlled it could ``docker run -v /:/host``. These tests pin what the agent
refuses: anything that would let a caller pick an image, a mount, a container
or service outside oCabra's catalogue, a shell command, or a host process.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ocabra.core import docker_ops
from ocabra.docker_agent import ops


@pytest.fixture
def argvs(monkeypatch):
    """Record every argv the operations would run; answer success."""
    seen: list[list[str]] = []

    async def fake_exec(argv, timeout=120.0):  # noqa: ASYNC109
        seen.append(list(argv))
        return {"code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(ops, "_exec", fake_exec)
    return seen


def _catalogue_container() -> str:
    return sorted(ops._catalogue()[0])[0]


def _catalogue_service() -> str:
    return sorted(ops._catalogue()[1])[0]


# ── operations: what is allowed ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_container_actions_build_a_plain_docker_command(argvs):
    container = _catalogue_container()

    await ops.run("container.action", {"container": container, "action": "stop"})
    await ops.run("container.inspect", {"container": container, "field": "running"})

    assert argvs == [
        ["docker", "stop", container],
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
    ]


@pytest.mark.asyncio
async def test_compose_runs_only_on_ocabras_own_project_files(argvs):
    service = _catalogue_service()

    await ops.run("compose.up_service", {"service": service})
    await ops.run("ollama.recreate", {})

    for argv in argvs:
        assert argv[:2] == ["docker", "compose"]
        assert argv[argv.index("-f") + 1] == "/opt/ocabra/docker-compose.yml"
        assert argv[argv.index("--env-file") + 1] == "/opt/ocabra/.env"
    assert argvs[0][-1] == service
    assert argvs[1][-5:] == ["up", "-d", "--no-deps", "--force-recreate", "ollama"]


@pytest.mark.asyncio
async def test_uses_build_answers_only_the_boolean(monkeypatch):
    """The api used to read `compose config` itself: every service's secrets."""
    config = {
        "services": {
            "ollama": {"build": {"context": "."}},
            "api": {"environment": {"JWT_SECRET": "very-secret"}},
        }
    }

    async def fake_exec(argv, timeout=120.0):  # noqa: ASYNC109
        return {"code": 0, "stdout": json.dumps(config), "stderr": ""}

    monkeypatch.setattr(ops, "_exec", fake_exec)

    result = await ops.run("ollama.uses_build", {})

    assert result["uses_build"] is True
    assert "very-secret" not in json.dumps(result)


# ── operations: what is refused ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "args"),
    [
        ("container.action", {"container": "ocabra-postgres-1", "action": "stop"}),
        ("container.action", {"container": "PLACEHOLDER", "action": "rm"}),
        ("container.action", {"container": "PLACEHOLDER", "action": "exec"}),
        ("container.inspect", {"container": "PLACEHOLDER", "field": "{{json .Config.Env}}"}),
        ("container.stats", {"container": "../../etc"}),
        ("compose.up_service", {"service": "api"}),
        ("compose.up_service", {"service": "postgres"}),
        ("docker.run", {"image": "alpine", "args": ["-v", "/:/host"]}),
        ("updater.launch", {"version": "1.2", "current": "1.0.0"}),
        ("updater.launch", {"version": "v1.2.3", "current": "1.0.0"}),
        ("updater.launch", {"version": "1.2.3; rm -rf /", "current": "1.0.0"}),
        ("trtllm.kill_orphans", {"pgids": [1]}),
        ("trtllm.kill_orphans", {"pgids": "123"}),
    ],
)
async def test_requests_outside_the_operations_are_refused(op, args, argvs):
    args = {k: (_catalogue_container() if v == "PLACEHOLDER" else v) for k, v in args.items()}

    with pytest.raises(ops.OpError):
        await ops.run(op, args)

    assert argvs == []  # refused before running anything


_GOOD_JOB = {
    "phase": "convert",
    "engine_name": "Qwen--Qwen2.5-7B-fp16",
    "source_model": "vllm/Qwen/Qwen2.5-7B-Instruct",
    "dtype": "fp16",
    "gpu_indices": [0],
    "config": {"max_batch_size": 4},
}


@pytest.mark.parametrize(
    "bad",
    [
        {"engine_name": "../../../etc"},
        {"engine_name": "a b"},
        {"source_model": "vllm/../../root"},
        {"source_model": "/etc/passwd"},
        {"dtype": "float64; id"},
        {"gpu_indices": ["0"]},
        {"gpu_indices": []},
        {"gpu_indices": [True]},
        {"config": {"max_batch_size": "4 --entrypoint sh"}},
        {"phase": "shell"},
    ],
)
@pytest.mark.asyncio
async def test_trtllm_compile_refuses_anything_that_reaches_the_command_line(bad):
    stream = ops.stream("trtllm.phase", {**_GOOD_JOB, **bad})

    with pytest.raises(ops.OpError):
        await anext(stream)


def test_trtllm_compile_uses_the_image_and_mounts_of_the_settings():
    from ocabra.config import settings
    from ocabra.core.trtllm_compile_manager import build_trtllm_docker_cmd

    argv = build_trtllm_docker_cmd(ops._trtllm_job(_GOOD_JOB), "convert")

    assert argv[:3] == [settings.tensorrt_llm_docker_bin, "run", "--rm"]
    assert settings.tensorrt_llm_docker_image in argv
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == [
        f"{settings.tensorrt_llm_docker_models_mount_host}:"
        f"{settings.tensorrt_llm_docker_models_mount_container}"
    ]


@pytest.mark.asyncio
async def test_only_orphaned_trt_groups_are_killed_whatever_is_asked(monkeypatch):
    """Killing host process groups by id would kill anything; only the ones
    that are orphaned trtllm-serve runs right now are touched."""
    calls: list[tuple] = []

    async def fake_orphans():
        return {"code": 0, "stdout": "", "stderr": ""}, {4242}

    async def fake_helper(entrypoint, *args):
        calls.append((entrypoint, *args))
        return {"code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(ops, "_trtllm_orphan_groups", fake_orphans)
    monkeypatch.setattr(ops, "_host_helper", fake_helper)

    result = await ops.run("trtllm.kill_orphans", {"pgids": [4242, 777, 1234]})

    assert result["killed"] == [4242]
    assert len(calls) == 1 and "-4242" in calls[0][-1] and "-777" not in calls[0][-1]


# ── the agent's HTTP side ────────────────────────────────────────────────────


@pytest.fixture
def agent(monkeypatch, tmp_path, argvs):
    from ocabra.docker_agent import server

    monkeypatch.setattr(server.settings, "docker_agent_token_file", str(tmp_path / "token"))
    monkeypatch.setattr(server, "_token", server._write_token())
    return server.app, (tmp_path / "token").read_text()


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent")


@pytest.mark.asyncio
async def test_every_operation_needs_the_agent_token(agent):
    app, token = agent
    body = {"op": "ollama.ps", "args": {}}

    async with _client(app) as client:
        missing = await client.post("/v1/op", json=body)
        wrong = await client.post("/v1/op", json=body, headers={"Authorization": "Bearer nope"})
        right = await client.post("/v1/op", json=body, headers={"Authorization": f"Bearer {token}"})
        health = await client.get("/health")

    assert (missing.status_code, wrong.status_code, right.status_code) == (401, 401, 200)
    assert health.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_a_refused_operation_is_a_400(agent):
    app, token = agent

    async with _client(app) as client:
        response = await client.post(
            "/v1/op",
            json={"op": "compose.up_service", "args": {"service": "postgres"}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_a_stream_is_refused_before_anything_runs_and_ends_with_its_exit_code(
    agent, monkeypatch
):
    app, token = agent

    async def fake_phase(_args):
        yield "converting"
        yield f"{ops.EXIT_MARKER}3"

    monkeypatch.setitem(ops.STREAM_OPERATIONS, "trtllm.phase", fake_phase)
    headers = {"Authorization": f"Bearer {token}"}

    async with _client(app) as client:
        ok = await client.post(
            "/v1/stream", json={"op": "trtllm.phase", "args": {}}, headers=headers
        )
        refused = await client.post("/v1/stream", json={"op": "nope", "args": {}}, headers=headers)

    assert [json.loads(line) for line in ok.text.splitlines()] == [
        {"line": "converting"},
        {"exit": 3},
    ]
    assert refused.status_code == 400


# ── the api's client ─────────────────────────────────────────────────────────


@pytest.fixture
def via_agent(monkeypatch, agent):
    """docker_ops talking to the agent app in-process."""
    app, _token = agent
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.ASGITransport(app=app), base_url="http://agent")

    monkeypatch.setattr(docker_ops.settings, "docker_agent_url", "http://agent")
    monkeypatch.setattr(docker_ops.httpx, "AsyncClient", client)
    return app


class _NoLocalOps:
    """Stands in for the api's local path: any use of it fails the test."""

    OpError = ops.OpError
    EXIT_MARKER = ops.EXIT_MARKER

    async def run(self, op, args):
        raise AssertionError(f"ran {op} locally instead of through the agent")

    def stream(self, op, args):
        raise AssertionError(f"streamed {op} locally instead of through the agent")


@pytest.mark.asyncio
async def test_the_api_goes_through_the_agent_when_there_is_one(via_agent, monkeypatch, argvs):
    monkeypatch.setattr(docker_ops, "ops", _NoLocalOps())

    result = await docker_ops.call("ollama.ps")

    assert result["code"] == 0
    assert argvs[0][:2] == ["docker", "compose"]  # ran by the agent


@pytest.mark.asyncio
async def test_a_refusal_from_the_agent_is_not_retried_locally(via_agent, monkeypatch, argvs):
    monkeypatch.setattr(docker_ops, "ops", _NoLocalOps())

    with pytest.raises(docker_ops.OpError):
        await docker_ops.call("compose.up_service", service="postgres")
    assert argvs == []


@pytest.mark.asyncio
async def test_a_stream_goes_through_the_agent_too(via_agent, monkeypatch):
    async def fake_phase(_args):
        yield "building"
        yield f"{ops.EXIT_MARKER}0"

    monkeypatch.setitem(ops.STREAM_OPERATIONS, "trtllm.phase", fake_phase)
    monkeypatch.setattr(docker_ops, "ops", _NoLocalOps())

    lines = [line async for line in docker_ops.stream("trtllm.phase")]

    assert lines == ["building", f"{ops.EXIT_MARKER}0"]


@pytest.mark.asyncio
async def test_an_unreachable_agent_is_an_error_not_a_local_fallback(monkeypatch, tmp_path, argvs):
    """Phase 2: the api has no socket; running "locally" would be pointless,
    and reopening Docker access from the api is exactly what this closes."""
    token = tmp_path / "token"
    token.write_text("t")
    monkeypatch.setattr(docker_ops.settings, "docker_agent_url", "http://127.0.0.1:1")
    monkeypatch.setattr(docker_ops.settings, "docker_agent_token_file", str(token))

    with pytest.raises(docker_ops.AgentUnavailable):
        await docker_ops.call("ollama.ps")
    assert argvs == []


@pytest.mark.asyncio
async def test_an_agent_without_its_token_yet_is_an_error_too(monkeypatch, tmp_path, argvs):
    monkeypatch.setattr(docker_ops.settings, "docker_agent_url", "http://docker-agent:8790")
    monkeypatch.setattr(docker_ops.settings, "docker_agent_token_file", str(tmp_path / "none"))

    with pytest.raises(docker_ops.AgentUnavailable):
        await docker_ops.call("ollama.ps")
    assert docker_ops.available() is False
    assert argvs == []


def test_without_an_agent_url_the_api_runs_operations_itself(monkeypatch):
    monkeypatch.setattr(docker_ops.settings, "docker_agent_url", "")

    assert docker_ops.using_agent() is False


# ── TensorRT-LLM serving (docker launch mode) ────────────────────────────────


def _serve_args(**overrides):
    from ocabra.config import settings

    models = settings.tensorrt_llm_docker_models_mount_container
    args = {
        "engine_dir": f"{models}/tensorrt_llm/Qwen--Qwen2.5-7B-fp16/engine",
        "tokenizer_path": f"{models}/huggingface/Qwen--Qwen2.5-7B-Instruct",
        "port": settings.worker_port_range_start + 5,
        "backend": "trt",
        "gpu_indices": [1],
        "trust_remote_code": False,
        "max_seq_len": 8192,
    }
    return {**args, **overrides}


@pytest.mark.parametrize(
    "bad",
    [
        {"engine_dir": "/etc/passwd"},
        {"engine_dir": "/data/models/tensorrt_llm/../../../etc"},
        {"engine_dir": "/data/models/huggingface/not-an-engine"},
        {"tokenizer_path": "/root/.ssh"},
        {"port": 22},
        {"port": "18005"},
        {"gpu_indices": ["all"]},
        {"backend": "trt; id"},
        {"trust_remote_code": "yes"},
        {"max_seq_len": "8192 --entrypoint sh"},
    ],
)
@pytest.mark.asyncio
async def test_trtllm_serve_refuses_anything_outside_its_shape(bad):
    stream = ops.stream("trtllm.serve", _serve_args(**bad))

    with pytest.raises(ops.OpError):
        await anext(stream)


@pytest.mark.asyncio
async def test_trtllm_serve_builds_the_container_from_settings(monkeypatch):
    from ocabra.config import settings

    started: list[list[str]] = []

    async def fake_attached(argv, name):
        started.append([*argv, f"name={name}"])
        yield f"{ops.EXIT_MARKER}0"

    async def fake_api_container():
        return "abc123def456"

    monkeypatch.setattr(ops, "_attached_container", fake_attached)
    monkeypatch.setattr(ops, "_api_container", fake_api_container)

    lines = [line async for line in ops.stream("trtllm.serve", _serve_args())]

    argv = started[0]
    host = settings.tensorrt_llm_docker_models_mount_host
    assert lines == [f"{ops.EXIT_MARKER}0"]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--gpus") + 1] == "device=1"  # the assigned GPU, not "all"
    assert "--network=container:abc123def456" in argv  # found by the agent, not given
    assert settings.tensorrt_llm_docker_image in argv
    assert f"{host}/tensorrt_llm/Qwen--Qwen2.5-7B-fp16/engine" in argv  # host path
    assert argv[argv.index("--tokenizer") + 1] == f"{host}/huggingface/Qwen--Qwen2.5-7B-Instruct"
    assert argv[-1] == f"name=ocabra-trtllm-serve-{settings.worker_port_range_start + 5}"


@pytest.mark.asyncio
async def test_a_container_left_early_is_stopped_and_removed(monkeypatch, argvs):
    """Killing only the `docker run` CLI used to leave the container running."""
    import asyncio

    class _Proc:
        returncode = None
        terminated = False

        def __init__(self):
            async def lines():
                yield b"listening\n"
                await asyncio.sleep(3600)

            self.stdout = lines()

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    proc = _Proc()

    async def fake_spawn(*argv, **kwargs):
        return proc

    monkeypatch.setattr(ops.asyncio, "create_subprocess_exec", fake_spawn)
    stream = ops._attached_container(["docker", "run", "--rm", "img"], "ocabra-trtllm-serve-18005")

    assert await anext(stream) == "listening"
    await stream.aclose()  # the api unloaded the model

    assert proc.terminated
    assert argvs[-1] == ["docker", "rm", "-f", "ocabra-trtllm-serve-18005"]


def test_the_trtllm_worker_relays_the_agents_stream(monkeypatch, capsys):
    import argparse

    from ocabra.workers import tensorrt_llm_worker as worker

    seen: dict = {}

    async def fake_stream(op, **params):
        seen.update(op=op, **params)
        yield "Uvicorn running on 127.0.0.1"
        yield f"{ops.EXIT_MARKER}0"

    monkeypatch.setattr(docker_ops, "stream", fake_stream)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    args = argparse.Namespace(
        engine_dir="/data/models/tensorrt_llm/x/engine",
        port=18005,
        backend="trt",
        trust_remote_code=False,
        tokenizer_path=None,
        max_batch_size=None,
        max_num_tokens=None,
        max_seq_len=4096,
    )

    code = worker._serve_through_docker_agent(args)

    assert code == 0
    assert seen["op"] == "trtllm.serve" and seen["gpu_indices"] == [1]
    assert "Uvicorn running" in capsys.readouterr().out
