"""ModelManager over the logical models / deployments schema (migration 0034).

Integration tests against the ``<db>_test`` Postgres (see conftest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from ocabra.core import artifacts
from ocabra.core.model_manager import ModelManager
from ocabra.database import AsyncSessionLocal
from ocabra.db.model_config import LogicalModel, ModelConfig
from tests.conftest import requires_db

KEY = "zz-test-unification:1b"


async def _cleanup() -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(sa.delete(ModelConfig).where(ModelConfig.model_key == KEY))
        await session.execute(sa.delete(LogicalModel).where(LogicalModel.model_key == KEY))
        await session.commit()


@pytest.fixture
async def clean_db():
    await _cleanup()
    yield
    await _cleanup()


@pytest.fixture
def ollama_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest = (
        tmp_path / "manifests" / "registry.ollama.ai" / "library" / "zz-test-unification" / "1b"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}
                ]
            }
        )
    )
    monkeypatch.setattr(artifacts, "ollama_root", lambda: str(tmp_path))
    return tmp_path


@requires_db
async def test_deployment_loads_with_backend_ref_and_paths_from_its_artifact(
    clean_db, ollama_store: Path
) -> None:
    async with AsyncSessionLocal() as session:
        session.add(LogicalModel(model_key=KEY, default_backend="llamacpp_mks"))
        await session.flush()
        session.add_all(
            [
                ModelConfig(
                    model_id=f"llamacpp_mks/{KEY}",
                    model_key=KEY,
                    backend_type="llamacpp_mks",
                    artifact={"kind": "gguf", "source": "ollama", "ref": "zz-test-unification:1b"},
                    extra_config={"llama_cpp": {"ctx_size": 4096}},
                ),
                ModelConfig(
                    model_id=f"vllm/{KEY}",
                    model_key=KEY,
                    backend_type="vllm",
                    artifact={"kind": "hf_repo", "source": "local", "ref": "Org/Repo-1B"},
                ),
            ]
        )
        await session.commit()

    mm = ModelManager(worker_pool=None)
    await mm._load_configs_from_db()

    llama = mm._states[f"llamacpp_mks/{KEY}"]
    assert llama.model_key == KEY
    assert llama.backend_model_id == "zz-test-unification:1b"
    assert llama.extra_config["llama_cpp"] == {
        "ctx_size": 4096,
        "model_path": str(ollama_store / "blobs" / "sha256-abc"),
    }
    # vLLM resolves the HF repo from what it receives as model id, not from the key
    assert mm._states[f"vllm/{KEY}"].backend_model_id == "Org/Repo-1B"


@requires_db
async def test_add_and_delete_manage_the_logical_model_row(clean_db, ollama_store: Path) -> None:
    mm = ModelManager(worker_pool=None)
    state = await mm.add_model(
        model_id=f"bitnet/{KEY}",
        backend_type="bitnet",
        extra_config={"bitnet": {"modelPath": "/models/x.gguf", "ctxSize": 2048}},
    )
    # camelCase from the UI is normalised; the path goes to the artifact, not to the row
    assert state.model_key == KEY
    async with AsyncSessionLocal() as session:
        row = await session.scalar(
            sa.select(ModelConfig).where(ModelConfig.model_id == f"bitnet/{KEY}")
        )
        assert row.extra_config == {"bitnet": {"ctx_size": 2048}}
        assert row.artifact["path"] == "/models/x.gguf"
        model = await session.get(LogicalModel, KEY)
        assert model is not None and model.default_backend == "bitnet"
    assert state.extra_config["bitnet"]["model_path"] == "/models/x.gguf"

    await mm.delete_model(f"bitnet/{KEY}")
    async with AsyncSessionLocal() as session:
        assert await session.get(LogicalModel, KEY) is None


@requires_db
async def test_ui_edit_of_ctx_is_applied_even_with_a_stale_snake_twin(
    clean_db, ollama_store: Path
) -> None:
    """Regression: the UI wrote ``ctxSize`` next to a stored ``ctx_size`` and the
    reader (snake first) kept using the old value."""
    mm = ModelManager(worker_pool=None)
    await mm.add_model(
        model_id=f"bitnet/{KEY}", backend_type="bitnet", extra_config={"bitnet": {"ctx_size": 2048}}
    )

    await mm.update_config(
        f"bitnet/{KEY}", {"extra_config": {"bitnet": {"ctxSize": 8192, "ctx_size": 2048}}}
    )

    async with AsyncSessionLocal() as session:
        row = await session.scalar(
            sa.select(ModelConfig).where(ModelConfig.model_id == f"bitnet/{KEY}")
        )
        assert row.extra_config["bitnet"]["ctx_size"] == 8192
        assert "ctxSize" not in row.extra_config["bitnet"]
