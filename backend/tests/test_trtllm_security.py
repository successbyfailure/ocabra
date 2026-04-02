from types import SimpleNamespace

import pytest

from ocabra.api.internal import trtllm


@pytest.mark.asyncio
async def test_delete_engine_rejects_path_escape(tmp_path, monkeypatch):
    engines_root = tmp_path / "engines"
    engines_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    (outside_root / "keep.txt").write_text("keep")

    monkeypatch.setattr(trtllm.settings, "tensorrt_llm_engines_dir", str(engines_root))

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(model_manager=SimpleNamespace(_states={})))
    )

    with pytest.raises(trtllm.HTTPException) as exc_info:
        await trtllm.delete_engine("../outside", request)

    assert exc_info.value.status_code == 400
    assert (outside_root / "keep.txt").exists()
