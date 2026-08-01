from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.registry.bitnet_registry import BitnetRegistry


@pytest.mark.asyncio
async def test_search_returns_bitnet_card(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = SimpleNamespace(
        id="microsoft/BitNet-b1.58-2B-4T-gguf",
        pipeline_tag="text-generation",
        downloads=100,
        likes=10,
        tags=["bitnet"],
        gated=False,
        siblings=[SimpleNamespace(rfilename="ggml-model-i2_s.gguf", size=1024)],
    )

    monkeypatch.setattr("ocabra.registry.bitnet_registry.list_models", lambda **_: [fake_model])

    registry = BitnetRegistry()
    cards = await registry.search("bitnet", limit=10)

    assert len(cards) == 1
    assert cards[0].repo_id == "microsoft/BitNet-b1.58-2B-4T-gguf"
    assert cards[0].suggested_backend == "bitnet"


@pytest.mark.asyncio
async def test_get_variants_only_returns_bitnet_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="ggml-model-i2_s.gguf", size=1024),
            SimpleNamespace(rfilename="README.md", size=32),
            SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=2048),
        ]
    )

    monkeypatch.setattr("ocabra.registry.bitnet_registry.model_info", lambda **_: fake_info)

    registry = BitnetRegistry()
    variants = await registry.get_variants("microsoft/BitNet-b1.58-2B-4T-gguf")

    assert len(variants) == 1
    assert variants[0].artifact == "ggml-model-i2_s.gguf"
    assert variants[0].backend_type == "bitnet"
    assert variants[0].is_default is True


@pytest.mark.asyncio
async def test_search_includes_bonsai(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = SimpleNamespace(
        id="prism-ml/Bonsai-27B-gguf",
        pipeline_tag="text-generation",
        downloads=50,
        likes=5,
        tags=["ternary", "1.58-bit"],
        gated=False,
        siblings=[SimpleNamespace(rfilename="Bonsai-27B-Q1_0.gguf", size=4096)],
    )

    monkeypatch.setattr("ocabra.registry.bitnet_registry.list_models", lambda **_: [fake_model])

    registry = BitnetRegistry()
    cards = await registry.search("bonsai", limit=10)

    assert len(cards) == 1
    assert cards[0].repo_id == "prism-ml/Bonsai-27B-gguf"
    assert cards[0].suggested_backend == "bitnet"


@pytest.mark.asyncio
async def test_get_variants_excludes_dspark_drafters(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="Bonsai-27B-Q1_0.gguf", size=4096),
            SimpleNamespace(rfilename="Bonsai-27B-dspark-Q4_1.gguf", size=1800),
            SimpleNamespace(rfilename="Bonsai-27B-mmproj-BF16.gguf", size=900),
            SimpleNamespace(rfilename="README.md", size=32),
        ]
    )

    monkeypatch.setattr("ocabra.registry.bitnet_registry.model_info", lambda **_: fake_info)

    registry = BitnetRegistry()
    variants = await registry.get_variants("prism-ml/Bonsai-27B-gguf")

    # dspark drafters and the mmproj vision projector are not servable LMs.
    assert len(variants) == 1
    assert variants[0].artifact == "Bonsai-27B-Q1_0.gguf"
    assert variants[0].quantization == "Q1_0"
    assert variants[0].backend_type == "bitnet"
    assert variants[0].is_default is True
