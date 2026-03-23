import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from ocabra.backends.vllm_recipes import get_vllm_recipe
from ocabra.core.model_manager import ModelManager
from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.local_scanner import LocalScanner
from ocabra.registry.ollama_registry import OllamaRegistry
from ocabra.schemas.registry import HFVLLMRuntimeProbe


@pytest.mark.asyncio
async def test_hf_search_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_list_models(**kwargs):
        captured.update(kwargs)
        return [
            SimpleNamespace(
                id="meta-llama/Llama-3.2-3B-Instruct",
                pipeline_tag="text-generation",
                downloads=123,
                likes=45,
                tags=["llama", "chat"],
                gated=False,
            )
        ]

    monkeypatch.setattr("ocabra.registry.huggingface.list_models", fake_list_models)

    registry = HuggingFaceRegistry()
    cards = await registry.search(query="llama", task="text-generation", limit=5)

    assert len(cards) == 1
    assert cards[0].repo_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert cards[0].task == "text-generation"
    assert cards[0].suggested_backend == "vllm"
    assert captured["search"] == "llama"
    assert captured["pipeline_tag"] == "text-generation"
    assert captured["limit"] == 5


@pytest.mark.asyncio
async def test_hf_detail_infers_vllm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="meta-llama/Llama-3.2-3B-Instruct",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3)],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("meta-llama/Llama-3.2-3B-Instruct")

    assert detail.suggested_backend == "vllm"
    assert detail.estimated_vram_gb is not None
    assert detail.estimated_vram_gb > 5.0


@pytest.mark.asyncio
async def test_hf_detail_classifies_transformers_backend_and_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="Qwen/Qwen3-8B-Instruct",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["SomeFutureCausalLM"]}
        if filename == "tokenizer_config.json":
            return {}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="supported_transformers_backend",
            recommended_model_impl="transformers",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("Qwen/Qwen3-8B-Instruct")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert detail.vllm_support.classification == "transformers_backend"
    assert detail.vllm_support.model_impl == "transformers"
    assert "chat_template" in detail.vllm_support.required_overrides
    assert "tool_call_parser" in detail.vllm_support.required_overrides
    assert detail.vllm_support.recipe_id == "qwen3"
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "supported_transformers_backend"


def test_vllm_recipe_detects_qwen3_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="Qwen/Qwen3-8B-Instruct",
        architectures=["Qwen3ForCausalLM"],
        tokenizer_config={},
    )

    assert recipe is not None
    assert recipe.recipe_id == "qwen3"
    assert recipe.model_impl == "vllm"
    assert recipe.runner == "generate"
    assert recipe.suggested_config["tool_call_parser"] == "qwen3_json"
    assert recipe.suggested_tuning["gpu_memory_utilization"] == 0.9
    assert "reasoning_parser" in recipe.required_overrides


def test_vllm_recipe_detects_qwen_vl_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        architectures=["Qwen2VLForConditionalGeneration"],
        tokenizer_config={},
    )

    assert recipe is not None
    assert recipe.recipe_id == "qwen-vl"
    assert recipe.runner == "generate"
    assert recipe.suggested_tuning["language_model_only"] is True
    assert "chat_template" in recipe.required_overrides


def test_vllm_recipe_detects_qwen3_moe_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="Qwen/Qwen3-30B-A3B-MoE",
        architectures=["Qwen3MoeForCausalLM"],
        tokenizer_config={"chat_template": "{{ messages }}"},
    )

    assert recipe is not None
    assert recipe.recipe_id == "qwen3-moe"
    assert recipe.suggested_tuning["max_num_seqs"] == 8
    assert "chat_template" not in recipe.required_overrides


def test_vllm_recipe_detects_internvl_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="OpenGVLab/InternVL3-8B",
        architectures=["InternVLChatModel"],
        tokenizer_config={"chat_template": "{{ messages }}"},
    )

    assert recipe is not None
    assert recipe.recipe_id == "internvl-chat"
    assert recipe.suggested_tuning["language_model_only"] is True


def test_vllm_recipe_detects_llama4_multimodal_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        architectures=["Llama4ForConditionalGeneration"],
        tokenizer_config={},
    )

    assert recipe is not None
    assert recipe.recipe_id == "llama4-multimodal"
    assert recipe.suggested_tuning["language_model_only"] is True
    assert "chat_template" in recipe.required_overrides


def test_vllm_recipe_detects_deepseek_r1_defaults() -> None:
    recipe = get_vllm_recipe(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        architectures=["DeepseekR1ForCausalLM"],
        tokenizer_config={"chat_template": "{{ messages }}"},
    )

    assert recipe is not None
    assert recipe.recipe_id == "deepseek-r1"
    assert recipe.suggested_config["reasoning_parser"] == "deepseek_r1"
    assert "chat_template" not in recipe.required_overrides


@pytest.mark.asyncio
async def test_hf_detail_exposes_recipe_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="Qwen/Qwen3-8B-Instruct",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["Qwen3ForCausalLM"]}
        if filename == "tokenizer_config.json":
            return {}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="supported_native",
            recommended_model_impl="vllm",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("Qwen/Qwen3-8B-Instruct")

    assert detail.vllm_support is not None
    assert detail.vllm_support.recipe_id == "qwen3"
    assert detail.vllm_support.recipe_model_impl == "vllm"
    assert detail.vllm_support.recipe_runner == "generate"
    assert detail.vllm_support.suggested_config["tool_call_parser"] == "qwen3_json"
    assert detail.vllm_support.suggested_tuning["gpu_memory_utilization"] == 0.9
    assert "reasoning_parser" in detail.vllm_support.required_overrides


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_remote_code_adds_required_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/fancy-custom-model",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["SomeFutureCausalLM"]}
        if filename == "tokenizer_config.json":
            return {"chat_template": "{{ messages }}"}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="needs_remote_code",
            reason="Custom code required by config class",
            recommended_model_impl="transformers",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/fancy-custom-model")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert "trust_remote_code" in detail.vllm_support.required_overrides
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "needs_remote_code"


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_missing_chat_template_adds_required_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/chat-model",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["Qwen3ForCausalLM"]}
        if filename == "tokenizer_config.json":
            return {}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="missing_chat_template",
            reason="Falta chat template",
            recommended_model_impl="vllm",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/chat-model")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert "chat_template" in detail.vllm_support.required_overrides
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "missing_chat_template"


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_missing_tool_parser_adds_required_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/tool-model",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["Qwen3ForCausalLM"]}
        if filename == "tokenizer_config.json":
            return {"chat_template": "{{ messages }}"}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="missing_tool_parser",
            reason="Falta tool_call_parser",
            recommended_model_impl="vllm",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/tool-model")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert "tool_call_parser" in detail.vllm_support.required_overrides
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "missing_tool_parser"


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_missing_reasoning_parser_adds_required_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/reasoning-model",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["DeepseekR1ForCausalLM"]}
        if filename == "tokenizer_config.json":
            return {"chat_template": "{{ messages }}"}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="missing_reasoning_parser",
            reason="Falta reasoning_parser",
            recommended_model_impl="vllm",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/reasoning-model")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert "reasoning_parser" in detail.vllm_support.required_overrides
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "missing_reasoning_parser"


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_needs_hf_overrides_adds_required_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/qwen-variant",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["Qwen3ForCausalLM"]}
        if filename == "tokenizer_config.json":
            return {"chat_template": "{{ messages }}"}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="needs_hf_overrides",
            reason="Hace falta hf_overrides para rope_scaling",
            recommended_model_impl="vllm",
            recommended_runner="generate",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/qwen-variant")

    assert detail.compatibility == "warning"
    assert detail.vllm_support is not None
    assert "hf_overrides" in detail.vllm_support.required_overrides
    assert detail.vllm_support.runtime_probe is not None
    assert detail.vllm_support.runtime_probe.status == "needs_hf_overrides"


@pytest.mark.asyncio
async def test_hf_detail_runtime_probe_overrides_model_impl_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/custom-pooling-model",
        library_name="transformers",
        pipeline_tag="feature-extraction",
        downloads=1000,
        likes=200,
        tags=["feature-extraction"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["SomeFuturePoolingModel"]}
        if filename == "tokenizer_config.json":
            return {}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="supported_pooling",
            recommended_model_impl="transformers",
            recommended_runner="pooling",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("org/custom-pooling-model")

    assert detail.vllm_support is not None
    assert detail.vllm_support.model_impl == "transformers"
    assert detail.vllm_support.runner == "pooling"


@pytest.mark.asyncio
async def test_hf_detail_gguf_only_repo_is_not_treated_as_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        library_name=None,
        pipeline_tag=None,
        downloads=1000,
        likes=200,
        tags=["gguf"],
        gated=False,
        siblings=[SimpleNamespace(rfilename="qwen2.5-7b-instruct-q4_k_m.gguf", size=4 * 1024**3)],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("bartowski/Qwen2.5-7B-Instruct-GGUF")

    assert detail.suggested_backend == "vllm"


@pytest.mark.asyncio
async def test_hf_detail_prefers_transformers_metadata_over_model_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="Qwen/Qwen3-8B",
        library_name="transformers",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation", "transformers"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="model_index.json", size=1_000),
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=4 * 1024**3),
        ],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("Qwen/Qwen3-8B")

    assert detail.suggested_backend == "vllm"


@pytest.mark.asyncio
async def test_hf_detail_detects_qwen_tts_from_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="Qwen/Qwen3-TTS-1.7B",
        library_name="qwen-tts",
        pipeline_tag="text-to-speech",
        downloads=1000,
        likes=200,
        tags=["text-to-speech"],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=10_000),
            SimpleNamespace(rfilename="model.safetensors", size=2 * 1024**3),
        ],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("Qwen/Qwen3-TTS-1.7B")

    assert detail.suggested_backend == "tts"


@pytest.mark.asyncio
async def test_hf_detail_detects_diffusers_from_repo_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="black-forest-labs/FLUX.1-schnell",
        library_name=None,
        pipeline_tag=None,
        downloads=1000,
        likes=200,
        tags=[],
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="model_index.json", size=1_000),
            SimpleNamespace(rfilename="transformer/config.json", size=10_000),
            SimpleNamespace(rfilename="vae/config.json", size=10_000),
            SimpleNamespace(rfilename="scheduler/scheduler_config.json", size=10_000),
        ],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("black-forest-labs/FLUX.1-schnell")

    assert detail.suggested_backend == "diffusers"


@pytest.mark.asyncio
async def test_hf_variants_exposes_standard_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="org/mixed-model",
        siblings=[
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=2 * 1024**3),
            SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=2 * 1024**3),
            SimpleNamespace(rfilename="model.safetensors.index.json", size=1000),
            SimpleNamespace(rfilename="model-q4_k_m.gguf", size=1 * 1024**3),
            SimpleNamespace(rfilename="model-q8_0.gguf", size=2 * 1024**3),
        ],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    registry = HuggingFaceRegistry()
    variants = await registry.get_variants("org/mixed-model")

    assert any(v.variant_id == "standard" and v.backend_type == "vllm" for v in variants)
    assert not any(v.artifact == "model-q4_k_m.gguf" for v in variants)


@pytest.mark.asyncio
async def test_hf_variants_classifies_pooling_models(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="intfloat/e5-base-v2",
        siblings=[
            SimpleNamespace(rfilename="config.json", size=1_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=1_000),
            SimpleNamespace(rfilename="model.safetensors", size=2 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "config.json":
            return {"architectures": ["BertModel"]}
        if filename == "tokenizer_config.json":
            return {}
        return None

    async def fake_probe(
        self, repo_id: str, config: dict | None, tokenizer_config: dict | None, support
    ):
        _ = repo_id, config, tokenizer_config, support
        return HFVLLMRuntimeProbe(
            status="supported_pooling",
            recommended_model_impl="transformers",
            recommended_runner="pooling",
            tokenizer_load=True,
            config_load=True,
        )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)
    monkeypatch.setattr(HuggingFaceRegistry, "_probe_vllm_runtime_support", fake_probe)

    registry = HuggingFaceRegistry()
    variants = await registry.get_variants("intfloat/e5-base-v2")

    assert variants[0].compatibility == "compatible"
    assert variants[0].vllm_support is not None
    assert variants[0].vllm_support.classification == "pooling"
    assert variants[0].vllm_support.runner == "pooling"


@pytest.mark.asyncio
async def test_hf_variants_marks_unsupported_tokenizer_as_not_installable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_info = SimpleNamespace(
        id="org/bad-tokenizer",
        siblings=[
            SimpleNamespace(rfilename="config.json", size=1_000),
            SimpleNamespace(rfilename="tokenizer_config.json", size=1_000),
            SimpleNamespace(rfilename="model.safetensors", size=2 * 1024**3),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "tokenizer_config.json":
            return {"tokenizer_class": "TokenizersBackendFast"}
        if filename == "config.json":
            return {"model_type": "qwen3_5"}
        return None

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)

    registry = HuggingFaceRegistry()
    variants = await registry.get_variants("org/bad-tokenizer")

    assert variants[0].installable is False
    assert variants[0].compatibility == "unsupported"
    assert "TokenizersBackendFast" in (variants[0].compatibility_reason or "")


@pytest.mark.asyncio
async def test_hf_download_rejects_incompatible_tokenizer_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_info = SimpleNamespace(
        id="org/bad-tokenizer",
        pipeline_tag="text-generation",
        library_name="transformers",
        tags=[],
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="tokenizer_config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
        ],
    )

    async def fake_load_repo_json(self, repo_id: str, filename: str, sibling_names: set[str]):
        _ = repo_id, sibling_names
        if filename == "tokenizer_config.json":
            return {"tokenizer_class": "TokenizersBackend"}
        if filename == "config.json":
            return {"model_type": "mistral"}
        return None

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr(HuggingFaceRegistry, "_load_repo_json", fake_load_repo_json)

    registry = HuggingFaceRegistry()
    with pytest.raises(ValueError, match="tokenizer_class=TokenizersBackend"):
        await registry.download(
            repo_id="org/bad-tokenizer",
            target_dir=tmp_path / "model",
            progress_callback=lambda *_args: None,
        )


@pytest.mark.asyncio
async def test_hf_download_rejects_gguf_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = HuggingFaceRegistry()

    with pytest.raises(ValueError, match="GGUF"):
        await registry.download(
            repo_id="org/model",
            target_dir=tmp_path / "model",
            progress_callback=lambda *_args: None,
            artifact="model-q4_k_m.gguf",
        )


@pytest.mark.asyncio
async def test_ollama_pull_parses_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                json.dumps({"status": "pulling", "completed": 50, "total": 100}),
                json.dumps({"status": "pulling", "completed": 100, "total": 100}),
                json.dumps({"status": "success"}),
            ]
            for line in lines:
                await asyncio.sleep(0)
                yield line

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def stream(self, method: str, url: str, json: dict):
            _ = method, url, json
            return FakeStreamContext()

    monkeypatch.setattr("ocabra.registry.ollama_registry.httpx.AsyncClient", FakeClient)

    updates: list[float] = []
    registry = OllamaRegistry()
    await registry.pull("llama3.2:3b", lambda pct, _speed: updates.append(pct))

    assert updates
    assert any(pct >= 50 for pct in updates)
    assert updates[-1] == 100.0


@pytest.mark.asyncio
async def test_ollama_search_scrapes_library_links(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <a href="/library/llama3.1">Llama</a>
    <a href="/library/gemma3">Gemma</a>
    <a href="/library/llama3.1">Duplicate</a>
    """

    class FakeResponse:
        text = html

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def get(self, url: str, params: dict):
            _ = url, params
            return FakeResponse()

    monkeypatch.setattr("ocabra.registry.ollama_registry.httpx.AsyncClient", FakeClient)

    registry = OllamaRegistry()
    cards = await registry.search("llama")
    names = [card.name for card in cards]
    assert "llama3.1" in names
    assert "gemma3" not in names


@pytest.mark.asyncio
async def test_ollama_list_installed_details(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "models": [
            {"name": "gemma3:4b", "size": 3338801804, "modified_at": "2026-03-11T15:26:31Z"},
            {"model": "llama3.1:8b", "size": 4920753328, "modified_at": "2026-03-11T15:28:30Z"},
        ]
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def get(self, url: str):
            _ = url
            return FakeResponse()

    monkeypatch.setattr("ocabra.registry.ollama_registry.httpx.AsyncClient", FakeClient)

    registry = OllamaRegistry()
    details = await registry.list_installed_details()
    assert details[0]["name"] == "gemma3:4b"
    assert details[0]["size"] == 3338801804
    assert details[1]["name"] == "llama3.1:8b"


@pytest.mark.asyncio
async def test_model_manager_syncs_native_ollama_inventory() -> None:
    manager = ModelManager(worker_pool=object())

    added = await manager.sync_ollama_models(["gemma3:4b", "llama3.2:3b", "gemma3:4b"])

    assert added == 2
    states = await manager.list_states()
    assert sorted(state.model_id for state in states) == ["gemma3:4b", "llama3.2:3b"]
    assert all(state.backend_type == "ollama" for state in states)


@pytest.mark.asyncio
async def test_model_manager_syncs_ollama_loaded_runtime_state() -> None:
    class DummyWorkerPool:
        def __init__(self) -> None:
            self.workers: dict[str, object] = {}

        def set_worker(self, model_id: str, worker_info) -> None:
            self.workers[model_id] = worker_info

        def remove_worker(self, model_id: str) -> None:
            self.workers.pop(model_id, None)

    manager = ModelManager(worker_pool=DummyWorkerPool())
    manager._publish_event = lambda *args, **kwargs: asyncio.sleep(0)  # type: ignore[method-assign]

    await manager.sync_ollama_inventory(["qwen3:32b"], ["qwen3:32b"], loaded_vram_mb={"qwen3:32b": 6144})
    state = await manager.get_state("qwen3:32b")

    assert state is not None
    assert state.status.value == "loaded"
    assert state.loaded_at is not None
    assert state.worker_info is not None
    assert state.vram_used_mb == 6144

    await manager.sync_ollama_inventory(["qwen3:32b"], [])
    state = await manager.get_state("qwen3:32b")

    assert state is not None
    assert state.status.value == "unloaded"
    assert state.worker_info is None


@pytest.mark.asyncio
async def test_ollama_get_variants_parses_tags_page(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <a href="/library/gemma3:4b" class="x">
      <span class="font-mono">abc</span> • 3.3GB • 128K context window • Text input • 2 months ago
    </a>
    <a href="/library/gemma3:4b-it-q4_K_M" class="x">
      <span class="font-mono">def</span> • 3.1GB • 128K context window • Text input • 2 months ago
    </a>
    """

    class FakeResponse:
        text = html

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def get(self, url: str):
            _ = url
            return FakeResponse()

    monkeypatch.setattr("ocabra.registry.ollama_registry.httpx.AsyncClient", FakeClient)

    registry = OllamaRegistry()
    variants = await registry.get_variants("gemma3")
    names = [v.name for v in variants]
    assert "gemma3:4b" in names
    q4 = next(v for v in variants if v.name == "gemma3:4b-it-q4_K_M")
    assert q4.quantization == "Q4_K_M"
    assert q4.parameter_size == "4b"


@pytest.mark.asyncio
async def test_local_scanner_discovers_hf_gguf_and_ollama(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf-model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "weights.safetensors").write_bytes(b"1234")

    gguf_file = tmp_path / "tiny.gguf"
    gguf_file.write_bytes(b"abcd")
    bitnet_gguf = tmp_path / "bitnet-b1.58-i2_s.gguf"
    bitnet_gguf.write_bytes(b"GGUF-bitnet-header")

    ollama_dir = tmp_path / "ollama-model"
    ollama_dir.mkdir()
    (ollama_dir / "Modelfile").write_text("FROM llama3.2", encoding="utf-8")

    scanner = LocalScanner()
    models = await scanner.scan(tmp_path)

    sources = {model.source for model in models}
    assert sources == {"huggingface", "gguf", "ollama"}
    assert any(model.model_ref == "hf-model" for model in models)
    assert any(model.model_ref == "tiny" for model in models)
    assert any(
        model.model_ref == "tiny" and model.backend_type == "vllm" for model in models
    )
    assert any(
        model.model_ref == "bitnet-b1.58-i2_s" and model.backend_type == "bitnet"
        for model in models
    )
    assert any(model.model_ref == "ollama-model" for model in models)
    assert any(
        model.model_ref == "ollama-model" and model.backend_type == "ollama" for model in models
    )


@pytest.mark.asyncio
async def test_registry_hf_endpoint(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_search(query: str, task: str | None, limit: int):
        _ = query, task, limit
        return [
            {
                "repo_id": "org/model",
                "model_name": "model",
                "task": "text-generation",
                "downloads": 1,
                "likes": 2,
                "size_gb": None,
                "tags": [],
                "gated": False,
                "suggested_backend": "vllm",
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._hf_registry.search", fake_search)
    response = await async_client.get("/ocabra/registry/hf", params={"q": "model"})

    assert response.status_code == 200
    assert response.json()[0]["repo_id"] == "org/model"


@pytest.mark.asyncio
async def test_registry_hf_variants_endpoint(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_variants(repo_id: str):
        _ = repo_id
        return [
            {
                "variant_id": "gguf:model-q4_k_m.gguf",
                "label": "model-q4_k_m.gguf",
                "artifact": "model-q4_k_m.gguf",
                "size_gb": 1.2,
                "format": "gguf",
                "quantization": "Q4_K_M",
                "backend_type": "ollama",
                "is_default": True,
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._hf_registry.get_variants", fake_variants)
    response = await async_client.get("/ocabra/registry/hf/org/model/variants")
    assert response.status_code == 200
    assert response.json()[0]["artifact"] == "model-q4_k_m.gguf"


@pytest.mark.asyncio
async def test_registry_ollama_endpoint(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_search(query: str):
        _ = query
        return [
            {
                "name": "llama3.2:3b",
                "description": "Test",
                "tags": ["latest"],
                "size_gb": 1.2,
                "pulls": 100,
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._ollama_registry.search", fake_search)
    response = await async_client.get("/ocabra/registry/ollama", params={"q": "llama"})

    assert response.status_code == 200
    assert response.json()[0]["name"] == "llama3.2:3b"


@pytest.mark.asyncio
async def test_registry_ollama_variants_endpoint(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_variants(model_name: str):
        _ = model_name
        return [
            {
                "name": "gemma3:4b-it-q4_K_M",
                "tag": "4b-it-q4_K_M",
                "size_gb": 3.3,
                "parameter_size": "4b",
                "quantization": "Q4_K_M",
                "context_window": "128K",
                "modality": "text",
                "updated_hint": "2 months ago",
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._ollama_registry.get_variants", fake_variants)
    response = await async_client.get("/ocabra/registry/ollama/gemma3/variants")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "gemma3:4b-it-q4_K_M"


@pytest.mark.asyncio
async def test_hf_variants_exposes_nemo_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="nvidia/parakeet-tdt-0.6b-v3",
        pipeline_tag="automatic-speech-recognition",
        library_name="nemo",
        tags=["nemo", "automatic-speech-recognition"],
        siblings=[
            SimpleNamespace(rfilename="README.md", size=1234),
            SimpleNamespace(rfilename="parakeet-tdt-0.6b-v3.nemo", size=6 * 1024**3),
        ],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    variants = await registry.get_variants("nvidia/parakeet-tdt-0.6b-v3")

    assert len(variants) == 1
    assert variants[0].backend_type == "whisper"
    assert variants[0].artifact == "parakeet-tdt-0.6b-v3.nemo"
    assert variants[0].format == "nemo"
    assert variants[0].installable is True
    assert variants[0].is_default is True


@pytest.mark.asyncio
async def test_hf_download_resolves_nemo_default_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_info = SimpleNamespace(
        id="nvidia/canary-1b-v2",
        pipeline_tag="automatic-speech-recognition",
        library_name="nemo",
        tags=["nemo", "automatic-speech-recognition"],
        siblings=[
            SimpleNamespace(rfilename="README.md"),
            SimpleNamespace(rfilename="canary-1b-v2.nemo"),
        ],
    )

    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)
    monkeypatch.setattr("ocabra.registry.huggingface.snapshot_download", fake_snapshot_download)

    registry = HuggingFaceRegistry()
    out = await registry.download(
        repo_id="nvidia/canary-1b-v2",
        target_dir=tmp_path / "download",
        progress_callback=lambda *_: None,
        artifact=None,
    )

    assert out == tmp_path / "download"
    allow_patterns = captured.get("allow_patterns")
    assert isinstance(allow_patterns, list)
    assert "canary-1b-v2.nemo" in allow_patterns

@pytest.mark.asyncio
async def test_ollama_list_loaded_details(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def json(self) -> dict:
            return {
                "models": [
                    {"name": "qwen3:32b", "model": "qwen3:32b", "size_vram": 6442450944, "digest": "sha256:abc"},
                    {"name": "mistral:7b", "size": 3221225472},
                ]
            }

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        async def get(self, url: str):
            _ = url
            return FakeResponse()

    monkeypatch.setattr("ocabra.registry.ollama_registry.httpx.AsyncClient", FakeClient)

    registry = OllamaRegistry()
    details = await registry.list_loaded_details()
    assert details[0]["name"] == "qwen3:32b"
    assert details[0]["size_vram"] == 6442450944
    assert details[1]["name"] == "mistral:7b"
    assert details[1]["size_vram"] == 3221225472
