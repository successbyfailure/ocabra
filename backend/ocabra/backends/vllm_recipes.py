from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VLLMRecipe:
    recipe_id: str
    notes: list[str] = field(default_factory=list)
    model_impl: str | None = None
    runner: str | None = None
    required_overrides: list[str] = field(default_factory=list)
    suggested_config: dict[str, Any] = field(default_factory=dict)
    suggested_tuning: dict[str, Any] = field(default_factory=dict)


def get_vllm_recipe(
    repo_id: str,
    architectures: list[str],
    tokenizer_config: dict[str, Any] | None,
) -> VLLMRecipe | None:
    repo = repo_id.lower()
    archs = {arch.lower() for arch in architectures}
    tokenizer_has_chat_template = bool((tokenizer_config or {}).get("chat_template"))

    if (
        "qwen2-vl" in repo
        or "qwen2.5-vl" in repo
        or "qwen-vl" in repo
        or "qwen2vlforconditionalgeneration" in archs
    ):
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="qwen-vl",
            notes=["Qwen-VL suele necesitar ajuste explícito para uso solo texto y ahorrar VRAM."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={},
            suggested_tuning={
                "language_model_only": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    if "qwen3moeforcausallm" in archs or ("qwen3" in repo and "moe" in repo):
        required = ["tool_call_parser", "reasoning_parser"]
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="qwen3-moe",
            notes=[
                "Qwen3 suele necesitar parser especifico para tools y reasoning.",
                "Las variantes MoE agradecen concurrencia algo mas conservadora.",
            ],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={
                "tool_call_parser": "qwen3_json",
                "reasoning_parser": "qwen3",
            },
            suggested_tuning={
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 8,
            },
        )

    if "qwen3" in repo or "qwen3forcausallm" in archs:
        required = ["tool_call_parser", "reasoning_parser"]
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="qwen3",
            notes=["Qwen3 suele necesitar parser especifico para tools y reasoning."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={
                "tool_call_parser": "qwen3_json",
                "reasoning_parser": "qwen3",
            },
            suggested_tuning={
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    if "deepseek-r1" in repo or "deepseekr1forcausallm" in archs:
        required = ["reasoning_parser"]
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        recipe_id = "deepseek-r1-distill" if "distill" in repo else "deepseek-r1"
        return VLLMRecipe(
            recipe_id=recipe_id,
            notes=["DeepSeek-R1 rinde mejor con parser de reasoning explícito."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"reasoning_parser": "deepseek_r1"},
            suggested_tuning={
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            },
        )

    if "deepseek-v3" in repo or "deepseekv3forcausallm" in archs:
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="deepseek-v3",
            notes=["DeepSeek-V3 suele beneficiarse de parser de tools cuando se usa como assistant."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "hermes"},
            suggested_tuning={
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    if "internvl" in repo or "internvlchatmodel" in archs:
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="internvl-chat",
            notes=["InternVL multimodal suele agradecer modo solo lenguaje para uso texto y menor VRAM."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={},
            suggested_tuning={
                "language_model_only": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    if "llama-4" in repo or "llama4forconditionalgeneration" in archs:
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="llama4-multimodal",
            notes=["Llama 4 multimodal puede servirse en modo solo lenguaje cuando se busca uso textual."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={},
            suggested_tuning={
                "language_model_only": True,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    if "glm-4" in repo or "glm4" in repo:
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="glm4",
            notes=["GLM-4 suele requerir Transformers backend para mejor compatibilidad."],
            model_impl="transformers",
            runner="generate",
            required_overrides=required,
            suggested_config={},
            suggested_tuning={
                "enable_prefix_caching": True,
                "enable_chunked_prefill": False,
            },
        )

    if "granite" in repo or "graniteforcausallm" in archs:
        return VLLMRecipe(
            recipe_id="granite-chat",
            notes=["Granite suele funcionar mejor vía Transformers backend con parser de tools dedicado."],
            model_impl="transformers",
            runner="generate",
            required_overrides=["tool_call_parser"],
            suggested_config={"tool_call_parser": "granite"},
            suggested_tuning={"enable_prefix_caching": True},
        )

    if "glmforcausallm" in archs:
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="glm4",
            notes=["GLM suele requerir Transformers backend para mejor compatibilidad."],
            model_impl="transformers",
            runner="generate",
            required_overrides=required,
            suggested_config={},
            suggested_tuning={"enable_prefix_caching": True, "enable_chunked_prefill": False},
        )

    if "functiongemma" in repo or "gemma" in repo and "function" in repo:
        required = ["tool_call_parser"]
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="functiongemma",
            notes=["FunctionGemma necesita parser de tools tipo Gemma para tool calling fiable."],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "gemma"},
            suggested_tuning={"enable_prefix_caching": True},
        )

    return None
