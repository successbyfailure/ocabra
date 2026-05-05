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

    # Qwen3.5 / Qwen3.6 multimodal MoE (linear+full hybrid attention).
    # Architecture: Qwen3_5MoeForConditionalGeneration. Repo names use
    # "qwen3.5", "qwen3.6". Native 256K context, but on a 24 GB card you'll
    # cap around 16-32 K depending on quant + concurrency.
    if (
        "qwen3_5moeforconditionalgeneration" in archs
        or "qwen3.5" in repo
        or "qwen3.6" in repo
    ):
        required = ["tool_call_parser"]
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="qwen3.5-moe-multimodal",
            notes=[
                "Qwen3.5 / Qwen3.6 MoE multimodal con linear/full attention hibrida.",
                "tool_call_parser=qwen3_xml en vLLM 0.19+ (qwen3_json fue eliminado).",
                "language_model_only=true ahorra ~1-2 GB de los towers vision/audio.",
                "kv_cache_dtype=fp8 funciona en Ampere para esta familia.",
            ],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "qwen3_xml"},
            suggested_tuning={
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "language_model_only": True,
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 8,
                "trust_remote_code": True,
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
                "tool_call_parser": "qwen3_xml",
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
                "tool_call_parser": "qwen3_xml",
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
        return VLLMRecipe(
            recipe_id="deepseek-r1",
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

    # Gemma 4 (Google, abr-2026). Multimodal text+vision+audio.
    # Sliding+full attention hybrid every 6 layers — vLLM 0.19 prefix caching
    # still has rough edges with this pattern (engine dies on first request
    # of the 26B-A4B variant). KV cache fp8 is hardcoded to fp8_e4m3 in the
    # Gemma 4 attention path, which only works on Hopper/Blackwell — Ampere
    # must use the bf16 default.
    if (
        "gemma4forconditionalgeneration" in archs
        or "gemma4forcausallm" in archs
        or "gemma-4" in repo
        or ("gemma" in repo and "26b-a4b" in repo)
        or ("gemma" in repo and "e4b" in repo)
        or ("gemma" in repo and "e2b" in repo)
        or ("gemma" in repo and "31b" in repo)
    ):
        required = ["tool_call_parser"]
        if not tokenizer_has_chat_template:
            # Gemma 4 ships chat_template.jinja next to the tokenizer; vLLM/
            # transformers picks it up automatically. Only flag if neither
            # tokenizer_config nor a sidecar file is present.
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="gemma-4",
            notes=[
                "Gemma 4 multimodal (texto+vision+audio).",
                "enable_prefix_caching=false: bug en vLLM 0.19 con sliding+full "
                "attention de Gemma 4 26B-A4B (engine core muere tras 1ª request).",
                "kv_cache_dtype default (bf16): Gemma 4 hardcodea fp8_e4m3 que "
                "solo soporta Hopper+. En Ampere hay que dejarlo en bf16.",
                "tool_call_parser=gemma4 (registrado en vLLM 0.19 para esta familia).",
                "language_model_only=true: ahorra ~1-2 GB del vision tower si solo "
                "se usa texto.",
            ],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "gemma4"},
            suggested_tuning={
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "language_model_only": True,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 16,
            },
        )

    # Gemma 3 / Gemma 3n (Google, 2025). Tool-use mejorado vs Gemma 2.
    if (
        "gemma3forconditionalgeneration" in archs
        or "gemma3forcausallm" in archs
        or "gemma3nforconditionalgeneration" in archs
        or "gemma3nforcausallm" in archs
        or "gemma-3" in repo
    ):
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="gemma-3",
            notes=[
                "Gemma 3 / 3n con tool-use estable. Ampere: kv_cache_dtype default "
                "(no fp8) por la misma limitacion que Gemma 4.",
            ],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "pythonic"},
            suggested_tuning={
                "enable_prefix_caching": True,
                "language_model_only": True,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.9,
            },
        )

    # NVIDIA Nemotron-H (hibrido Mamba2 + attention). KV cache pequenisimo
    # gracias a las capas SSM, asi que admite contextos largos con mucha
    # concurrencia incluso en 24 GB.
    if (
        "nemotronhforcausallm" in archs
        or "nemotron-h" in repo
        or "nemotron-3-nano" in repo
    ):
        required = []
        if not tokenizer_has_chat_template:
            required.append("chat_template")
        return VLLMRecipe(
            recipe_id="nemotron-h",
            notes=[
                "Nemotron-H (Mamba2 + attention hibrido). KV cache pequeno → ctx "
                "largo + alta concurrencia OK.",
                "trust_remote_code obligatorio (auto_map en config.json).",
                "kv_cache_dtype=fp8 funciona en Ampere (a diferencia de Gemma 4).",
            ],
            model_impl="vllm",
            runner="generate",
            required_overrides=required,
            suggested_config={"tool_call_parser": "hermes"},
            suggested_tuning={
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "trust_remote_code": True,
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 32,
            },
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
