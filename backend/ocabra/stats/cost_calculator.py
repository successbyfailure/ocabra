"""
Estimate OpenAI-equivalent cost for local inference requests.

Provides a reference point for how much the same workload would cost
on OpenAI's API, helping users understand the value of self-hosting.
"""
from __future__ import annotations

import re

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]", re.IGNORECASE)


def classify_model_tier(
    model_id: str,
    backend_type: str | None,
    request_kind: str | None,
) -> str:
    """Classify a model into a pricing tier based on its metadata.

    Returns one of: ``"small"``, ``"medium"``, ``"large"``, ``"embedding"``,
    ``"audio_stt"``, ``"tts"``, ``"image"``.
    """
    if request_kind:
        rk = request_kind.lower()
        if "embedding" in rk:
            return "embedding"
        if "audio_transcription" in rk:
            return "audio_stt"
        if rk == "tts":
            return "tts"
        if rk == "image_generation":
            return "image"

    # Try to extract parameter count from model_id
    match = _PARAM_RE.search(model_id)
    if match:
        params_b = float(match.group(1))
        if params_b < 10:
            return "small"
        if params_b <= 34:
            return "medium"
        return "large"

    # Safe default
    return "medium"


def get_tier_pricing(tier: str) -> tuple[float, float]:
    """Return ``(input_price_per_1M_tokens, output_price_per_1M_tokens)`` in USD.

    Prices are read from :pydata:`ocabra.config.settings` with sensible
    defaults matching OpenAI public pricing as of 2025.
    """
    from ocabra.config import settings

    _MAP: dict[str, tuple[float, float]] = {
        "small": (settings.openai_ref_small_input, settings.openai_ref_small_output),
        "medium": (settings.openai_ref_medium_input, settings.openai_ref_medium_output),
        "large": (settings.openai_ref_large_input, settings.openai_ref_large_output),
        "embedding": (settings.openai_ref_embedding_input, 0.0),
        "audio_stt": (settings.openai_ref_audio_stt_input, 0.0),
        "tts": (settings.openai_ref_tts_input, 0.0),
        "image": (settings.openai_ref_image_input, 0.0),
    }
    return _MAP.get(tier, (2.50, 10.00))


def estimate_request_cost(input_tokens: int, output_tokens: int, tier: str) -> float:
    """Return the estimated USD cost for a single request."""
    input_rate, output_rate = get_tier_pricing(tier)
    return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000


def estimate_cost_for_rows(rows: list) -> float:
    """Sum estimated cost across a list of ``RequestStat``-like objects.

    Each row must expose ``.input_tokens``, ``.output_tokens``,
    ``.model_id``, ``.backend_type``, and ``.request_kind``.
    Tier classification is cached per ``model_id`` to avoid repeated regex.
    """
    tier_cache: dict[str, str] = {}
    total = 0.0
    for r in rows:
        mid = r.model_id
        if mid not in tier_cache:
            tier_cache[mid] = classify_model_tier(mid, r.backend_type, r.request_kind)
        tier = tier_cache[mid]
        total += estimate_request_cost(
            max(0, int(r.input_tokens or 0)),
            max(0, int(r.output_tokens or 0)),
            tier,
        )
    return total
