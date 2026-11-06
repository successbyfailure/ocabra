"""One source of truth for "how many slots does this worker have" (2026-11-06).

There were three implementations and two disagreed. The fair-share gate
(``fairness_slots``) already fell back to the configured llama.cpp
``parallel_slots``; admission (``effective_inflight_cap``) only ever read
``_runtime.parallelism`` and otherwise assumed one slot. That value is written
only by the runtime-preset coordinator, so every llama.cpp model without a
preset ladder ran on a 4-slot gate while being admitted as a 1-slot worker:
cap 1 x max_queued_per_slot = 4, so the 5th concurrent request got a 429 with
three slots idle and room to queue twelve more.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.api.openai._deps import fairness_slots
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState
from ocabra.core.model_manager_helpers import resolve_worker_slots
from ocabra.core.worker_pool import WorkerPool


def _state(backend_type: str, extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(backend_type=backend_type, extra_config=extra or {})


class TestResolveWorkerSlots:
    def test_configured_parallel_slots_without_a_preset(self) -> None:
        """The regression: this returned nothing for admission before."""
        assert (
            resolve_worker_slots(_state("llamacpp_mks", {"llama_cpp": {"parallel_slots": 4}})) == 4
        )

    def test_runtime_preset_wins_over_configured(self) -> None:
        """The coordinator's value reflects the preset actually loaded, so it
        must beat whatever the model config says."""
        state = _state(
            "llama_cpp", {"_runtime": {"parallelism": 2}, "llama_cpp": {"parallel_slots": 4}}
        )
        assert resolve_worker_slots(state) == 2

    def test_camel_case_and_use_case_fallback(self) -> None:
        assert resolve_worker_slots(_state("llama_cpp", {"llama_cpp": {"parallelSlots": 3}})) == 3
        assert resolve_worker_slots(_state("llama_cpp", {"use_case": {"slots": 5}})) == 5

    def test_none_for_backends_with_no_slot_pool(self) -> None:
        """Honest None, so the dashboard renders "N en curso" instead of a
        fabricated ratio — whisper once showed "1/32" from the admission cap."""
        assert resolve_worker_slots(_state("whisper")) is None
        assert resolve_worker_slots(_state("ollama")) is None

    def test_piper_declares_its_own_ceiling(self) -> None:
        assert resolve_worker_slots(_state("piper")) == 4

    def test_garbage_values_do_not_crash(self) -> None:
        assert (
            resolve_worker_slots(_state("llama_cpp", {"llama_cpp": {"parallel_slots": "x"}}))
            is None
        )
        assert (
            resolve_worker_slots(_state("llama_cpp", {"_runtime": {"parallelism": None}})) is None
        )


class TestAdmissionMatchesTheGate:
    @staticmethod
    def _mm(backend_type: str, extra: dict) -> tuple[ModelManager, str, ModelState]:
        mm = ModelManager(WorkerPool())
        model_id = f"{backend_type}/m"
        state = ModelState(
            model_id=model_id,
            display_name=model_id,
            backend_type=backend_type,
            load_policy=LoadPolicy.ON_DEMAND,
            extra_config=extra,
        )
        mm._states[model_id] = state
        return mm, model_id, state

    def test_llama_cpp_without_preset_admits_its_real_slot_count(self, monkeypatch) -> None:
        monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
        mm, model_id, _ = self._mm("llamacpp_mks", {"llama_cpp": {"parallel_slots": 4}})
        # Was 4 (1 slot x 4): four concurrent, nothing queued.
        assert mm.effective_inflight_cap(model_id, 32) == 16

    @pytest.mark.parametrize(
        ("backend_type", "extra"),
        [
            ("llamacpp_mks", {"llama_cpp": {"parallel_slots": 4}}),
            ("llama_cpp", {"_runtime": {"parallelism": 2}}),
            ("whisper", {}),
            ("ollama", {}),
            ("piper", {}),
        ],
    )
    def test_gate_and_admission_never_disagree(self, backend_type, extra, monkeypatch) -> None:
        """The invariant that keeps them from drifting again: admission must
        scale the cap off exactly the slot count the gate sizes itself with."""
        monkeypatch.setattr("ocabra.core.model_manager.settings.max_queued_per_slot", 4)
        mm, model_id, state = self._mm(backend_type, extra)
        gate_slots = fairness_slots(state)
        assert mm.effective_inflight_cap(model_id, 32) == min(32, gate_slots * 4)
