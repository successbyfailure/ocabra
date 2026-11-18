"""NVML PIDs of the host's namespace mapped to the worker (Mapache's 1660)."""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.core.gpu_manager import GPUManager


def _gm(monkeypatch, after: set[int]) -> GPUManager:
    gm = GPUManager()
    monkeypatch.setattr(gm, "raw_compute_pids", lambda gpus: set(after))
    return gm


def test_a_new_foreign_pid_becomes_the_worker(monkeypatch):
    gm = _gm(monkeypatch, {111, 2095806})
    monkeypatch.setattr("os.path.exists", lambda p: False)  # not in our namespace
    assert gm.adopt_new_processes([0], {111}, 4242) == 2095806
    proc = gm._to_process_info(
        SimpleNamespace(pid=2095806, usedGpuMemory=3776 * 1024 * 1024), "compute"
    )
    assert proc.pid == 4242 and proc.used_vram_mb == 3776
    gm.forget_worker_pid(4242)
    assert (
        gm._to_process_info(SimpleNamespace(pid=2095806, usedGpuMemory=0), "compute").pid == 2095806
    )


def test_nothing_to_do_when_nvml_speaks_our_pids(monkeypatch):
    gm = _gm(monkeypatch, {4242})
    monkeypatch.setattr("os.path.exists", lambda p: True)
    assert gm.adopt_new_processes([0], set(), 4242) is None
    assert gm._pid_alias == {}


def test_two_new_pids_are_ambiguous(monkeypatch):
    gm = _gm(monkeypatch, {1, 2})
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert gm.adopt_new_processes([0], set(), 4242) is None


def test_a_new_pid_that_does_not_look_like_this_load_is_not_adopted(monkeypatch):
    # The only new PID is an embedding (3058 MB) reloaded at the same moment
    # as qwen3:30b (expected ~4400 MB): not adopted.
    gm = _gm(monkeypatch, {985})
    monkeypatch.setattr("os.path.exists", lambda p: False)
    monkeypatch.setattr(
        gm,
        "_read_local_processes",
        lambda g: [SimpleNamespace(pid=985, used_vram_mb=1198, process_type="compute")],
    )
    assert gm.adopt_new_processes([0], set(), 4242, expected_mb=4416) is None
    monkeypatch.setattr(
        gm,
        "_read_local_processes",
        lambda g: [SimpleNamespace(pid=985, used_vram_mb=4262, process_type="compute")],
    )
    assert gm.adopt_new_processes([0], set(), 4242, expected_mb=4416) == 985
