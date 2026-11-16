"""Does CUDA actually work on this machine? (not just "is a GPU visible")

NVML (what reports GPU names and memory) can work while CUDA compute doesn't:
a rented host whose GPU needs a reboot ("GPU Recovery Action: Reboot") or a
container without the driver's compute libraries shows the GPU fine, and every
llama.cpp model then loads "successfully" and runs on the CPU. ``cuInit`` is
the call that actually fails in those cases, so that is what we check.
"""

from __future__ import annotations

import ctypes
from typing import Any

_RESULT: dict[str, Any] | None = None


def probe_cuda() -> dict[str, Any]:
    """``{"ok", "devices", "error", "driver_cuda"}`` (cached).

    ``driver_cuda`` is the newest CUDA the driver supports (``"13.0"``): a
    backend built for a newer CUDA than that does not start here.

    ``ok`` is ``None`` when there is no NVIDIA driver at all (CPU-only node):
    nothing is broken, there is just no GPU to use.
    """
    global _RESULT
    if _RESULT is not None:
        return _RESULT
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        _RESULT = {"ok": None, "devices": 0, "error": "libcuda.so.1 not found"}
        return _RESULT
    version = ctypes.c_int(0)
    driver_cuda = None
    if lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value:
        driver_cuda = f"{version.value // 1000}.{(version.value % 1000) // 10}"
    rc = lib.cuInit(0)
    if rc != 0:
        _RESULT = {
            "ok": False,
            "devices": 0,
            "error": f"cuInit failed with CUDA error {rc}",
            "driver_cuda": driver_cuda,
        }
        return _RESULT
    count = ctypes.c_int(0)
    rc = lib.cuDeviceGetCount(ctypes.byref(count))
    if rc != 0 or count.value == 0:
        _RESULT = {
            "ok": False,
            "devices": count.value,
            "error": f"no CUDA devices (error {rc})",
            "driver_cuda": driver_cuda,
        }
        return _RESULT
    _RESULT = {"ok": True, "devices": count.value, "error": None, "driver_cuda": driver_cuda}
    return _RESULT
