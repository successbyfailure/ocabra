"""LoRA adapters on llama.cpp models (llama_cpp.lora)."""

from __future__ import annotations

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend


def test_lora_option_forms():
    n = LlamaCppBackend._normalize_lora
    assert n(None) is None and n([]) is None
    assert n("/m/pi.gguf") == [("/m/pi.gguf", 1.0)]
    assert n([{"path": "/m/a.gguf", "scale": 0.5}, "/m/b.gguf"]) == [
        ("/m/a.gguf", 0.5),
        ("/m/b.gguf", 1.0),
    ]
    with pytest.raises(ValueError):
        n([{"scale": 1}])
    with pytest.raises(ValueError):
        n("/m/a,b.gguf")


def test_lora_arguments():
    args = LlamaCppBackend.lora_args
    assert args(None) == []
    assert args([("/m/pi.gguf", 1.0)]) == ["--lora", "/m/pi.gguf"]
    assert args([("/m/a.gguf", 0.5), ("/m/b.gguf", 1.0)]) == [
        "--lora-scaled",
        "/m/a.gguf:0.5,/m/b.gguf:1",
    ]


def test_the_worker_wrapper_forwards_lora(monkeypatch):
    import sys

    from ocabra.workers import llama_cpp_worker as w

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llama_cpp_worker.py", "--server-bin", "/bin/llama-server", "--model-id", "m",
            "--model-path", "/m/base.gguf", "--port", "0", "--ctx-size", "4096",
            "--batch-size", "512", "--ubatch-size", "128", "--lora", "/m/pi.gguf",
        ],
    )  # fmt: skip
    cmd = w.build_command(w.parse_args())
    assert cmd[cmd.index("--lora") + 1] == "/m/pi.gguf"
