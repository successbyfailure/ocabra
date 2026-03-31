"""Tests for tensorrt_llm_worker.py — verifies command construction for TRT-LLM 1.0."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from ocabra.workers.tensorrt_llm_worker import (
    _dockerize_serve_args,
    _map_path_to_host,
    _serve_args,
)


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "launch_mode": "binary",
        "python_bin": "/usr/bin/python3",
        "serve_module": "tensorrt_llm.commands.serve",
        "serve_bin": "/usr/local/bin/trtllm-serve",
        "docker_bin": "/usr/bin/docker",
        "docker_image": "nvcr.io/nvidia/tensorrt-llm/release:latest",
        "docker_models_mount_host": "/docker/ai-models/ocabra/models",
        "docker_models_mount_container": "/data/models",
        "docker_hf_cache_mount_host": None,
        "docker_hf_cache_mount_container": "/data/hf_cache",
        "model_id": "tensorrt_llm/My--Engine",
        "engine_dir": "/data/models/tensorrt_llm/My--Engine/engine",
        "host": "127.0.0.1",
        "port": 18000,
        "backend": "trt",
        "tokenizer_path": None,
        "max_batch_size": None,
        "max_num_tokens": None,
        "trust_remote_code": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_serve_args_includes_serve_subcommand() -> None:
    """TRT-LLM 1.0: trtllm-serve requires 'serve' as first subcommand."""
    args = _args()
    cmd = _serve_args(args)
    assert cmd[0] == "serve", f"Expected 'serve' subcommand, got {cmd[0]!r}"
    assert cmd[1] == args.engine_dir


def test_serve_args_tokenizer_included_when_set() -> None:
    args = _args(tokenizer_path="/data/models/huggingface/Org--Model")
    cmd = _serve_args(args)
    assert "--tokenizer" in cmd
    assert cmd[cmd.index("--tokenizer") + 1] == "/data/models/huggingface/Org--Model"


def test_serve_args_no_tokenizer_when_none() -> None:
    args = _args(tokenizer_path=None)
    cmd = _serve_args(args)
    assert "--tokenizer" not in cmd


def test_serve_args_max_batch_size() -> None:
    args = _args(max_batch_size=4)
    cmd = _serve_args(args)
    assert "--max_batch_size" in cmd
    assert cmd[cmd.index("--max_batch_size") + 1] == "4"


def test_map_path_to_host_maps_container_prefix() -> None:
    result = _map_path_to_host(
        "/data/models/tensorrt_llm/foo/engine",
        "/data/models",
        "/docker/ai-models/ocabra/models",
    )
    assert result == "/docker/ai-models/ocabra/models/tensorrt_llm/foo/engine"


def test_map_path_to_host_leaves_unrelated_path() -> None:
    result = _map_path_to_host("/tmp/other", "/data/models", "/docker/ai-models/ocabra/models")
    assert result == "/tmp/other"


def test_dockerize_serve_args_maps_engine_dir() -> None:
    args = _args()
    # serve_args: ["serve", "/data/models/tensorrt_llm/My--Engine/engine", "--host", ...]
    serve = _serve_args(args)
    dockerized = _dockerize_serve_args(args, serve)
    # Engine dir at index 1 should be remapped to host path
    assert dockerized[0] == "serve"
    assert dockerized[1] == "/docker/ai-models/ocabra/models/tensorrt_llm/My--Engine/engine"


def test_dockerize_serve_args_maps_tokenizer_path() -> None:
    args = _args(tokenizer_path="/data/models/huggingface/Org--Model")
    serve = _serve_args(args)
    dockerized = _dockerize_serve_args(args, serve)
    assert "--tokenizer" in dockerized
    idx = dockerized.index("--tokenizer") + 1
    assert dockerized[idx] == "/docker/ai-models/ocabra/models/huggingface/Org--Model"
