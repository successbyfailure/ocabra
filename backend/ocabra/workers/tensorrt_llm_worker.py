#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra TensorRT-LLM worker wrapper")
    parser.add_argument("--launch-mode", default="binary", choices=["binary", "module", "docker"])
    parser.add_argument("--python-bin", default="/usr/bin/python3")
    parser.add_argument("--serve-module", default="tensorrt_llm.commands.serve")
    parser.add_argument("--serve-bin", required=True)
    parser.add_argument("--docker-bin", default="/usr/bin/docker")
    parser.add_argument("--docker-image", default="nvcr.io/nvidia/tensorrt-llm/release:latest")
    parser.add_argument("--docker-models-mount-host", default="/docker/ai-models/ocabra/models")
    parser.add_argument("--docker-models-mount-container", default="/data/models")
    parser.add_argument("--docker-hf-cache-mount-host", default=None)
    parser.add_argument("--docker-hf-cache-mount-container", default="/data/hf_cache")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", default="trt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _serve_args(args: argparse.Namespace) -> list[str]:
    # TRT-LLM 1.0: trtllm-serve is a multi-command CLI, inference uses "serve" subcommand
    cmd = [
        "serve",
        args.engine_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--backend",
        args.backend,
    ]
    if args.tokenizer_path:
        cmd.extend(["--tokenizer", args.tokenizer_path])
    if args.max_batch_size:
        cmd.extend(["--max_batch_size", str(args.max_batch_size)])
    if args.max_num_tokens:
        cmd.extend(["--max_num_tokens", str(args.max_num_tokens)])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def _map_path_to_host(path: str, mount_container: str, mount_host: str) -> str:
    p = Path(path)
    container = Path(mount_container)
    host = Path(mount_host)
    try:
        rel = p.relative_to(container)
    except ValueError:
        return path
    return str(host / rel)


def _dockerize_serve_args(args: argparse.Namespace, serve_args: list[str]) -> list[str]:
    mapped = list(serve_args)
    # serve_args[0] = "serve" (subcommand), serve_args[1] = engine_dir
    if len(mapped) > 1:
        mapped[1] = _map_path_to_host(
            mapped[1],
            args.docker_models_mount_container,
            args.docker_models_mount_host,
        )
    if "--tokenizer" in mapped:
        idx = mapped.index("--tokenizer") + 1
        if idx < len(mapped):
            mapped[idx] = _map_path_to_host(
                mapped[idx],
                args.docker_models_mount_container,
                args.docker_models_mount_host,
            )
    return mapped


def main() -> None:
    args = parse_args()
    serve_args = _serve_args(args)

    if args.launch_mode == "module":
        cmd = [args.python_bin, "-m", args.serve_module, *serve_args]
        print("[trtllm_worker] exec:", " ".join(cmd), file=sys.stderr, flush=True)
        os.execvpe(args.python_bin, cmd, os.environ.copy())

    if args.launch_mode == "docker":
        docker_serve_args = _dockerize_serve_args(args, serve_args)
        cmd = [
            args.docker_bin,
            "run",
            "--rm",
            "--gpus",
            "all",
            "--ipc=host",
            f"--network=container:{os.environ.get('HOSTNAME', '')}",
            "-v",
            f"{args.docker_models_mount_host}:{args.docker_models_mount_host}",
        ]
        if args.docker_hf_cache_mount_host:
            cmd.extend([
                "-v",
                f"{args.docker_hf_cache_mount_host}:{args.docker_hf_cache_mount_host}",
            ])
        cmd.extend([
            args.docker_image,
            "trtllm-serve",
            *docker_serve_args,
        ])
        print("[trtllm_worker] exec:", " ".join(cmd), file=sys.stderr, flush=True)
        os.execvpe(args.docker_bin, cmd, os.environ.copy())

    cmd = [args.serve_bin, *serve_args]
    print("[trtllm_worker] exec:", " ".join(cmd), file=sys.stderr, flush=True)
    os.execvpe(args.serve_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
