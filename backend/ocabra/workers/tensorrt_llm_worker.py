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
    parser.add_argument("--max-seq-len", type=int, default=None)
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
    if getattr(args, "max_seq_len", None):
        cmd.extend(["--max_seq_len", str(args.max_seq_len)])
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


def _serve_through_docker_agent(args: argparse.Namespace) -> int:
    """Docker launch mode: the docker agent runs the container, this relays it.

    The api has no Docker socket (SEG-03): the agent starts ``trtllm-serve``
    in the TensorRT-LLM image (image and mounts from its settings) and streams
    its output, which goes to this process's stdout — the worker log, as
    before. SIGTERM (unload) closes the stream, and the agent stops the
    container. Returns the container's exit code.
    """
    import asyncio
    import signal

    from ocabra.core import docker_ops

    gpus = [
        int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()
    ]
    params = {
        "engine_dir": args.engine_dir,
        "port": args.port,
        "backend": args.backend,
        "gpu_indices": gpus,
        "trust_remote_code": bool(args.trust_remote_code),
        "tokenizer_path": args.tokenizer_path or None,
        "max_batch_size": args.max_batch_size or None,
        "max_num_tokens": args.max_num_tokens or None,
        "max_seq_len": args.max_seq_len or None,
    }

    async def relay() -> int:
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        lines = docker_ops.stream("trtllm.serve", **params)
        try:
            async for line in lines:
                if line.startswith(docker_ops.EXIT_MARKER):
                    return int(line[len(docker_ops.EXIT_MARKER) :])
                print(line, flush=True)
        except asyncio.CancelledError:
            return 143
        finally:
            await lines.aclose()
        return 1

    print("[trtllm_worker] serving through the docker agent", file=sys.stderr, flush=True)
    return asyncio.run(relay())


def main() -> None:
    args = parse_args()
    serve_args = _serve_args(args)

    if args.launch_mode == "module":
        cmd = [args.python_bin, "-m", args.serve_module, *serve_args]
        print("[trtllm_worker] exec:", " ".join(cmd), file=sys.stderr, flush=True)
        os.execvpe(args.python_bin, cmd, os.environ.copy())

    if args.launch_mode == "docker":
        sys.exit(_serve_through_docker_agent(args))

    cmd = [args.serve_bin, *serve_args]
    print("[trtllm_worker] exec:", " ".join(cmd), file=sys.stderr, flush=True)
    os.execvpe(args.serve_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
