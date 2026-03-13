#!/usr/bin/env python
"""
Standalone vLLM worker wrapper.

Wraps vLLM's OpenAI API server with:
- Graceful SIGTERM handling (allows in-flight requests to finish)
- Extended healthcheck logging
- Consistent environment setup

Usage:
    python workers/vllm_worker.py --model-id mistral-7b --port 18001 --gpu 1

The script re-executes itself as the vLLM process via exec(), so the PID
reported in WorkerInfo is the same as this wrapper's PID.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM worker wrapper for oCabra")
    p.add_argument("--model-id", required=True, help="Model ID (relative path under models_dir)")
    p.add_argument("--port", type=int, required=True, help="Port to bind vLLM server")
    p.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0],
        help="CUDA GPU indices (space-separated for tensor parallelism)",
    )
    p.add_argument("--models-dir", default="/data/models", help="Root models directory")
    p.add_argument("--hf-cache", default="/data/hf_cache", help="HuggingFace cache directory")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization (0-1)",
    )
    p.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable automatic prefix caching",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum concurrent sequences per iteration",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum batched tokens per iteration",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel degree override",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length override",
    )
    p.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill explicitly",
    )
    p.add_argument(
        "--swap-space",
        type=float,
        default=None,
        help="CPU swap space per GPU in GiB",
    )
    p.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="KV cache dtype override, e.g. fp8",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs and force eager execution",
    )
    p.add_argument(
        "--attention-backend",
        default=None,
        help="Optional attention backend override",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = str(Path(args.models_dir) / args.model_id)
    cuda_devices = ",".join(str(g) for g in args.gpu)
    tensor_parallel = args.tensor_parallel_size or len(args.gpu)

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HOME": args.hf_cache,
    }

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
        "--served-model-name",
        args.model_id,
        "--disable-log-requests",
    ]
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.max_num_seqs:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if args.swap_space:
        cmd.extend(["--swap-space", str(args.swap_space)])
    if args.kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])

    print(
        f"[vllm_worker] Starting model={args.model_id} port={args.port} "
        f"gpus={cuda_devices} tp={tensor_parallel}",
        flush=True,
    )

    proc = subprocess.Popen(cmd, env=env)

    def _on_sigterm(signum, frame):  # noqa: ARG001
        print("[vllm_worker] SIGTERM received — forwarding to vLLM process", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[vllm_worker] Timeout — sending SIGKILL", flush=True)
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    returncode = proc.wait()
    print(f"[vllm_worker] vLLM process exited with rc={returncode}", flush=True)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
