#!/usr/bin/env python
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra llama.cpp worker wrapper")
    parser.add_argument("--server-bin", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ctx-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--ubatch-size", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--alias", default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    # --- Sprint 17.1 (Tier 1) flags ---
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument("--rope-freq-base", type=float, default=None)
    parser.add_argument("--rope-freq-scale", type=float, default=None)
    # Sprint 17.2 — KV cache quantization
    parser.add_argument("--cache-type-k", default=None)
    parser.add_argument("--cache-type-v", default=None)
    # --- Sprint 17.3 (Multi-GPU + MoE) ---
    parser.add_argument("--main-gpu", type=int, default=None)
    parser.add_argument("--tensor-split", default=None)
    parser.add_argument(
        "--split-mode",
        choices=("layer", "row", "none"),
        default=None,
    )
    parser.add_argument("--n-cpu-moe", type=int, default=None)
    parser.add_argument("--override-tensor", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        args.server_bin,
        "--model",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--alias",
        args.alias or args.model_id,
    ]
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.flash_attn:
        cmd.append("--flash-attn")
    if args.mlock:
        cmd.append("--mlock")
    if args.embedding:
        cmd.append("--embedding")
    # --- Sprint 17.1 (Tier 1) flags ---
    if args.no_mmap:
        cmd.append("--no-mmap")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.no_kv_offload:
        cmd.append("--no-kv-offload")
    if args.rope_freq_base is not None:
        cmd.extend(["--rope-freq-base", str(args.rope_freq_base)])
    if args.rope_freq_scale is not None:
        cmd.extend(["--rope-freq-scale", str(args.rope_freq_scale)])
    # Sprint 17.2 — KV cache quantization
    if args.cache_type_k:
        cmd.extend(["--cache-type-k", str(args.cache_type_k)])
    if args.cache_type_v:
        cmd.extend(["--cache-type-v", str(args.cache_type_v)])
    # --- Sprint 17.3 (Multi-GPU + MoE) ---
    if args.main_gpu is not None:
        cmd.extend(["--main-gpu", str(args.main_gpu)])
    if args.tensor_split:
        cmd.extend(["--tensor-split", args.tensor_split])
    if args.split_mode is not None:
        cmd.extend(["--split-mode", args.split_mode])
    if args.n_cpu_moe is not None:
        cmd.extend(["--n-cpu-moe", str(args.n_cpu_moe)])
    if args.override_tensor:
        cmd.extend(["--override-tensor", args.override_tensor])

    os.execvpe(args.server_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
