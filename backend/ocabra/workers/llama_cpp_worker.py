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

    os.execvpe(args.server_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
