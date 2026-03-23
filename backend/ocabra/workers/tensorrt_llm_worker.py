#!/usr/bin/env python
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra TensorRT-LLM worker wrapper")
    parser.add_argument("--launch-mode", default="binary", choices=["binary", "module"])
    parser.add_argument("--python-bin", default="/usr/bin/python3")
    parser.add_argument("--serve-module", default="tensorrt_llm.commands.serve")
    parser.add_argument("--serve-bin", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", default="tensorrt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _serve_args(args: argparse.Namespace) -> list[str]:
    cmd = [
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
        cmd.extend(["--max-batch-size", str(args.max_batch_size)])
    if args.max_num_tokens:
        cmd.extend(["--max-num-tokens", str(args.max_num_tokens)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def main() -> None:
    args = parse_args()
    serve_args = _serve_args(args)

    if args.launch_mode == "module":
        cmd = [args.python_bin, "-m", args.serve_module, *serve_args]
        os.execvpe(args.python_bin, cmd, os.environ.copy())

    cmd = [args.serve_bin, *serve_args]
    os.execvpe(args.serve_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
