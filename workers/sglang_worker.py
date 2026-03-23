#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra SGLang worker wrapper")
    parser.add_argument("--server-module", default="sglang.launch_server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        args.server_module,
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp",
        str(args.tp),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--served-model-name",
        args.served_model_name or args.model_id,
    ]
    if args.context_length:
        cmd.extend(["--context-length", str(args.context_length)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.disable_radix_cache:
        cmd.append("--disable-radix-cache")

    os.execvpe(sys.executable, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
