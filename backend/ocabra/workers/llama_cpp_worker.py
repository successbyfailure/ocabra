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
    # --- Sprint 17.4 — speculative decoding + concurrent slots ---
    parser.add_argument("--model-draft", default=None)
    parser.add_argument("--spec-type", default=None)
    parser.add_argument("--spec-draft-n-max", type=int, default=None)
    parser.add_argument("--spec-draft-n-min", type=int, default=None)
    parser.add_argument("--spec-draft-p-min", type=float, default=None)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--cont-batching", action="store_true")
    # Multimodal projector (vision). See backend_load.py's
    # LlamaCppLoadConfig.mmproj_model_id docstring for resolution rules.
    parser.add_argument("--mmproj", default=None)
    # Chat template override forwarded to llama-server. Accepts built-in names
    # (llama3, llama2, chatml, mistral, gemma, …) or a full Jinja string. Needed
    # for GGUFs that ship without a baked ``tokenizer.chat_template`` — see the
    # backend's option docstring for context.
    parser.add_argument("--chat-template", default=None)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
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
        # Newer llama-server takes a value: --flash-attn on|off|auto (a bare
        # flag consumes the next arg as its value and aborts). Older builds
        # accepted the bare form; "on" is understood by both.
        cmd.extend(["--flash-attn", "on"])
    if args.embedding:
        cmd.append("--embedding")
    # --- Sprint 17.1 (Tier 1) flags ---
    # Current llama-server dropped the separate --mlock/--no-mmap boolean
    # flags in favour of a single --load-mode {auto,none,mmap,mlock,
    # mmap+mlock,dio} — passing the old bare flags now aborts with
    # "error: invalid argument: --mlock". Map the two old booleans onto the
    # new mode: mlock alone means "no mmap, keep locked in RAM" in the new
    # scheme, so it must combine with --no-mmap's intent, not replace it.
    if args.mlock and args.no_mmap:
        load_mode = "mlock"
    elif args.mlock:
        load_mode = "mmap+mlock"
    elif args.no_mmap:
        load_mode = "none"
    else:
        load_mode = None
    if load_mode is not None:
        cmd.extend(["--load-mode", load_mode])
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

    # --- Sprint 17.4 ---
    if args.model_draft:
        cmd.extend(["--model-draft", args.model_draft])
    if args.spec_type:
        cmd.extend(["--spec-type", args.spec_type])
    if args.spec_draft_n_max is not None:
        cmd.extend(["--spec-draft-n-max", str(args.spec_draft_n_max)])
    if args.spec_draft_n_min is not None:
        cmd.extend(["--spec-draft-n-min", str(args.spec_draft_n_min)])
    if args.spec_draft_p_min is not None:
        cmd.extend(["--spec-draft-p-min", str(args.spec_draft_p_min)])
    if args.parallel is not None:
        cmd.extend(["--parallel", str(args.parallel)])
    if args.cont_batching:
        cmd.append("--cont-batching")
    if args.mmproj:
        cmd.extend(["--mmproj", args.mmproj])
    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])

    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    os.execvpe(args.server_bin, cmd, os.environ.copy())


if __name__ == "__main__":
    main()
