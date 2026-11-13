from ocabra.workers.llama_cpp_worker import build_command, parse_args


def _args(*extra: str):
    import sys

    argv = [
        "--server-bin",
        "/bin/llama-server",
        "--model-id",
        "test/model",
        "--model-path",
        "/data/model.gguf",
        "--port",
        "18000",
        "--ctx-size",
        "4096",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        *extra,
    ]
    old_argv = sys.argv
    sys.argv = ["llama_cpp_worker.py", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_neither_mlock_nor_no_mmap_omits_load_mode():
    cmd = build_command(_args())
    assert "--load-mode" not in cmd


def test_no_mmap_alone_maps_to_load_mode_none():
    cmd = build_command(_args("--no-mmap"))
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "none"


def test_mlock_alone_maps_to_load_mode_mmap_plus_mlock():
    """Old bare --mlock combined with mmap defaulting on. The new
    --load-mode scheme has no bare "keep the old default mmap behaviour
    AND lock" option other than mmap+mlock — "mlock" alone in the new
    scheme means mmap is OFF, which old --mlock never implied."""
    cmd = build_command(_args("--mlock"))
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "mmap+mlock"


def test_mlock_and_no_mmap_together_maps_to_load_mode_mlock():
    cmd = build_command(_args("--mlock", "--no-mmap"))
    idx = cmd.index("--load-mode")
    assert cmd[idx + 1] == "mlock"


def test_bare_mlock_flag_is_never_passed_to_llama_server():
    """Current llama-server rejects a bare --mlock with "error: invalid
    argument: --mlock" — regression guard for the CLI redesign that
    replaced it with --load-mode."""
    for extra in ((), ("--mlock",), ("--no-mmap",), ("--mlock", "--no-mmap")):
        cmd = build_command(_args(*extra))
        assert "--mlock" not in cmd
        assert "--no-mmap" not in cmd


def test_mmproj_omitted_when_not_set():
    cmd = build_command(_args())
    assert "--mmproj" not in cmd


def test_mmproj_forwarded_to_llama_server():
    cmd = build_command(_args("--mmproj", "/data/mmproj.gguf"))
    idx = cmd.index("--mmproj")
    assert cmd[idx + 1] == "/data/mmproj.gguf"


def test_props_endpoint_is_opt_in():
    assert "--props" not in build_command(_args())
    assert "--props" in build_command(_args("--props"))


def test_reasoning_controls_are_forwarded():
    cmd = build_command(
        _args(
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-format",
            "none",
            "--reasoning-budget",
            "0",
        )
    )

    assert "--jinja" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert cmd[cmd.index("--reasoning-format") + 1] == "none"
    assert cmd[cmd.index("--reasoning-budget") + 1] == "0"


def test_elastic_kv_flags_are_omitted_when_unset():
    cmd = build_command(_args())
    assert "--kv-unified" not in cmd
    assert "--no-kv-unified" not in cmd
    assert "--kv-unified-per-slot" not in cmd
    assert "--cache-ram" not in cmd
    assert "--cache-idle-slots" not in cmd
    assert "--no-cache-idle-slots" not in cmd


def test_elastic_kv_flags_are_forwarded():
    cmd = build_command(
        _args(
            "--kv-unified",
            "--kv-unified-per-slot",
            "32768",
            "--cache-ram",
            "0",
            "--no-cache-idle-slots",
        )
    )
    assert "--kv-unified" in cmd
    assert cmd[cmd.index("--kv-unified-per-slot") + 1] == "32768"
    assert cmd[cmd.index("--cache-ram") + 1] == "0"
    assert "--no-cache-idle-slots" in cmd


def test_split_mode_tensor_is_passed_through():
    cmd = build_command(_args("--split-mode", "tensor"))
    idx = cmd.index("--split-mode")
    assert cmd[idx + 1] == "tensor"


def test_draft_placement_flags_reach_llama_server():
    """2026-11-13: el MTP de 512 expertos de Qwen3.8-Flash-Next no cabia en la
    3090 junto al modelo; con sus expertos en RAM si."""
    cmd = build_command(
        _args(
            "--model-draft", "/m/mtp.gguf",
            "--spec-type", "draft-mtp",
            "--cpu-moe-draft",
            "--n-cpu-moe-draft", "1",
            "--device-draft", "CUDA1",
        )
    )
    assert "--cpu-moe-draft" in cmd
    assert cmd[cmd.index("--n-cpu-moe-draft") + 1] == "1"
    assert cmd[cmd.index("--device-draft") + 1] == "CUDA1"


def test_no_draft_placement_flags_by_default():
    cmd = build_command(_args("--model-draft", "/m/mtp.gguf", "--spec-type", "draft-mtp"))
    assert "--cpu-moe-draft" not in cmd and "--device-draft" not in cmd
