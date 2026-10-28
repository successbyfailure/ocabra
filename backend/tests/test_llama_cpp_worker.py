from ocabra.workers.llama_cpp_worker import build_command, parse_args


def _args(*extra: str):
    import sys

    argv = [
        "--server-bin", "/bin/llama-server",
        "--model-id", "test/model",
        "--model-path", "/data/model.gguf",
        "--port", "18000",
        "--ctx-size", "4096",
        "--batch-size", "512",
        "--ubatch-size", "512",
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
