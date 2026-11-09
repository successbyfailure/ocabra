from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends.bitnet_backend import BitnetBackend
from ocabra.core.model_manager import ModelState
from ocabra.core.model_manager_helpers import (
    estimate_bitnet_vram_from_config,
    resolve_bitnet_gpu_layers,
)


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


@pytest.mark.asyncio
async def test_load_requires_port(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        with pytest.raises(ValueError, match="requires 'port'"):
            await backend.load("model", [0])


@pytest.mark.asyncio
async def test_load_success(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    # El binario tiene que existir: _resolve_microsoft_bin ya no devuelve una
    # ruta configurada que no esta en disco.
    servidor = tmp_path / "bitnet-server"
    servidor.write_bytes(b"#!/bin/sh\n")

    proc = _fake_proc(returncode=None)
    backend = BitnetBackend()
    with (
        patch("ocabra.backends.bitnet_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
        patch.object(BitnetBackend, "_metadata_extra_bins", return_value={}),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_server_bin = str(servidor)
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = True
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None
        mock_settings.bitnet_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        info = await backend.load("model", [0], port=18021, extra_config={})

    assert info.backend_type == "bitnet"
    assert info.port == 18021
    assert info.pid == 4242
    assert info.vram_used_mb == 0


@pytest.mark.asyncio
async def test_get_vram_estimate_from_gpu_layers(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_gpu_layers = 16
        estimate = await backend.get_vram_estimate_mb("model")
    assert estimate == 200


@pytest.mark.asyncio
async def test_capabilities_use_context_from_settings() -> None:
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.bitnet_ctx_size = 8192
        caps = await backend.get_capabilities("any-model")
    assert caps.chat is True
    assert caps.completion is True
    assert caps.streaming is True
    assert caps.context_length == 8192


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Bonsai-27B-Q1_0.gguf", True),
        ("Ternary-Bonsai-27B-Q1_0.gguf", True),
        ("ggml-model-i2_s.gguf", False),
        ("falcon3-1b-instruct-1.58bit-gguf-i2_s.gguf", False),
    ],
)
def test_is_prismml_model(filename: str, expected: bool) -> None:
    assert BitnetBackend._is_prismml_model(Path(filename)) is expected


def test_select_server_bin_prefers_prismml_for_bonsai() -> None:
    backend = BitnetBackend()
    with (
        patch.object(
            BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
        ),
        patch.object(BitnetBackend, "_resolve_microsoft_bin", return_value="/bin/bitnet-server"),
    ):
        assert (
            backend._select_server_bin(Path("Bonsai-27B-Q1_0.gguf")) == "/bin/prismml/bonsai-server"
        )
        assert backend._select_server_bin(Path("ggml-model-i2_s.gguf")) == "/bin/bitnet-server"


def test_select_server_bin_raises_when_prismml_missing() -> None:
    backend = BitnetBackend()
    with patch.object(BitnetBackend, "_resolve_prismml_bin", return_value=None):
        with pytest.raises(FileNotFoundError, match="PrismML llama.cpp fork"):
            backend._select_server_bin(Path("Bonsai-27B-Q1_0.gguf"))


def test_build_options_defaults_gpu_layers_for_prismml() -> None:
    backend = BitnetBackend()
    with patch("ocabra.backends.bitnet_backend.settings") as mock_settings:
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = True
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None

        micro = backend._build_options({}, is_prismml=False)
        bonsai = backend._build_options({}, is_prismml=True)
        # Explicit override still wins over the GPU-first Bonsai default.
        overridden = backend._build_options({"gpu_layers": 10}, is_prismml=True)

    assert micro["gpu_layers"] == 0
    assert bonsai["gpu_layers"] == 99
    assert overridden["gpu_layers"] == 10


def test_scheduler_defaults_prismml_to_gpu_and_uses_gguf_size(tmp_path: Path) -> None:
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    with gguf.open("wb") as handle:
        handle.truncate(600 * 1024 * 1024)
    state = ModelState(
        model_id="bitnet/prism-ml/Bonsai-27B-gguf",
        backend_model_id="prism-ml/Bonsai-27B-gguf",
        display_name="Bonsai 27B",
        backend_type="bitnet",
        extra_config={"model_path": str(gguf)},
    )

    assert resolve_bitnet_gpu_layers(state, 0) == 99
    assert (
        estimate_bitnet_vram_from_config(
            state,
            default_gpu_layers=0,
            models_dir=tmp_path,
        )
        >= 648
    )


def test_scheduler_respects_explicit_prismml_cpu_override() -> None:
    state = ModelState(
        model_id="bitnet/prism-ml/Bonsai-27B-gguf",
        display_name="Bonsai 27B",
        backend_type="bitnet",
        extra_config={"gpu_layers": 0},
    )

    assert resolve_bitnet_gpu_layers(state, 0) == 0


@pytest.mark.asyncio
async def test_quantised_kv_cache_forces_flash_attn(tmp_path: Path) -> None:
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    gguf.write_bytes(b"GGUF")

    proc = _fake_proc(returncode=None)
    sub = AsyncMock(return_value=proc)
    backend = BitnetBackend()
    with (
        patch("ocabra.backends.bitnet_backend.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=sub),
        patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
        patch.object(
            BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
        ),
    ):
        mock_settings.models_dir = str(tmp_path)
        mock_settings.bitnet_gpu_layers = 0
        mock_settings.bitnet_ctx_size = 4096
        mock_settings.bitnet_threads = None
        mock_settings.bitnet_batch_size = 512
        mock_settings.bitnet_ubatch_size = 128
        mock_settings.bitnet_parallel = 1
        mock_settings.bitnet_flash_attn = False
        mock_settings.bitnet_mlock = False
        mock_settings.bitnet_cache_type_k = None
        mock_settings.bitnet_cache_type_v = None
        mock_settings.bitnet_startup_timeout_s = 30
        mock_settings.cuda_device_order = "PCI_BUS_ID"

        await backend.load(
            "Bonsai-27B-Q1_0", [0], port=18022, extra_config={"cache_type_k": "q8_0"}
        )

    cmd = list(sub.call_args.args)
    assert "--cache-type-k" in cmd
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
    # A non-f16 KV cache requires flash-attention even though it was disabled.
    # The PrismML fork needs the explicit value form `--flash-attn on`.
    assert "--flash-attn" in cmd
    assert cmd[cmd.index("--flash-attn") + 1] == "on"


def test_resolve_model_file_reads_path_from_bitnet_section(tmp_path: Path) -> None:
    """Regression: ``/registry/local/register`` pins the file under
    ``extra_config.bitnet.model_path`` but the resolver only read the root, so
    a GGUF with an unrelated filename (e.g. an Ollama sha256 blob) was never found."""
    weights = tmp_path / "sha256-deadbeef"
    weights.write_bytes(b"GGUF")
    backend = BitnetBackend()

    resolved = backend._resolve_model_file(
        "bitnet/some-model", {"bitnet": {"model_path": str(weights)}}
    )

    assert resolved == weights


def test_resolve_model_file_still_accepts_root_model_path(tmp_path: Path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"GGUF")

    assert BitnetBackend()._resolve_model_file("bitnet/x", {"model_path": str(weights)}) == weights


async def test_kv_unified_lets_one_request_use_the_whole_context(tmp_path: Path) -> None:
    """Regression 2026-11-08: Bonsai-27B inherited the BitNet globals (4096 ctx,
    16 slots) and llama.cpp split the pool → 256 tokens per request. With
    ``kv_unified`` the PrismML server gets --kv-unified; the Microsoft build
    (no such flag) never does."""
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    gguf.write_bytes(b"GGUF")
    servidor = tmp_path / "bitnet-server"
    servidor.write_bytes(b"#!/bin/sh\n")

    async def load_cmd(prismml: bool) -> list[str]:
        proc = _fake_proc(returncode=None)
        sub = AsyncMock(return_value=proc)
        backend = BitnetBackend()
        with (
            patch("ocabra.backends.bitnet_backend.settings") as mock_settings,
            patch("asyncio.create_subprocess_exec", new=sub),
            patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(
                BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
            ),
            patch.object(BitnetBackend, "_is_prismml_model", return_value=prismml),
        ):
            mock_settings.models_dir = str(tmp_path)
            mock_settings.bitnet_server_bin = str(servidor)
            mock_settings.bitnet_gpu_layers = 0
            mock_settings.bitnet_ctx_size = 4096
            mock_settings.bitnet_threads = None
            mock_settings.bitnet_batch_size = 512
            mock_settings.bitnet_ubatch_size = 128
            mock_settings.bitnet_parallel = 16
            mock_settings.bitnet_flash_attn = False
            mock_settings.bitnet_mlock = False
            mock_settings.bitnet_cache_type_k = None
            mock_settings.bitnet_cache_type_v = None
            mock_settings.bitnet_startup_timeout_s = 30
            mock_settings.cuda_device_order = "PCI_BUS_ID"
            await backend.load(
                "Bonsai-27B-Q1_0",
                [1],
                port=18022,
                extra_config={"bitnet": {"ctx_size": 262144, "parallel": 4, "kv_unified": True}},
            )
        return list(sub.call_args.args)

    cmd = await load_cmd(prismml=True)
    assert cmd[cmd.index("--ctx-size") + 1] == "262144"
    assert cmd[cmd.index("--parallel") + 1] == "4"
    assert "--kv-unified" in cmd
    assert "--kv-unified" not in await load_cmd(prismml=False)


# --- Que binario, decidido por la cuantizacion del fichero ----------------
# PrismML aporta Q1_0 (40) y Q2_0 (41), tipos que solo existen en su fork.
# Decidirlo por el nombre falla con un modelo instalado como tag de Ollama: su
# fichero es un blob llamado por su sha256, asi que ninguna pista de nombre
# acierta y arrancaba el binario llano, que se aborta con
# GGML_ASSERT(info->type < GGML_TYPE_COUNT) — bonsai-2:27b, 2026-11-09.


def _gguf_con_tipo(path: Path, ftype: int | None, arch: str = "qwen35") -> Path:
    kv, n = b"", 0
    clave = b"general.architecture"
    kv += struct.pack("<Q", len(clave)) + clave + struct.pack("<I", 8)
    v = arch.encode()
    kv += struct.pack("<Q", len(v)) + v
    n += 1
    if ftype is not None:
        clave = b"general.file_type"
        kv += struct.pack("<Q", len(clave)) + clave + struct.pack("<I", 4)
        kv += struct.pack("<I", ftype)
        n += 1
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", n) + kv
    )
    return path


@pytest.mark.parametrize(("ftype", "espera"), [(40, True), (41, True), (15, False), (37, False)])
def test_el_binario_se_elige_por_la_cuantizacion(tmp_path: Path, ftype: int, espera: bool) -> None:
    # Nombre de blob: ninguna pista, como en una instalacion desde Ollama.
    ruta = _gguf_con_tipo(tmp_path / ("sha256-" + "ab" * 16), ftype)
    assert BitnetBackend._is_prismml_model(ruta) is espera


def test_un_blob_de_ollama_con_q1_0_va_al_fork(tmp_path: Path) -> None:
    """La regresion exacta: el nombre no dice nada y el fichero si."""
    ruta = _gguf_con_tipo(tmp_path / "sha256-4f99aed01b8a877e153f9aa6569a4440", 40)
    assert BitnetBackend._is_prismml_model(ruta) is True


def test_si_el_fichero_no_se_puede_leer_se_mira_el_nombre(tmp_path: Path) -> None:
    ilegible = tmp_path / "Bonsai-27B-Q1_0.gguf"
    ilegible.write_bytes(b"no soy un gguf")
    assert BitnetBackend._is_prismml_model(ilegible) is True
    otro = tmp_path / "sha256-deadbeef"
    otro.write_bytes(b"no soy un gguf")
    assert BitnetBackend._is_prismml_model(otro) is False


def test_sin_binario_de_microsoft_se_dice_claro(tmp_path: Path) -> None:
    """Antes devolvia la ruta configurada existiera o no, y se ejecutaba nada."""
    backend = BitnetBackend()
    ruta = _gguf_con_tipo(tmp_path / "modelo.gguf", 15)
    with (
        patch.object(BitnetBackend, "_metadata_extra_bins", return_value={}),
        patch("ocabra.backends.bitnet_backend.settings") as s,
    ):
        s.bitnet_server_bin = str(tmp_path / "no-existe")
        s.bitnet_prismml_server_bin = None
        assert backend._resolve_microsoft_bin() is None
        with pytest.raises(FileNotFoundError, match="not installed on this node"):
            backend._select_server_bin(ruta)


@pytest.mark.asyncio
async def test_mlock_segun_el_binario(tmp_path: Path) -> None:
    """--mlock paso a ser un valor de --load-mode en el fork actualizado.

    Pasarle la bandera vieja aborta el servidor: "error: invalid argument:
    --mlock", visto al subir el fork al commit 6447906 el 2026-11-09. El
    binario de Microsoft solo conoce --mlock.
    """
    gguf = tmp_path / "Bonsai-27B-Q1_0.gguf"
    gguf.write_bytes(b"GGUF")
    microsoft = tmp_path / "bitnet-server"
    microsoft.write_bytes(b"#!/bin/sh\n")

    async def args(prismml: bool) -> list[str]:
        sub = AsyncMock(return_value=_fake_proc(returncode=None))
        backend = BitnetBackend()
        with (
            patch("ocabra.backends.bitnet_backend.settings") as st,
            patch("asyncio.create_subprocess_exec", new=sub),
            patch.object(BitnetBackend, "_wait_for_startup", new=AsyncMock()),
            patch.object(
                BitnetBackend, "_resolve_prismml_bin", return_value="/bin/prismml/bonsai-server"
            ),
            patch.object(BitnetBackend, "_is_prismml_model", return_value=prismml),
        ):
            st.models_dir = str(tmp_path)
            st.bitnet_server_bin = str(microsoft)
            st.bitnet_gpu_layers = 0
            st.bitnet_ctx_size = 4096
            st.bitnet_threads = None
            st.bitnet_batch_size = 512
            st.bitnet_ubatch_size = 128
            st.bitnet_parallel = 1
            st.bitnet_flash_attn = False
            st.bitnet_mlock = True
            st.bitnet_cache_type_k = None
            st.bitnet_cache_type_v = None
            st.bitnet_startup_timeout_s = 30
            st.cuda_device_order = "PCI_BUS_ID"
            await backend.load("Bonsai-27B-Q1_0", [0], port=18099, extra_config={})
        return list(sub.await_args.args)

    nuevos = await args(True)
    assert "--mlock" not in nuevos
    assert "--load-mode" in nuevos
    assert nuevos[nuevos.index("--load-mode") + 1] == "mlock"

    viejos = await args(False)
    assert "--mlock" in viejos
    assert "--load-mode" not in viejos
