"""Un borrador sin ``spec_type`` no puede llegar a llama-server.

Un draft model no es "aceleracion opcional": si se le pasa a llama-server sin
decirle de que tipo es, lo carga igual y se aborta al sondear su contexto
(``GGML_ASSERT(ubatch->embd_state)``, rc=-6). El modelo entero deja de cargar.

Pasó de verdad el 2026-11-08 con ``gemma4:12b``: su tag de Ollama trae una
capa ``draft`` (una cabeza ``gemma4-assistant``), oCabra la derivó al
``artifact`` a las 02:13 UTC y desde entonces cada carga murió con SIGABRT y
el modelo respondió 503 durante horas. Nadie habia configurado nada: el
borrador apareció solo.

llama.cpp intenta deducir el tipo del propio GGUF, pero solo reconoce el
tensor exacto ``blk.<ultima>.nextn.eh_proj.weight`` o la arquitectura
``dflash``; la cabeza de Gemma 4 lleva ``nextn.pre_projection`` y
``nextn.post_projection``, asi que su deducción se queda vacía.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend
from ocabra.registry.local_scanner import parse_gguf_speculative_type

_TIPO_STRING = 8


def _gguf(path: Path, arch: str | None, tensores: list[str]) -> Path:
    """Escribe un GGUF minimo: cabecera + metadatos + tabla de tensores."""

    kv = b""
    n_kv = 0
    if arch is not None:
        clave = b"general.architecture"
        valor = arch.encode()
        kv += struct.pack("<Q", len(clave)) + clave
        kv += struct.pack("<I", _TIPO_STRING)
        kv += struct.pack("<Q", len(valor)) + valor
        n_kv += 1

    tabla = b""
    for nombre in tensores:
        crudo = nombre.encode()
        tabla += struct.pack("<Q", len(crudo)) + crudo
        tabla += struct.pack("<I", 1)  # n_dims
        tabla += struct.pack("<Q", 4096)  # dims[0]
        tabla += struct.pack("<I", 0)  # tipo ggml
        tabla += struct.pack("<Q", 0)  # offset

    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensores))
        + struct.pack("<Q", n_kv)
        + kv
        + tabla
    )
    return path


# --- Deducción desde el fichero ------------------------------------------


def test_la_cabeza_de_gemma4_pide_mtp(tmp_path):
    """El caso real: los tensores que llama.cpp no reconoce."""
    ruta = _gguf(
        tmp_path / "d.gguf",
        "gemma4-assistant",
        ["nextn.pre_projection.weight", "nextn.post_projection.weight", "output_norm.weight"],
    )
    assert parse_gguf_speculative_type(ruta) == "draft-mtp"


def test_tambien_el_nextn_que_llama_cpp_si_reconoce(tmp_path):
    ruta = _gguf(tmp_path / "d.gguf", "qwen3next", ["blk.47.nextn.eh_proj.weight"])
    assert parse_gguf_speculative_type(ruta) == "draft-mtp"


def test_dflash_sin_cabeza_markov(tmp_path):
    ruta = _gguf(tmp_path / "d.gguf", "dflash", ["blk.0.attn_q.weight"])
    assert parse_gguf_speculative_type(ruta) == "draft-dflash"


def test_dflash_con_cabeza_markov_es_dspark(tmp_path):
    ruta = _gguf(tmp_path / "d.gguf", "dflash", ["markov_w1.weight", "blk.0.attn_q.weight"])
    assert parse_gguf_speculative_type(ruta) == "draft-dspark"


def test_un_gguf_normal_no_dice_nada(tmp_path):
    ruta = _gguf(tmp_path / "d.gguf", "gemma3", ["blk.0.attn_q.weight", "output_norm.weight"])
    assert parse_gguf_speculative_type(ruta) is None


@pytest.mark.parametrize(
    "contenido",
    [b"", b"GGUF", b"no soy un gguf en absoluto", b"GGUF" + struct.pack("<I", 1) + b"\x00" * 16],
)
def test_lo_ilegible_no_dice_nada(tmp_path, contenido):
    ruta = tmp_path / "roto.gguf"
    ruta.write_bytes(contenido)
    assert parse_gguf_speculative_type(ruta) is None


def test_un_fichero_que_no_existe_no_revienta(tmp_path):
    assert parse_gguf_speculative_type(tmp_path / "ni-existe.gguf") is None


# --- Lo que acaba en la linea de comandos --------------------------------


@pytest.mark.asyncio
async def test_el_borrador_del_artifact_llega_con_su_tipo(tmp_path):
    """Lo que arregla el incidente: el artifact solo sabe el fichero."""
    ruta = _gguf(tmp_path / "d.gguf", "gemma4-assistant", ["nextn.pre_projection.weight"])
    args = await LlamaCppBackend()._speculative_args(
        "llamacpp_mks/gemma4:12b", {"speculative": {"draft_model_id": str(ruta)}}
    )
    assert args == ["--spec-type", "draft-mtp", "--model-draft", str(ruta)]


@pytest.mark.asyncio
async def test_un_borrador_que_no_sabemos_conducir_no_se_pasa(tmp_path):
    """Preferible perder la especulacion que matar al worker."""
    ruta = _gguf(tmp_path / "d.gguf", "arquitectura-futura", ["blk.0.attn_q.weight"])
    args = await LlamaCppBackend()._speculative_args(
        "m", {"speculative": {"draft_model_id": str(ruta)}}
    )
    assert args == []


@pytest.mark.asyncio
async def test_nunca_sale_un_model_draft_sin_spec_type(tmp_path):
    """Invariante: es la combinacion que aborta el servidor.

    Cualquier forma de llegar aqui —tipo puesto a mano, deducido o ninguno—
    tiene que respetarla.
    """
    casos = [
        {"draft_model_id": str(_gguf(tmp_path / "a.gguf", "gemma4-assistant", ["nextn.x.weight"]))},
        {"draft_model_id": str(_gguf(tmp_path / "b.gguf", "dflash", ["markov_w1.weight"]))},
        {"draft_model_id": str(_gguf(tmp_path / "c.gguf", "otra", ["blk.0.attn_q.weight"]))},
        {"draft_model_id": str(_gguf(tmp_path / "d.gguf", "otra", [])), "draft_n": 4},
        {"spec_type": "draft-mtp"},
        {
            "spec_type": "draft-eagle3",
            "draft_model_id": str(_gguf(tmp_path / "e.gguf", "otra", [])),
        },
        {},
    ]
    backend = LlamaCppBackend()
    for caso in casos:
        args = await backend._speculative_args("m", {"speculative": caso})
        if "--model-draft" in args:
            assert "--spec-type" in args, caso


@pytest.mark.asyncio
async def test_el_tipo_configurado_a_mano_manda(tmp_path):
    """No se lee el fichero cuando ya esta dicho: el operador decide."""
    ruta = tmp_path / "d.gguf"
    ruta.write_bytes(b"GGUF no parseable")
    args = await LlamaCppBackend()._speculative_args(
        "m", {"speculative": {"spec_type": "draft-eagle3", "draft_model_id": str(ruta)}}
    )
    assert args == ["--spec-type", "draft-eagle3", "--model-draft", str(ruta)]


@pytest.mark.asyncio
async def test_la_auto_especulacion_sigue_sin_borrador():
    """qwen3.8: el bloque nextn va dentro del GGUF principal."""
    args = await LlamaCppBackend()._speculative_args(
        "m", {"speculative": {"spec_type": "draft-mtp", "draft_n": 4}}
    )
    assert args == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]


@pytest.mark.asyncio
async def test_sin_bloque_especulativo_no_hay_banderas():
    backend = LlamaCppBackend()
    assert await backend._speculative_args("m", {}) == []
    assert await backend._speculative_args("m", {"speculative": None}) == []


@pytest.mark.asyncio
async def test_los_topes_del_borrador_siguen_pasando(tmp_path):
    ruta = _gguf(tmp_path / "d.gguf", "gemma4-assistant", ["nextn.pre_projection.weight"])
    args = await LlamaCppBackend()._speculative_args(
        "m",
        {
            "speculative": {
                "draft_model_id": str(ruta),
                "draft_n": 6,
                "draft_min": 2,
                "draft_p_min": 0.7,
            }
        },
    )
    assert args[-6:] == [
        "--spec-draft-n-max",
        "6",
        "--spec-draft-n-min",
        "2",
        "--spec-draft-p-min",
        "0.7",
    ]


@pytest.mark.asyncio
async def test_los_topes_no_salen_si_el_borrador_se_descarta(tmp_path):
    """Sin borrador ni tipo, --spec-draft-* no tiene a que aplicarse."""
    ruta = _gguf(tmp_path / "d.gguf", "otra", ["blk.0.attn_q.weight"])
    args = await LlamaCppBackend()._speculative_args(
        "m", {"speculative": {"draft_model_id": str(ruta), "draft_n": 6}}
    )
    assert args == []
