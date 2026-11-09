"""Un modelo recien dado de alta tiene que poder cargar.

Dar de alta deja ``extra_config`` vacio cuando quien llama no manda uno, y
desde Explore nunca lo manda. Daba igual mientras los pesos los servia Ollama,
que decide capas y contexto por su cuenta; desde 0.47.0 un pull de Ollama
registra ademas un despliegue ``llamacpp_mks``, y llama.cpp solo hace lo que
se le dice: sin configuracion cae a ``LLAMA_CPP_GPU_LAYERS=0`` (CPU) y
``LLAMA_CPP_CTX_SIZE=4096``, asi que un modelo de 20 GiB intenta correr en CPU
y no carga nunca.

Reproducido el 2026-11-09: ``ornith-1.5:35b``, ``flint:40b`` y
``nemotron-3.5-lightning:30b`` se instalaron desde Explore y los tres quedaron
con ``extra_config`` a NULL.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from ocabra.core.registration_defaults import (
    _FALLBACK_CTX,
    _MAX_CTX,
    llama_cpp_launch_defaults,
)

_TIPO_STRING = 8
_TIPO_U32 = 4


def _gguf(path: Path, arch: str, ctx: int | None) -> Path:
    kv, n = b"", 0
    for clave, tipo, valor in [
        ("general.architecture", _TIPO_STRING, arch),
        (f"{arch}.context_length", _TIPO_U32, ctx),
    ]:
        if valor is None:
            continue
        c = clave.encode()
        kv += struct.pack("<Q", len(c)) + c + struct.pack("<I", tipo)
        if tipo == _TIPO_STRING:
            v = str(valor).encode()
            kv += struct.pack("<Q", len(v)) + v
        else:
            kv += struct.pack("<I", int(valor))
        n += 1
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", n) + kv
    )
    return path


def _tienda_ollama(raiz: Path, tag: str, ctx: int | None, con_projector: bool = False) -> dict:
    """Monta un store de Ollama minimo y devuelve el artifact que lo apunta."""
    blobs = raiz / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    modelo = _gguf(raiz / "m.gguf", "qwen35moe", ctx)
    digest_modelo = "sha256-" + hashlib.sha256(modelo.read_bytes()).hexdigest()
    (blobs / digest_modelo).write_bytes(modelo.read_bytes())
    capas = [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": digest_modelo.replace("sha256-", "sha256:"),
        }
    ]
    if con_projector:
        (blobs / "sha256-proj").write_bytes(b"GGUF")
        capas.append(
            {"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:proj"}
        )
    man = (
        raiz
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / tag.split(":")[0]
        / tag.split(":")[1]
    )
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"schemaVersion": 2, "layers": capas}))
    artifact = {"kind": "gguf", "source": "ollama", "ref": tag}
    if con_projector:
        artifact["mmproj"] = {"source": "ollama", "layer": "projector", "optional": True}
    return artifact


def test_un_alta_sin_configuracion_sale_en_gpu(tmp_path):
    artifact = _tienda_ollama(tmp_path, "ornith-1.5:35b", 262144)
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert cfg["gpu_layers"] == 99
    assert cfg["ctx_size"] == 262144
    assert cfg["flash_attn"] is True
    assert cfg["cache_type_k"] == cfg["cache_type_v"] == "q8_0"
    assert cfg["kv_unified"] is True
    assert cfg["parallel_slots"] == 4
    assert cfg["cont_batching"] is True
    assert cfg["startup_timeout_s"] == 300


def test_nunca_sale_en_cpu_con_4096(tmp_path):
    """Invariante: es exactamente la combinacion que no carga."""
    casos = [
        _tienda_ollama(tmp_path / "a", "x:1", 262144),
        _tienda_ollama(tmp_path / "b", "x:1", None),
        _tienda_ollama(tmp_path / "c", "x:1", 8192),
        {"kind": "gguf", "source": "ollama", "ref": "no-instalado:1"},
    ]
    for i, artifact in enumerate(casos):
        raiz = [tmp_path / "a", tmp_path / "b", tmp_path / "c", tmp_path / "z"][i]
        cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=raiz)["llama_cpp"]
        assert cfg["gpu_layers"] > 0, artifact
        assert cfg["ctx_size"] >= _FALLBACK_CTX or cfg["ctx_size"] == 8192, cfg


def test_el_contexto_nativo_se_acota(tmp_path):
    """nemotron-3.5-lightning declara 1.048.576 tokens; no cabe en ninguna
    tarjeta de aqui, y la escalera de presets es donde se sube de ahi."""
    artifact = _tienda_ollama(tmp_path, "lightning:30b", 1048576)
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert cfg["ctx_size"] == _MAX_CTX


def test_un_contexto_pequeno_se_respeta(tmp_path):
    artifact = _tienda_ollama(tmp_path, "pequeno:1b", 8192)
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert cfg["ctx_size"] == 8192


def test_si_el_gguf_no_se_puede_leer_queda_algo_usable(tmp_path):
    artifact = {"kind": "gguf", "source": "ollama", "ref": "no-instalado:1"}
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert cfg["ctx_size"] == _FALLBACK_CTX
    assert cfg["gpu_layers"] == 99


def test_el_projector_se_carga_solo_cuando_llega_una_imagen(tmp_path):
    artifact = _tienda_ollama(tmp_path, "ornith-1.5:35b", 262144, con_projector=True)
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert cfg["vision_mode"] == "on_demand"


def test_sin_projector_no_se_inventa_vision(tmp_path):
    artifact = _tienda_ollama(tmp_path, "flint:40b", 262144)
    cfg = llama_cpp_launch_defaults(None, artifact=artifact, ollama_root=tmp_path)["llama_cpp"]
    assert "vision_mode" not in cfg


@pytest.mark.parametrize(
    ("clave", "valor"),
    [("gpu_layers", 0), ("ctx_size", 4096), ("flash_attn", False), ("vision_mode", "disabled")],
)
def test_lo_que_pide_quien_llama_manda(tmp_path, clave, valor):
    """Un operador que configura a mano no puede verse sobrescrito."""
    artifact = _tienda_ollama(tmp_path, "ornith-1.5:35b", 262144, con_projector=True)
    cfg = llama_cpp_launch_defaults(
        {"llama_cpp": {clave: valor}}, artifact=artifact, ollama_root=tmp_path
    )["llama_cpp"]
    assert cfg[clave] == valor


def test_el_alta_local_lee_el_contexto_del_model_path(tmp_path):
    """``/registry/local/register`` no trae artifact, solo la ruta del fichero."""
    ruta = _gguf(tmp_path / "suelto.gguf", "flint", 131072)
    cfg = llama_cpp_launch_defaults({"llama_cpp": {"model_path": str(ruta)}})["llama_cpp"]
    assert cfg["ctx_size"] == 131072
    assert cfg["gpu_layers"] == 99


def test_no_se_pierde_nada_de_otras_secciones(tmp_path):
    artifact = _tienda_ollama(tmp_path, "ornith-1.5:35b", 262144)
    cfg = llama_cpp_launch_defaults(
        {"registry_metadata": {"a": 1}, "llama_cpp": {"model_path": "/x.gguf"}},
        artifact=artifact,
        ollama_root=tmp_path,
    )
    assert cfg["registry_metadata"] == {"a": 1}
    assert cfg["llama_cpp"]["model_path"] == "/x.gguf"


# --- El backend ternario tiene el mismo agujero, peor ---------------------
# Sus defaults son BITNET_CTX_SIZE=4096 con BITNET_PARALLEL=16, y la ventana
# se reparte entre slots: 256 tokens por peticion. Es con lo que vivio
# bonsai-27b-ternario hasta 0.51.3.


def test_un_alta_ternaria_no_queda_en_256_tokens(tmp_path):
    from ocabra.core.registration_defaults import launch_defaults

    artifact = _tienda_ollama(tmp_path, "bonsai-2:27b", 131072)
    cfg = launch_defaults("bitnet", None, artifact=artifact, ollama_root=tmp_path)["bitnet"]
    assert cfg["ctx_size"] >= 32768
    assert cfg["parallel"] == 4
    assert cfg["ctx_size"] // cfg["parallel"] > 4096
    assert cfg["kv_unified"] is True
    assert cfg["cache_type_k"] == "q8_0"


def test_al_ternario_no_se_le_fuerzan_capas_de_gpu(tmp_path):
    """El backend ya elige 99 para PrismML, y la build llana puede ser de CPU."""
    from ocabra.core.registration_defaults import launch_defaults

    artifact = _tienda_ollama(tmp_path, "bonsai-2:27b", 131072)
    cfg = launch_defaults("bitnet", None, artifact=artifact, ollama_root=tmp_path)["bitnet"]
    assert "gpu_layers" not in cfg


def test_un_backend_ajeno_pasa_de_largo():
    from ocabra.core.registration_defaults import launch_defaults

    assert launch_defaults("vllm", {"vllm": {"runner": "generate"}}) == {
        "vllm": {"runner": "generate"}
    }
    assert launch_defaults("whisper", None) is None
