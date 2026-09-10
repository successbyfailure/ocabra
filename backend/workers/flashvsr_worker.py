#!/usr/bin/env python3
"""Worker de escalado de vídeo con FlashVSR v1.1 (difusión de un paso).

Es el nivel de máxima calidad. Medido en una RTX 3090 con salida 1080p:
3,23 fps y 19,1 GB de VRAM. Dos hechos que condicionan su uso, ambos medidos:

  * La VRAM es **constante** con la duración del segmento (19,1 GB con 2 s y
    19,2 GB con 10 s): el modelo es streaming de verdad. La longitud del
    segmento la decide la política, no la memoria.
  * A 3,23 fps son ~9x el tiempo real. Diez minutos de vídeo son hora y media,
    así que esto va detrás de una cola, nunca en una petición interactiva.

La escala es fija 4x y el objetivo se redondea a múltiplos de 128 con recorte
centrado; el ajuste a la resolución que pide el cliente se hace después con
ffmpeg.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - el instalador provee las deps
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

# Los demás workers no dependen de structlog (no está en el runtime base que
# siembra el instalador), así que aquí se usa logging estándar con un pequeño
# ayudante para conservar el estilo clave=valor de los registros de oCabra.
logger = logging.getLogger(__name__)


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, "%s %s", event, detail)


FFMPEG = os.getenv("FFMPEG_BINARY", "ffmpeg")
NATIVE_SCALE = 4.0
# "sparse_ratio": 1.5 va más rápido, 2.0 es más estable. Recomendación upstream.
SPARSE_RATIO = float(os.getenv("FLASHVSR_SPARSE_RATIO", "2.0"))
# local_range: 9 da más nitidez, 11 más estabilidad temporal.
LOCAL_RANGE = int(os.getenv("FLASHVSR_LOCAL_RANGE", "11"))


class WorkerState:
    def __init__(self, model_id: str, model_path: Path, src_dir: Path) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.src_dir = src_dir
        self.pipe: Any = None
        self.helpers: Any = None
        self.load_error: str | None = None


def _load_reference_helpers(src_dir: Path) -> Any:
    """Reutiliza los helpers del script de referencia sin ejecutar su main().

    Se hace así a propósito: ``prepare_input_tensor`` implementa el redondeo a
    múltiplos de 128, el recorte centrado y la restricción de 8n+1 fotogramas.
    Reimplementarlo sería duplicar reglas que cambian con cada versión suya.
    """
    import types

    script = src_dir / "examples" / "WanVSR" / "infer_flashvsr_v1.1_tiny_long_video.py"
    if not script.exists():
        raise FileNotFoundError(f"FlashVSR reference script not found: {script}")
    source = script.read_text(encoding="utf-8").replace(
        'if __name__ == "__main__":\n    main()', ""
    )
    module = types.ModuleType("flashvsr_ref")
    module.__dict__["__file__"] = str(script)
    exec(compile(source, str(script), "exec"), module.__dict__)
    return module


def load_pipeline(state: WorkerState) -> None:
    wan_dir = state.src_dir / "examples" / "WanVSR"
    sys.path.insert(0, str(wan_dir))
    os.chdir(wan_dir)

    from diffsynth import FlashVSRTinyLongPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    state.helpers = _load_reference_helpers(state.src_dir)
    weights = state.model_path

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=torch.bfloat16)
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(weights / "LQ_proj_in.ckpt", map_location="cpu"), strict=True
    )
    pipe.denoising_model().LQ_proj_in.to("cuda")

    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
    )
    pipe.TCDecoder.load_state_dict(
        torch.load(weights / "TCDecoder.ckpt"), strict=False
    )

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])

    state.pipe = pipe
    _log("flashvsr_pipeline_loaded", model_id=state.model_id)


def upscale_sync(
    state: WorkerState, src: Path, dst: Path, *, target_height: int | None, crf: int
) -> dict:
    helpers = state.helpers
    started = time.monotonic()

    lq, th, tw, frames, fps = helpers.prepare_input_tensor(
        str(src), scale=NATIVE_SCALE, dtype=torch.bfloat16, device="cuda"
    )
    video = state.pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=0,
        LQ_video=lq,
        num_frames=frames,
        height=th,
        width=tw,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=SPARSE_RATIO * 768 * 1280 / (th * tw),
        kv_ratio=3.0,
        local_range=LOCAL_RANGE,
        color_fix=True,
    )
    pil_frames = helpers.tensor2video(video)

    encoder_cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-",
    ]
    # Solo se reduce, nunca se amplía: si la salida del modelo se queda por
    # debajo del objetivo (FlashVSR recorta a múltiplos de 128, y 4x de una
    # fuente pequeña puede no llegar), estirarla aquí sería un escalado falso
    # que añade desenfoque y finge una resolución que no existe. Se entrega lo
    # que el modelo produjo y el cliente decide.
    if target_height and target_height < th:
        encoder_cmd += ["-vf", f"scale=-2:{target_height}:flags=lanczos"]
    encoder_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", str(dst)]
    encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, bufsize=1 << 24)
    try:
        for frame in pil_frames:
            encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait()

    elapsed = time.monotonic() - started
    peak_mb = int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0
    return {
        "frames": int(frames),
        "model_resolution": f"{tw}x{th}",
        "delivered_height": target_height if (target_height and target_height < th) else th,
        "seconds": round(elapsed, 2),
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "vram_peak_mb": peak_mb,
    }


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title="oCabra FlashVSR Worker")

    @app.post("/upscale")
    async def upscale(
        file: UploadFile, target_height: int | None = None, crf: int = 14
    ) -> Response:
        if state.pipe is None:
            raise HTTPException(status_code=503, detail=state.load_error or "Pipeline not ready")

        payload = await file.read()
        with tempfile.TemporaryDirectory(prefix="ocabra_flashvsr_") as tmp:
            src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
            src.write_bytes(payload)
            loop = asyncio.get_running_loop()
            run = partial(upscale_sync, state, src, dst, target_height=target_height, crf=crf)
            try:
                stats = await loop.run_in_executor(None, run)
            except torch.cuda.OutOfMemoryError as exc:
                # Su VRAM la fija la resolución de salida, no la duración: si
                # no cabe, trocear más corto no arregla nada. Se dice claro.
                raise HTTPException(
                    status_code=507,
                    detail=(
                        "FlashVSR ran out of VRAM. Its memory depends on output "
                        "resolution, not segment length: lower the target "
                        "resolution instead of shortening the segment."
                    ),
                ) from exc
            if not dst.exists() or dst.stat().st_size == 0:
                raise HTTPException(status_code=500, detail="Encoder produced no output")
            body = dst.read_bytes()

        _log("flashvsr_segment_done", model_id=state.model_id, **stats)
        return Response(
            content=body,
            media_type="video/mp4",
            headers={f"x-ocabra-{k.replace('_', '-')}": str(v) for k, v in stats.items()},
        )

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": state.pipe is not None}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        vram = 0
        if torch is not None and torch.cuda.is_available():
            vram = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return {"model_id": state.model_id, "scale": NATIVE_SCALE, "vram_used_mb": vram}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra FlashVSR upscaler worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True, help="Directorio de pesos FlashVSR-v1.1")
    parser.add_argument("--src-dir", required=True, help="Clon del repo FlashVSR")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def main() -> None:
    if torch is None or np is None:
        raise RuntimeError("torch and numpy are required to run flashvsr_worker")

    # Sin configurar logging, Python descarta los INFO de este módulo y el
    # worker se queda mudo: se pierden fps, VRAM y avisos como el de NVENC.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    state = WorkerState(
        model_id=args.model_id,
        model_path=Path(args.model_path),
        src_dir=Path(args.src_dir),
    )
    try:
        load_pipeline(state)
    except Exception as exc:
        state.load_error = str(exc)
        raise

    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
