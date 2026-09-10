#!/usr/bin/env python3
"""Worker de escalado de vídeo (Real-ESRGAN Compact / SRVGGNetCompact).

Recibe un segmento de vídeo sin audio y devuelve el segmento escalado. No
guarda estado entre peticiones, así que el planificador puede expulsar el
modelo entre segmentos sin romper nada.

Nota de rendimiento: el modelo va sobrado (~90 fps a 1080p en una RTX 3090) y
el cuello está en mover fotogramas. Por eso se trabaja por lotes, se convierte
a YUV en la GPU y se escribe a ffmpeg con buffer grande: pasar fotogramas de
uno en uno por Python cuesta un 4x.
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
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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


DEFAULT_BATCH = int(os.getenv("UPSCALER_BATCH_SIZE", "8"))
FFMPEG = os.getenv("FFMPEG_BINARY", "ffmpeg")
# Escala nativa de los pesos Compact publicados por Real-ESRGAN.
NATIVE_SCALE = 4
# El segmento que sale de aquí es un intermedio: el cliente lo recodifica al
# unir. Así que interesa un codificador rápido y de calidad alta, y ese es
# NVENC: medido, libx264 a 1080p tapona el proceso en ~15 fps mientras el
# modelo va a ~90, y NVENC pasa de 107. Se usa un cq bajo para no acumular
# pérdida de generación en el intermedio.
INTERMEDIATE_ENCODER = os.getenv("UPSCALER_INTERMEDIATE_ENCODER", "auto").strip().lower()
# Enviar los fotogramas al codificador ya en YUV 4:2:0 en vez de RGB24 recorta
# a la mitad los bytes que cruzan el pipe (1,5 frente a 3 bytes por píxel), que
# es el verdadero cuello: el modelo va a ~90 fps y la tubería taponaba en ~19.
# La conversión se hace en la GPU, donde es gratis.
YUV_FAST_PATH = os.getenv("UPSCALER_YUV_FAST_PATH", "1").strip().lower() in {
    "1", "true", "yes", "on",
}


class SRVGGNetCompact(nn.Module):
    """Arquitectura Compact de Real-ESRGAN.

    Se implementa aquí en vez de depender de ``basicsr``/``realesrgan``, que
    están sin mantener y rompen con torchvision moderno (``functional_tensor``).
    Son 30 líneas y el ``state_dict`` oficial encaja al 100%.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = NATIVE_SCALE,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat),
        ]
        for _ in range(num_conv):
            body += [
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.PReLU(num_parameters=num_feat),
            ]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsampler(self.body(x))
        # Rama residual sobre el original interpolado, como el modelo oficial.
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    """Bloque denso residual de RRDBNet (Real-ESRGAN x4plus)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4plus: ~16,7M parámetros frente a los ~1,2M de Compact.

    Es el nivel intermedio: bastante mejor detalle que Compact y mucho más
    rápido que los modelos de difusión. Se implementa aquí por lo mismo que
    Compact: ``basicsr`` está sin mantener y rompe con torchvision moderno.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def build_model(state_dict: dict) -> nn.Module:
    """Elige la arquitectura mirando las claves del ``state_dict``.

    Detectar en vez de configurar evita que un cambio de pesos exija tocar la
    configuración del modelo en oCabra: los dos formatos son inconfundibles.
    """
    if any(k.startswith("body.") and ".rdb" in k for k in state_dict):
        num_block = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("body.")
        )
        return RRDBNet(num_block=num_block)
    return SRVGGNetCompact()


class WorkerState:
    def __init__(self, model_id: str, model_path: Path) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.model: Any = None
        self.load_error: str | None = None


def load_model(state: WorkerState) -> None:
    weights = state.model_path
    if weights.is_dir():
        candidates = sorted(weights.glob("*.pth")) + sorted(weights.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No weights found under {weights}")
        weights = candidates[0]

    raw = torch.load(weights, map_location="cpu", weights_only=True)
    sd = raw.get("params") or raw.get("params_ema") or raw
    model = build_model(sd).eval().half().cuda()
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        # No se aborta: pesos con nombres ligeramente distintos siguen siendo
        # utilizables, pero conviene que quede en el log.
        _log(
            "upscaler_state_dict_partial",
            level=logging.WARNING,
            missing=len(result.missing_keys),
            unexpected=len(result.unexpected_keys),
        )
    state.model = model
    _log("upscaler_model_loaded", model_id=state.model_id,
         arch=type(model).__name__, weights=str(weights))


def _rgb_to_yuv420p(y: Any) -> bytes:
    """Convierte un lote RGB [0,1] (N,3,H,W) a bytes YUV420p planares.

    Matriz BT.709 con rango limitado (16-235 / 16-240), que es lo que asume
    ffmpeg para HD cuando se le declara ``-colorspace bt709 -color_range tv``.
    Cualquier desviación aquí sale como un cambio de color en el resultado, así
    que los coeficientes no se tocan sin volver a medir PSNR contra la ruta RGB.
    """
    r, g, b = y[:, 0], y[:, 1], y[:, 2]
    luma = 16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)
    cb = 128.0 + 224.0 * (-0.1146 * r - 0.3854 * g + 0.5000 * b)
    cr = 128.0 + 224.0 * (0.5000 * r - 0.4542 * g - 0.0458 * b)
    # 4:2:0: la crominancia se promedia en bloques de 2x2.
    cb = F.avg_pool2d(cb.unsqueeze(1), 2).squeeze(1)
    cr = F.avg_pool2d(cr.unsqueeze(1), 2).squeeze(1)

    luma_b = luma.clamp_(0, 255).round_().byte().cpu().numpy()
    cb_b = cb.clamp_(0, 255).round_().byte().cpu().numpy()
    cr_b = cr.clamp_(0, 255).round_().byte().cpu().numpy()
    chunks = []
    for i in range(luma_b.shape[0]):
        chunks.append(luma_b[i].tobytes())
        chunks.append(cb_b[i].tobytes())
        chunks.append(cr_b[i].tobytes())
    return b"".join(chunks)


def _encoder_args(crf: int) -> list[str]:
    """Argumentos del codificador para el segmento intermedio."""
    use_nvenc = INTERMEDIATE_ENCODER == "nvenc" or (
        INTERMEDIATE_ENCODER == "auto" and _has_nvenc()
    )
    if use_nvenc:
        # El desplazamiento es pequeño a propósito. Este segmento es un
        # intermedio que el cliente vuelve a codificar, así que la prioridad es
        # no acumular pérdida de generación: con cq = crf + 6 el SSIM final
        # medido bajaba de 0,938 a 0,903, y con +2 se recupera. El coste en
        # velocidad es despreciable porque NVENC va sobrado.
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", str(min(51, crf + 2))]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]


@lru_cache(maxsize=1)
def _has_nvenc() -> bool:
    """¿Se puede abrir *de verdad* una sesión NVENC?

    No basta con que ffmpeg liste el codificador: las GPU de consumo limitan
    las sesiones NVENC simultáneas y, si están agotadas o el dispositivo no
    sirve, el encoder muere al arrancar y el worker se lleva un broken pipe a
    mitad del segmento. Por eso se prueba una codificación real de un
    fotograma; el resultado se cachea porque esto no cambia en caliente.
    """
    try:
        probe = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
             "-i", "color=black:s=256x256:d=0.1", "-c:v", "h264_nvenc",
             "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        ok = probe.returncode == 0
        if not ok:
            _log(
                "upscaler_nvenc_unavailable",
                level=logging.WARNING,
                detail=probe.stderr.decode("utf-8", errors="ignore").strip()[:200],
            )
        return ok
    except (OSError, subprocess.SubprocessError):
        return False


def probe_video(path: Path) -> tuple[int, int, str]:
    """Devuelve (ancho, alto, fps) del segmento de entrada."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height, fps = out.split(",")[:3]
    return int(width), int(height), fps


def upscale_sync(
    state: WorkerState,
    src: Path,
    dst: Path,
    *,
    target_height: int | None,
    batch_size: int,
    crf: int,
) -> dict:
    width, height, fps = probe_video(src)
    out_w, out_h = width * NATIVE_SCALE, height * NATIVE_SCALE

    decoder = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=1 << 24,
    )
    # El reescalado al objetivo final lo hace ffmpeg: la escala del modelo es
    # fija (4x) y la resolución que pide el cliente es independiente de ella.
    # Solo se reduce, nunca se amplía: si la salida del modelo se queda por
    # debajo del objetivo (FlashVSR recorta a múltiplos de 128, y 4x de una
    # fuente pequeña puede no llegar), estirarla aquí sería un escalado falso
    # que añade desenfoque y finge una resolución que no existe. Se entrega lo
    # que el modelo produjo y el cliente decide.
    filters = []
    if target_height and target_height < out_h:
        filters.append(f"scale=-2:{target_height}:flags=lanczos")
    in_pix_fmt = "yuv420p" if YUV_FAST_PATH else "rgb24"
    encoder_cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", in_pix_fmt,
        "-s", f"{out_w}x{out_h}", "-r", fps,
    ]
    if YUV_FAST_PATH:
        encoder_cmd += ["-colorspace", "bt709", "-color_range", "tv"]
    encoder_cmd += ["-i", "-"]
    if filters:
        encoder_cmd += ["-vf", ",".join(filters)]
    encoder_cmd += [*_encoder_args(crf), "-pix_fmt", "yuv420p", str(dst)]
    encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, bufsize=1 << 24)

    frame_bytes = width * height * 3
    frames = 0
    started = time.monotonic()
    try:
        with torch.no_grad():
            while True:
                chunk = decoder.stdout.read(frame_bytes * batch_size)
                if not chunk:
                    break
                n = len(chunk) // frame_bytes
                if n == 0:
                    break
                arr = np.frombuffer(chunk[: n * frame_bytes], np.uint8).reshape(
                    n, height, width, 3
                )
                x = torch.from_numpy(arr.copy()).cuda().permute(0, 3, 1, 2).half().div_(255)
                y = state.model(x).clamp_(0, 1)
                if YUV_FAST_PATH:
                    encoder.stdin.write(_rgb_to_yuv420p(y))
                else:
                    encoder.stdin.write(
                        y.mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy().tobytes()
                    )
                frames += n
    except BrokenPipeError as exc:
        # El codificador murió a mitad. Sin este mensaje el fallo llega al
        # cliente como un 500 opaco y cuesta media hora averiguar que fue NVENC.
        raise RuntimeError(
            "El codificador de vídeo se cerró inesperadamente. Suele ser NVENC "
            "sin sesiones libres: prueba con UPSCALER_INTERMEDIATE_ENCODER=libx264."
        ) from exc
    finally:
        if decoder.stdout:
            decoder.stdout.close()
        decoder.wait()
        if encoder.stdin:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        encoder.wait()

    elapsed = time.monotonic() - started
    delivered_h = target_height if (target_height and target_height < out_h) else out_h
    return {
        "frames": frames,
        "input_resolution": f"{width}x{height}",
        "model_resolution": f"{out_w}x{out_h}",
        "delivered_height": delivered_h,
        "seconds": round(elapsed, 2),
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
    }


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title="oCabra Upscaler Worker")

    @app.post("/upscale")
    async def upscale(
        file: UploadFile,
        target_height: int | None = None,
        batch_size: int = DEFAULT_BATCH,
        crf: int = 14,
    ) -> Response:
        if state.model is None:
            raise HTTPException(status_code=503, detail=state.load_error or "Model not ready")

        payload = await file.read()
        with tempfile.TemporaryDirectory(prefix="ocabra_upscale_") as tmp:
            src = Path(tmp) / "in.mp4"
            dst = Path(tmp) / "out.mp4"
            src.write_bytes(payload)
            loop = asyncio.get_running_loop()
            run = partial(
                upscale_sync, state, src, dst,
                target_height=target_height, batch_size=batch_size, crf=crf,
            )
            try:
                stats = await loop.run_in_executor(None, run)
            except subprocess.CalledProcessError as exc:
                raise HTTPException(status_code=400, detail=f"ffprobe failed: {exc}") from exc
            if not dst.exists() or dst.stat().st_size == 0:
                raise HTTPException(status_code=500, detail="Encoder produced no output")
            body = dst.read_bytes()

        _log("upscaler_segment_done", model_id=state.model_id, **stats)
        return Response(
            content=body,
            media_type="video/mp4",
            headers={f"x-ocabra-{k.replace('_', '-')}": str(v) for k, v in stats.items()},
        )

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": state.model is not None}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        vram = 0
        if torch is not None and torch.cuda.is_available():
            vram = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return {"model_id": state.model_id, "scale": NATIVE_SCALE, "vram_used_mb": vram}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra video upscaler worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def main() -> None:
    if torch is None or np is None:
        raise RuntimeError("torch and numpy are required to run upscaler_worker")

    # Sin configurar logging, Python descarta los INFO de este módulo y el
    # worker se queda mudo: se pierden fps, VRAM y avisos como el de NVENC.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    state = WorkerState(model_id=args.model_id, model_path=Path(args.model_path))
    try:
        load_model(state)
    except Exception as exc:
        state.load_error = str(exc)
        raise

    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
