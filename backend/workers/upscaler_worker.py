#!/usr/bin/env python3
"""Worker de escalado de vídeo y fotos (Real-ESRGAN Compact / SRVGGNetCompact).

Recibe un segmento de vídeo sin audio (``/upscale``) o una foto
(``/upscale/image``) y devuelve el resultado escalado. No guarda estado entre
peticiones, así que el planificador puede expulsar el
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


def _is_legacy_esrgan(state_dict: dict) -> bool:
    return any(k.startswith("model.1.sub.") and ".RDB" in k for k in state_dict)


def translate_legacy_esrgan(state_dict: dict) -> dict:
    """Traduce el nombrado ESRGAN antiguo al de RRDBNet.

    Casi todo el catálogo de la comunidad (UltraSharp, Remacri, Siax,
    NMKD-Superscale...) se publicó con el nombrado original de ESRGAN
    (``model.1.sub.0.RDB1.conv1.0.weight``) en vez del de Real-ESRGAN
    (``body.0.rdb1.conv1.weight``). Son la misma red: sin esta traducción esos
    pesos no cargan y el catálogo entero queda inaccesible.
    """
    mapping = {
        "model.0": "conv_first",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    # El último sub-bloque no es un RRDB, es la convolución del cuerpo.
    sub_indices = [
        int(part)
        for k in state_dict
        if k.startswith("model.1.sub.")
        for part in [k.split(".")[3]]
        if part.isdigit()
    ]
    body_conv_index = max(sub_indices) if sub_indices else None

    out: dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.1.sub."):
            pieces = key.split(".")
            index = pieces[3]
            if index.isdigit() and int(index) == body_conv_index and ".RDB" not in key:
                out[f"conv_body.{pieces[-1]}"] = value
                continue
            # model.1.sub.{i}.RDB{n}.conv{m}.0.{weight|bias}
            rdb = pieces[4].replace("RDB", "rdb").lower()
            conv = pieces[5]
            out[f"body.{index}.{rdb}.{conv}.{pieces[-1]}"] = value
            continue
        prefix = ".".join(key.split(".")[:2])
        if prefix in mapping:
            out[f"{mapping[prefix]}.{key.split('.')[-1]}"] = value
        else:
            out[key] = value
    return out


def build_model(state_dict: dict) -> tuple[nn.Module, dict]:
    """Elige la arquitectura por las claves y devuelve (modelo, state_dict listo).

    Detectar en vez de configurar evita que cambiar de pesos obligue a tocar la
    configuración del modelo en oCabra: los tres formatos son inconfundibles.
    """
    if _is_legacy_esrgan(state_dict):
        state_dict = translate_legacy_esrgan(state_dict)
    if any(k.startswith("body.") and ".rdb" in k for k in state_dict):
        num_block = 1 + max(
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("body.") and k.split(".")[1].isdigit()
        )
        return RRDBNet(num_block=num_block), state_dict
    return SRVGGNetCompact(), state_dict


class WorkerState:
    def __init__(self, model_id: str, model_path: Path) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.model: Any = None
        self.scale: int = NATIVE_SCALE
        self.half: bool = True
        self.load_error: str | None = None


def _load_with_spandrel(weights: Path):
    """Carga con spandrel, que reconoce 42 arquitecturas y detecta la correcta.

    Se prefiere a la implementación propia de más abajo por una razón de
    ingeniería: mantener a mano dos arquitecturas era razonable, pero DAT, SPAN,
    PLKSR, ATD o DRCT son transformers de cientos de líneas y reimplementarlos
    significa arriesgarse a errores sutiles que degradan la salida sin avisar.
    spandrel está mantenida, tiene licencia permisiva y es la que usa chaiNNer.
    """
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_file(str(weights))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(
            f"El fichero no es un modelo de imagen utilizable: {type(descriptor).__name__}"
        )
    return descriptor


def load_model(state: WorkerState) -> None:
    weights = state.model_path
    if weights.is_dir():
        candidates = sorted(weights.glob("*.pth")) + sorted(weights.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No weights found under {weights}")
        weights = candidates[0]

    try:
        descriptor = _load_with_spandrel(weights)
    except ImportError:
        # Respaldo para instalaciones anteriores del backend, sin spandrel:
        # cubre Compact y RRDBNet, que es lo que había antes.
        _log("upscaler_spandrel_ausente", level=logging.WARNING, weights=str(weights))
        raw = torch.load(weights, map_location="cpu", weights_only=True)
        sd = raw.get("params") or raw.get("params_ema") or raw
        model, sd = build_model(sd)
        model = model.eval().half().cuda()
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys:
            _log(
                "upscaler_state_dict_partial",
                level=logging.WARNING,
                missing=len(result.missing_keys),
                unexpected=len(result.unexpected_keys),
            )
        state.model = model
        state.scale = NATIVE_SCALE
        _log("upscaler_model_loaded", model_id=state.model_id,
             arch=type(model).__name__, scale=state.scale, weights=str(weights))
        return

    # half() no vale para todas: algunas arquitecturas solo soportan fp32 y
    # spandrel lo declara, así que se respeta en vez de forzar y romper.
    if descriptor.supports_half:
        descriptor.model.half()
    descriptor.model.eval().cuda()
    state.model = descriptor.model
    state.half = descriptor.supports_half
    # La escala la declara el modelo: dejar de asumir 4x permite usar pesos 2x.
    state.scale = int(descriptor.scale)
    _log(
        "upscaler_model_loaded",
        model_id=state.model_id,
        arch=descriptor.architecture.name,
        scale=state.scale,
        half=state.half,
        weights=str(weights),
    )


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
    scale = state.scale
    out_w, out_h = width * scale, height * scale

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
                x = torch.from_numpy(arr.copy()).cuda().permute(0, 3, 1, 2)
                x = (x.half() if state.half else x.float()).div_(255)
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


# ── Fotos: la misma red, sin códec de vídeo ─────────────────────────────

# Lado de la tesela (en píxeles de entrada). La ruta de vídeo no tesela porque
# sus fotogramas son pequeños (≤1080p); una foto de móvil de 12 MP pasada entera
# por RRDBNet pide varios GiB de activaciones y tumba la tarjeta compartida.
IMAGE_TILE = int(os.getenv("UPSCALER_IMAGE_TILE", "512"))
# Margen de contexto alrededor de cada tesela. Se recorta de la salida, así que
# no hay mezcla: sin margen se ven costuras en los bordes de cada trozo porque
# la red no "ve" lo que hay al otro lado.
IMAGE_TILE_PAD = int(os.getenv("UPSCALER_IMAGE_TILE_PAD", "32"))
# Píxeles de entrada máximos. La salida se ensambla en RAM a la escala nativa
# del modelo (16x los píxeles con un 4x): 4K de entrada ya son ~400 MB.
IMAGE_MAX_PIXELS = int(os.getenv("UPSCALER_IMAGE_MAX_PIXELS", str(3840 * 2160)))
# Límites de dimensión de los propios formatos.
_FORMAT_MAX_SIDE = {"png": 2**31 - 1, "jpeg": 65_535, "webp": 16_383}
_FORMAT_MEDIA_TYPE = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
_FORMAT_ALIASES = {"jpg": "jpeg"}


class ImageRequestError(ValueError):
    """Petición de foto inválida: se responde con ``status_code`` y el mensaje."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def normalize_output_format(value: str | None) -> str:
    fmt = (value or "png").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in _FORMAT_MEDIA_TYPE:
        raise ImageRequestError(f"output_format '{value}' no soportado; usa png, jpeg o webp.")
    return fmt


def plan_delivered_size(
    width: int,
    height: int,
    model_scale: int,
    *,
    scale: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> tuple[int, int]:
    """Tamaño que se entrega, con el mismo criterio que la ruta de vídeo.

    El modelo siempre produce ``model_scale`` x; lo que pide el cliente solo
    puede *reducir* esa salida (``scale`` menor que la nativa o una caja
    ``target_width``/``target_height``), nunca ampliarla: estirar por encima de
    lo que dio el modelo sería un escalado falso que finge una resolución que
    no existe. Se conserva la proporción y gana la restricción más pequeña.
    """
    out_w, out_h = width * model_scale, height * model_scale
    factor = 1.0
    if scale is not None:
        factor = min(factor, scale / model_scale)
    if target_width:
        factor = min(factor, target_width / out_w)
    if target_height:
        factor = min(factor, target_height / out_h)
    if factor >= 1.0:
        return out_w, out_h
    return max(1, round(out_w * factor)), max(1, round(out_h * factor))


def _model_device_dtype(state: WorkerState) -> tuple[Any, Any]:
    try:
        device = next(state.model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cuda")
    return device, (torch.float16 if state.half else torch.float32)


def upscale_rgb_tiled(state: WorkerState, rgb: Any, *, tile: int, pad: int) -> tuple[Any, int]:
    """Escala un array HxWx3 uint8 por teselas; devuelve (array escalado, teselas).

    Cada tesela se procesa con ``pad`` píxeles de contexto por cada lado y de su
    salida solo se copia la parte central, así que el resultado es el de
    procesar la foto entera salvo en los bordes del campo receptivo. Imágenes
    que caben en una tesela van de una pasada.
    """
    height, width = rgb.shape[:2]
    scale = state.scale
    device, dtype = _model_device_dtype(state)
    out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    src = torch.from_numpy(np.require(rgb, requirements=["C", "W"])).permute(2, 0, 1)[None]
    tile = max(16, tile)
    tiles = 0
    with torch.no_grad():
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(height, y1 + pad), min(width, x1 + pad)
                x = src[:, :, py0:py1, px0:px1].to(device=device, dtype=dtype).div_(255)
                y = state.model(x).clamp_(0, 1)
                oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
                core = y[:, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale]
                out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = (
                    core.mul(255).round_().byte()[0].permute(1, 2, 0).cpu().numpy()
                )
                tiles += 1
    return out, tiles


def _is_oom(exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def upscale_image_sync(
    state: WorkerState,
    payload: bytes,
    *,
    scale: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    output_format: str | None = None,
    quality: int | None = None,
    tile: int | None = None,
) -> tuple[bytes, str, dict]:
    """Escala una foto: decodifica con PIL, pasa la red por teselas y codifica.

    - Orientación EXIF aplicada (las fotos de móvil vienen giradas por EXIF).
    - Transparencia: el RGB pasa por la red y el canal alfa se reescala aparte
      con Lanczos, porque estas redes son de 3 canales. JPEG no tiene alfa: se
      descarta.
    - Si la VRAM no alcanza se reintenta con teselas de la mitad de lado hasta
      64 px antes de rendirse con un 507.
    """
    from io import BytesIO

    from PIL import Image, ImageOps, UnidentifiedImageError

    fmt = normalize_output_format(output_format)
    try:
        img = Image.open(BytesIO(payload))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageRequestError(f"No se reconoce la imagen: {exc}") from exc
    width, height = img.size
    # Antes de decodificar: un PNG pequeño puede declarar 50.000 x 50.000 px.
    if width * height > IMAGE_MAX_PIXELS:
        raise ImageRequestError(
            f"La imagen tiene {width}x{height} px; el máximo son {IMAGE_MAX_PIXELS} px "
            "de entrada (UPSCALER_IMAGE_MAX_PIXELS).",
            status_code=413,
        )
    try:
        img = ImageOps.exif_transpose(img)
        img.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageRequestError(f"No se pudo decodificar la imagen: {exc}") from exc
    width, height = img.size

    has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        img.mode == "P" and "transparency" in img.info
    )
    alpha = None
    if has_alpha:
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb_img = rgba.convert("RGB")
    else:
        rgb_img = img.convert("RGB")

    out_w, out_h = plan_delivered_size(
        width,
        height,
        state.scale,
        scale=scale,
        target_width=target_width,
        target_height=target_height,
    )
    if max(out_w, out_h) > _FORMAT_MAX_SIDE[fmt]:
        raise ImageRequestError(
            f"{fmt} admite como mucho {_FORMAT_MAX_SIDE[fmt]} px por lado y la salida "
            f"sería {out_w}x{out_h}; pide png o un target_width/target_height menor."
        )

    started = time.monotonic()
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    tile_side = tile or IMAGE_TILE
    while True:
        try:
            up, tiles = upscale_rgb_tiled(state, rgb, tile=tile_side, pad=IMAGE_TILE_PAD)
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if tile_side <= 64:
                raise ImageRequestError(
                    "Sin VRAM para escalar la imagen ni con teselas de 64 px.",
                    status_code=507,
                ) from exc
            tile_side //= 2
            _log("upscaler_image_oom_retry", level=logging.WARNING, tile=tile_side)
    model_h, model_w = up.shape[:2]

    result = Image.fromarray(up, "RGB")
    del up
    keep_alpha = alpha is not None and fmt != "jpeg"
    if keep_alpha:
        result.putalpha(alpha.resize((model_w, model_h), Image.Resampling.LANCZOS))
    if (out_w, out_h) != (model_w, model_h):
        result = result.resize((out_w, out_h), Image.Resampling.LANCZOS)

    buf = BytesIO()
    q = 95 if quality is None else max(1, min(100, int(quality)))
    if fmt == "png":
        # Nivel 4 y no el 6 por defecto: en salidas de 100+ MP el 9 tarda
        # segundos y apenas gana; el cliente suele recomprimir de todas formas.
        result.save(buf, format="PNG", compress_level=4)
    elif fmt == "jpeg":
        result.save(buf, format="JPEG", quality=q, subsampling=0 if q >= 90 else 2)
    else:
        result.save(buf, format="WEBP", quality=q, method=4)
    elapsed = time.monotonic() - started
    stats = {
        "input_resolution": f"{width}x{height}",
        "model_resolution": f"{model_w}x{model_h}",
        "delivered_resolution": f"{out_w}x{out_h}",
        "delivered_width": out_w,
        "delivered_height": out_h,
        "scale": state.scale,
        "tiles": tiles,
        "alpha": int(keep_alpha),
        "output_format": fmt,
        "seconds": round(elapsed, 2),
    }
    return buf.getvalue(), _FORMAT_MEDIA_TYPE[fmt], stats


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

    @app.post("/upscale/image")
    async def upscale_image(
        file: UploadFile,
        scale: float | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        output_format: str = "png",
        quality: int | None = None,
        tile: int | None = None,
    ) -> Response:
        """Escala una foto sin pasar por ningún códec de vídeo.

        Devuelve la imagen en ``output_format`` (png por defecto) con cabeceras
        ``x-ocabra-*`` de estadísticas, como ``/upscale``. 400 si la imagen o
        los parámetros no valen, 413 si pasa de ``UPSCALER_IMAGE_MAX_PIXELS``,
        507 si no hay VRAM ni con teselas mínimas.
        """
        if state.model is None:
            raise HTTPException(status_code=503, detail=state.load_error or "Model not ready")
        for name, value in (
            ("scale", scale),
            ("target_width", target_width),
            ("target_height", target_height),
            ("tile", tile),
        ):
            if value is not None and value <= 0:
                raise HTTPException(status_code=400, detail=f"{name} debe ser mayor que 0")
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Imagen vacía")
        loop = asyncio.get_running_loop()
        run = partial(
            upscale_image_sync,
            state,
            payload,
            scale=scale,
            target_width=target_width,
            target_height=target_height,
            output_format=output_format,
            quality=quality,
            tile=tile,
        )
        try:
            body, media_type, stats = await loop.run_in_executor(None, run)
        except ImageRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ImportError as exc:  # pragma: no cover - instalación sin Pillow
            raise HTTPException(
                status_code=501,
                detail=f"El backend upscaler no tiene Pillow instalado: {exc}",
            ) from exc

        _log("upscaler_image_done", model_id=state.model_id, **stats)
        return Response(
            content=body,
            media_type=media_type,
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
        return {"model_id": state.model_id, "scale": state.scale, "vram_used_mb": vram}

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
