"""Dimensiones de una imagen leyendo solo su cabecera (PNG, JPEG, WebP, GIF).

El proceso api no tiene Pillow y no lo necesita para esto: basta con saber el
ancho y el alto para elegir el tamaño de un vídeo que respete la proporción de
la imagen de partida. Devuelve ``None`` si el formato no se reconoce; quien
llama decide si eso es un error.
"""

from __future__ import annotations

import struct

_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def image_size(data: bytes) -> tuple[int, int] | None:
    """``(ancho, alto)`` en píxeles, o ``None`` si no es una imagen conocida."""
    if len(data) < 26:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return _valid(w, h)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return _valid(w, h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp(data)
    if data[:2] == b"\xff\xd8":
        return _jpeg(data)
    return None


def _valid(w: int, h: int) -> tuple[int, int] | None:
    return (w, h) if w > 0 and h > 0 else None


def _webp(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return _valid(w, h)
    if chunk == b"VP8 " and len(data) >= 30:
        w, h = struct.unpack("<HH", data[26:30])
        return _valid(w & 0x3FFF, h & 0x3FFF)
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return _valid((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


def _jpeg(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # relleno
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        if marker in _JPEG_SOF:
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return _valid(w, h)
        i += 2 + length
    return None
