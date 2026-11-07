"""Páginas de documentación en prosa para quien consume la API.

``/docs`` sirve el Swagger de FastAPI, que documenta la *forma* de cada
endpoint pero no los cambios de comportamiento entre versiones ni cómo rinde
cada modelo. Estas páginas cubren eso, y están pensadas para dos públicos a la
vez: personas (HTML legible) y agentes (la misma fuente en Markdown crudo,
sufijo ``.md``).

El Markdown es la única fuente de verdad: vive en ``ocabra/docs_content/`` y se
renderiza aquí. El renderizador es deliberadamente mínimo — soporta justo lo que
usan estas páginas — para no añadir una dependencia de Markdown al backend.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()

CONTENT_DIR = Path(__file__).resolve().parent.parent / "docs_content"

#: Páginas publicadas: ruta -> (fichero, título).
PAGES: dict[str, tuple[str, str]] = {
    "changelog": ("changelog.md", "Changelog"),
    "models": ("models.md", "Modelos"),
}


# ── Renderizado de Markdown ──────────────────────────────────────────────────
# Subconjunto soportado a proposito: encabezados, parrafos, listas, reglas,
# bloques de codigo, tablas, negrita, codigo en linea y enlaces. Cualquier otra
# sintaxis se muestra literal en lugar de romper la pagina.

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _inline(texto: str) -> str:
    """Aplica el formato de línea sobre texto ya escapado."""
    escapado = html.escape(texto, quote=False)
    escapado = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", escapado)
    escapado = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", escapado)
    escapado = _LINK.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escapado,
    )
    return escapado


def _slug(texto: str) -> str:
    # Plegamos los acentos antes de limpiar: sin esto "Qué falla y por qué"
    # daba "qu-falla-y-por-qu" y los enlaces internos de la propia página no
    # casaban con el id del encabezado al que apuntaban.
    plano = unicodedata.normalize("NFKD", texto.lower())
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    limpio = re.sub(r"[^a-z0-9]+", "-", plano).strip("-")
    return limpio or "seccion"


def markdown_to_html(fuente: str) -> str:
    """Convierte el subconjunto de Markdown que usan estas páginas a HTML."""
    salida: list[str] = []
    lineas = fuente.split("\n")
    i = 0
    n = len(lineas)

    while i < n:
        linea = lineas[i]
        pelada = linea.strip()

        # bloque de codigo
        if pelada.startswith("```"):
            i += 1
            cuerpo: list[str] = []
            while i < n and not lineas[i].strip().startswith("```"):
                cuerpo.append(lineas[i])
                i += 1
            i += 1  # cierre
            salida.append(f"<pre><code>{html.escape(chr(10).join(cuerpo))}</code></pre>")
            continue

        # linea vacia
        if not pelada:
            i += 1
            continue

        # regla horizontal
        if re.fullmatch(r"-{3,}", pelada):
            salida.append("<hr>")
            i += 1
            continue

        # encabezado
        enc = re.match(r"(#{1,4})\s+(.*)", pelada)
        if enc:
            nivel = len(enc.group(1))
            texto = enc.group(2).strip()
            salida.append(f'<h{nivel} id="{_slug(texto)}">{_inline(texto)}</h{nivel}>')
            i += 1
            continue

        # tabla: cabecera + separador + filas
        if (
            pelada.startswith("|")
            and i + 1 < n
            and re.fullmatch(r"[|\-: ]+", lineas[i + 1].strip())
        ):

            def celdas(fila: str) -> list[str]:
                return [c.strip() for c in fila.strip().strip("|").split("|")]

            cabecera = celdas(linea)
            i += 2
            filas: list[list[str]] = []
            while i < n and lineas[i].strip().startswith("|"):
                filas.append(celdas(lineas[i]))
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in cabecera)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in f) + "</tr>" for f in filas
            )
            salida.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
            continue

        # lista
        if re.match(r"[-*]\s+", pelada) or re.match(r"\d+\.\s+", pelada):
            ordenada = bool(re.match(r"\d+\.\s+", pelada))
            etiqueta = "ol" if ordenada else "ul"
            elementos: list[str] = []
            while i < n:
                actual = lineas[i]
                pelada_actual = actual.strip()
                marca = re.match(r"(?:[-*]|\d+\.)\s+(.*)", pelada_actual)
                if marca:
                    elementos.append(marca.group(1))
                    i += 1
                    continue
                # continuacion: linea indentada que pertenece al elemento previo
                if pelada_actual and actual.startswith("  ") and elementos:
                    elementos[-1] += " " + pelada_actual
                    i += 1
                    continue
                break
            lis = "".join(f"<li>{_inline(e)}</li>" for e in elementos)
            salida.append(f"<{etiqueta}>{lis}</{etiqueta}>")
            continue

        # parrafo: junta lineas hasta una vacia o el inicio de otro bloque
        parrafo = [pelada]
        i += 1
        while i < n:
            siguiente = lineas[i].strip()
            if not siguiente or re.match(r"(#{1,4})\s|[-*]\s|\d+\.\s|\|", siguiente):
                break
            if siguiente.startswith("```") or re.fullmatch(r"-{3,}", siguiente):
                break
            parrafo.append(siguiente)
            i += 1
        salida.append(f"<p>{_inline(' '.join(parrafo))}</p>")

    return "\n".join(salida)


_ESTILO = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem 6rem;
  background: #0b0e14; color: #d6dae3;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 56rem; margin: 0 auto; }
nav { max-width: 56rem; margin: 0 auto 2.5rem; padding-bottom: 1rem; border-bottom: 1px solid #222838; }
nav a { color: #8ab4f8; text-decoration: none; margin-right: 1.25rem; font-size: 0.9rem; }
nav a:hover { text-decoration: underline; }
nav .marca { color: #e8ebf0; font-weight: 600; margin-right: 1.75rem; }
h1 { font-size: 1.9rem; margin: 0 0 0.75rem; color: #f2f4f8; letter-spacing: -0.01em; }
h2 { font-size: 1.3rem; margin: 2.75rem 0 0.9rem; padding-top: 0.6rem; color: #f2f4f8;
     border-top: 1px solid #1c2230; }
h3 { font-size: 1.02rem; margin: 1.8rem 0 0.6rem; color: #b9c0cd; text-transform: uppercase;
     letter-spacing: 0.06em; font-weight: 600; }
h4 { font-size: 0.98rem; margin: 1.4rem 0 0.5rem; color: #b9c0cd; }
p { margin: 0 0 1rem; }
ul, ol { margin: 0 0 1.15rem; padding-left: 1.4rem; }
li { margin-bottom: 0.5rem; }
li::marker { color: #5b6476; }
strong { color: #f2f4f8; font-weight: 600; }
code { background: #161b26; border: 1px solid #222838; border-radius: 4px;
       padding: 0.1em 0.38em; font-size: 0.875em; white-space: nowrap;
       font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: #11151e; border: 1px solid #222838; border-radius: 8px;
      padding: 0.9rem 1.1rem; overflow-x: auto; margin: 0 0 1.2rem; }
pre code { background: none; border: 0; padding: 0; font-size: 0.85rem;
           white-space: pre; }
hr { border: 0; border-top: 1px solid #1c2230; margin: 2.25rem 0; }
a { color: #8ab4f8; }
table { width: 100%; border-collapse: collapse; margin: 0 0 1.4rem; font-size: 0.92rem; }
th, td { text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid #1c2230;
         vertical-align: top; }
th { color: #9aa3b2; font-weight: 600; font-size: 0.82rem; text-transform: uppercase;
     letter-spacing: 0.04em; }
.pie { max-width: 56rem; margin: 3.5rem auto 0; padding-top: 1.2rem;
       border-top: 1px solid #1c2230; color: #6b7280; font-size: 0.84rem; }
"""


def _envolver(pagina: str, titulo: str, cuerpo: str, version: str) -> str:
    # Solo se enlaza lo que existe: una pagina sin su fichero no debe aparecer
    # en la navegacion como enlace roto.
    enlaces = "".join(
        f'<a href="/docs/{ruta}">{html.escape(nombre)}</a>'
        for ruta, (fichero, nombre) in PAGES.items()
        if (CONTENT_DIR / fichero).is_file()
    )
    return f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titulo)} · oCabra</title>
<style>{_ESTILO}</style>
</head><body>
<nav><span class="marca">oCabra {html.escape(version)}</span>{enlaces}
<a href="/docs">Referencia de la API</a></nav>
<main>{cuerpo}</main>
<div class="pie">Fuente en Markdown: <code>/docs/{html.escape(pagina)}.md</code></div>
</body></html>"""


def _leer(nombre_fichero: str) -> str:
    ruta = CONTENT_DIR / nombre_fichero
    if not ruta.is_file():
        raise HTTPException(status_code=404, detail="documentation page not found")
    return ruta.read_text(encoding="utf-8")


@router.get(
    "/docs/{pagina}.md",
    include_in_schema=False,
    response_class=PlainTextResponse,
)
async def pagina_markdown(pagina: str) -> PlainTextResponse:
    """Devuelve la fuente en Markdown de una página de documentación.

    Pensado para agentes y para ``curl``: es exactamente el mismo contenido que
    renderiza la versión HTML, sin maquetar.

    Parámetros:
        pagina: ``changelog`` o ``models``.

    Respuesta:
        ``text/markdown`` con el documento, o 404 si la página no existe.
    """
    entrada = PAGES.get(pagina)
    if entrada is None:
        raise HTTPException(status_code=404, detail="documentation page not found")
    return PlainTextResponse(_leer(entrada[0]), media_type="text/markdown; charset=utf-8")


@router.get("/docs/{pagina}", include_in_schema=False, response_class=HTMLResponse)
async def pagina_html(pagina: str) -> HTMLResponse:
    """Renderiza una página de documentación en prosa como HTML.

    Complementa a ``/docs`` (la referencia OpenAPI) con lo que esa no cubre:
    cambios de comportamiento entre versiones y rendimiento de los modelos.

    Parámetros:
        pagina: ``changelog`` o ``models``.

    Respuesta:
        ``text/html`` con la página, o 404 si no existe.
    """
    from ocabra.config import settings

    entrada = PAGES.get(pagina)
    if entrada is None:
        raise HTTPException(status_code=404, detail="documentation page not found")
    fichero, titulo = entrada
    cuerpo = markdown_to_html(_leer(fichero))
    return HTMLResponse(_envolver(pagina, titulo, cuerpo, settings.app_version))
