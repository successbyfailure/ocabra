"""Páginas de documentación en prosa: /docs/changelog y /docs/models.

Complementan al Swagger de ``/docs``, que documenta la forma de los endpoints
pero no los cambios de comportamiento entre versiones. Se sirven en dos
formatos desde la misma fuente: HTML para personas y Markdown crudo (sufijo
``.md``) para agentes.
"""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.docs_pages import CONTENT_DIR, PAGES, markdown_to_html
from ocabra.api.docs_pages import router as docs_router


@pytest.fixture
def client():
    app = FastAPI(docs_url="/docs", redoc_url="/redoc")
    app.include_router(docs_router)
    return TestClient(app)


# ── Rutas ────────────────────────────────────────────────────────────────────


def test_el_swagger_de_docs_sigue_funcionando(client):
    """``/docs/{pagina}`` no debe capturar el ``/docs`` de FastAPI."""
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower()


def test_oauth2_redirect_de_swagger_sigue_accesible(client):
    # FastAPI registra /docs/oauth2-redirect; mi ruta comodin no debe comersela.
    resp = client.get("/docs/oauth2-redirect")
    assert resp.status_code == 200


@pytest.mark.parametrize("pagina", sorted(PAGES))
def test_cada_pagina_se_sirve_en_html(client, pagina):
    resp = client.get(f"/docs/{pagina}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<h1" in resp.text
    # navegacion y enlace a la referencia de la API
    assert 'href="/docs"' in resp.text


@pytest.mark.parametrize("pagina", sorted(PAGES))
def test_cada_pagina_se_sirve_en_markdown_crudo(client, pagina):
    resp = client.get(f"/docs/{pagina}.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    # es la misma fuente, sin maquetar
    fichero = PAGES[pagina][0]
    assert resp.text == (CONTENT_DIR / fichero).read_text(encoding="utf-8")
    assert "<html" not in resp.text


def test_una_pagina_inexistente_da_404(client):
    assert client.get("/docs/no-existe").status_code == 404
    assert client.get("/docs/no-existe.md").status_code == 404


def test_la_pagina_lleva_la_version_servida(client):
    from ocabra.config import settings

    assert settings.app_version in client.get("/docs/changelog").text


def test_el_changelog_menciona_la_version_mas_reciente(client):
    """Si se sube la version sin anotar el cambio, el changelog se queda viejo
    en silencio. Esto no valida el contenido, solo que existe la entrada."""
    from ocabra.config import settings

    md = client.get("/docs/changelog.md").text
    mayor_menor = ".".join(settings.app_version.split(".")[:2])
    assert mayor_menor in md, f"el changelog no menciona la serie {mayor_menor}"


# ── Renderizador de Markdown ─────────────────────────────────────────────────


def test_encabezados_con_ancla():
    html = markdown_to_html("## Cambios de comportamiento")
    assert '<h2 id="cambios-de-comportamiento">' in html


def test_el_ancla_de_un_encabezado_pliega_los_acentos():
    # Sin plegar, "Qué falla y por qué" daba "qu-falla-y-por-qu" y los enlaces
    # internos de la propia pagina apuntaban a un id que no existia.
    html = markdown_to_html("## Qué falla y por qué")
    assert '<h2 id="que-falla-y-por-que">' in html


def test_los_enlaces_internos_de_cada_pagina_apuntan_a_un_encabezado_real():
    for nombre, _ in PAGES.values():
        fuente = (CONTENT_DIR / nombre).read_text(encoding="utf-8")
        html = markdown_to_html(fuente)
        ids = set(re.findall(r'<h[1-4] id="([^"]+)"', html))
        for destino in re.findall(r'href="#([^"]+)"', html):
            assert destino in ids, f"{nombre}: ancla #{destino} sin encabezado"


def test_negrita_y_codigo_en_linea():
    html = markdown_to_html("Pon **max_tokens** o usa `reasoning_effort`.")
    assert "<strong>max_tokens</strong>" in html
    assert "<code>reasoning_effort</code>" in html


def test_listas():
    html = markdown_to_html("- uno\n- dos\n")
    assert html.count("<li>") == 2
    assert "<ul>" in html


def test_una_linea_indentada_continua_el_elemento_anterior():
    html = markdown_to_html("- primera parte\n  segunda parte\n")
    assert html.count("<li>") == 1
    assert "primera parte segunda parte" in html


def test_lista_ordenada():
    html = markdown_to_html("1. uno\n2. dos\n")
    assert "<ol>" in html and html.count("<li>") == 2


def test_bloque_de_codigo_no_se_interpreta():
    html = markdown_to_html("```\n## no es un encabezado\n**ni negrita**\n```")
    assert "<pre><code>" in html
    assert "<h2" not in html
    assert "<strong>" not in html


def test_tabla():
    html = markdown_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in html and "<th>a</th>" in html and "<td>2</td>" in html


def test_regla_horizontal():
    assert "<hr>" in markdown_to_html("uno\n\n---\n\ndos")


def test_enlaces():
    html = markdown_to_html("ver el [changelog](/docs/changelog)")
    assert '<a href="/docs/changelog">changelog</a>' in html


def test_el_html_del_contenido_se_escapa():
    """El contenido es nuestro, pero escapar evita que un ejemplo con etiquetas
    rompa la pagina."""
    html = markdown_to_html("Usa <script>alert(1)</script> con cuidado")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_el_codigo_en_bloque_tambien_se_escapa():
    html = markdown_to_html("```\n<div>hola</div>\n```")
    assert "<div>hola</div>" not in html
    assert "&lt;div&gt;" in html


def test_los_parrafos_se_juntan_hasta_la_linea_vacia():
    html = markdown_to_html("una linea\nsigue aqui\n\notro parrafo")
    assert html.count("<p>") == 2
    assert "una linea sigue aqui" in html


# ── Descubribilidad desde /docs ──────────────────────────────────────────────


@pytest.mark.parametrize("pagina", sorted(PAGES))
def test_la_referencia_de_la_api_enlaza_cada_pagina(pagina):
    """Quien aterriza en el Swagger tiene que encontrar estas paginas.

    La descripcion de /docs es un bloque largo de Markdown que se edita a mano;
    sin esto, un enlace puede desaparecer en una reescritura sin que nadie se
    entere.
    """
    from ocabra.main import _API_DESCRIPTION

    assert f"(/docs/{pagina})" in _API_DESCRIPTION
    assert f"(/docs/{pagina}.md)" in _API_DESCRIPTION


def test_la_descripcion_llega_al_openapi():
    """El enlace solo sirve si viaja en el esquema que renderiza Swagger."""
    from ocabra.main import app

    descripcion = app.openapi().get("info", {}).get("description", "")
    assert "/docs/changelog" in descripcion


def test_la_referencia_documenta_todos_los_endpoints_de_batches():
    """La lista de endpoints admitidos en lote vive en el codigo; la tabla de
    /docs se escribe a mano. Ya se han desincronizado antes: rerank, score,
    pooling, classify y audio/generate se añadieron a ALLOWED_ENDPOINTS sin
    aparecer en la documentacion, asi que nadie sabia que podia usarlos."""
    from ocabra.api.openai.batches import ALLOWED_ENDPOINTS
    from ocabra.main import _API_DESCRIPTION

    sin_documentar = sorted(e for e in ALLOWED_ENDPOINTS if f"`{e}`" not in _API_DESCRIPTION)
    assert not sin_documentar, f"endpoints de Batches sin documentar en /docs: {sin_documentar}"


# ── Markdown crudo frente a navegador ───────────────────────────────────────

NAVEGADOR = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,*/*;q=0.8"


@pytest.mark.parametrize("pagina", ["changelog", "models"])
def test_un_navegador_recibe_la_pagina_maquetada_en_la_url_md(client, pagina):
    """Abrir /docs/models.md en el navegador mostraba el Markdown como texto."""
    resp = client.get(f"/docs/{pagina}.md", headers={"Accept": NAVEGADOR})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<table" in resp.text or "<h1" in resp.text
    assert resp.headers["vary"] == "Accept"


@pytest.mark.parametrize(
    "accept",
    [
        "",
        "*/*",
        "text/markdown",
        "text/plain",
        "text/markdown, text/html;q=0.9",
        "text/html;q=0.5, text/plain",
    ],
)
def test_curl_y_agentes_siguen_recibiendo_markdown(client, accept):
    headers = {"Accept": accept} if accept else {}
    resp = client.get("/docs/models.md", headers=headers)
    assert resp.headers["content-type"].startswith("text/markdown")
    assert resp.text.startswith("# Modelos")


def test_raw_fuerza_el_markdown_desde_el_navegador(client):
    resp = client.get("/docs/models.md?raw=1", headers={"Accept": NAVEGADOR})
    assert resp.headers["content-type"].startswith("text/markdown")
    assert resp.text.startswith("# Modelos")


def test_el_pie_enlaza_a_la_fuente_cruda(client):
    assert 'href="/docs/models.md?raw=1"' in client.get("/docs/models").text
