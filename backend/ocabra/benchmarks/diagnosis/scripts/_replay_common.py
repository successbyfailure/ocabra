"""Piezas comunes de los scripts de repetición (solo stdlib).

Los scripts se ejecutan dentro del entorno de cada app (``uv run``), así que no
pueden importar oCabra: este fichero se carga por ruta.

Entrada (JSON en ``--input``)::

    {"base_url": "...", "cases": [{"case_id": ..., "payload": {...}}],
     "settings": [{"name": "original", "model": "...", "params": {...},
                   "repeats": 1}, ...]}

La clave va en ``OPENAI_COMPATIBLE_KEY``. Cada repetición se escribe como una
línea JSON en ``--output`` según termina, para poder seguir el progreso.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    return parser.parse_args()


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def chat(base_url, model, messages, params, tools=None, timeout=900):
    """Una petición por la cola normal de oCabra. Reintenta 429/503 con Retry-After."""
    body = {"model": model, "messages": messages, "stream": False, **params}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + os.environ["OPENAI_COMPATIBLE_KEY"],
        "Content-Type": "application/json",
    }
    for attempt in range(6):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504) and attempt < 5:
                retry = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(min(120.0, float(retry) if retry else 5.0 * (attempt + 1)))
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 5:
                time.sleep(5.0 * (attempt + 1))
                continue
            raise RuntimeError(str(exc)) from exc
    raise RuntimeError("reintentos agotados")


def summarize(response):
    choice = (response.get("choices") or [{}])[0]
    usage = response.get("usage") or {}
    return choice.get("message") or {}, {
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def run(spec_path, output_path, score):
    """Repite cada caso en cada ajuste y deja una línea por repetición.

    ``score(case, setting, rep)`` devuelve un dict con al menos ``passed``.
    """
    spec = load(spec_path)
    with open(output_path, "a", encoding="utf-8") as out:
        for case in spec["cases"]:
            for setting in spec["settings"]:
                for rep in range(int(setting.get("repeats") or 1)):
                    started = time.monotonic()
                    row = {"case_id": case["case_id"], "setting": setting["name"], "rep": rep}
                    try:
                        row.update(score(spec["base_url"], case, setting))
                    except Exception as exc:  # noqa: BLE001 — se registra y se sigue
                        row.update({"passed": None, "error": f"{type(exc).__name__}: {exc}"[:400]})
                    row["latency_s"] = round(time.monotonic() - started, 2)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
