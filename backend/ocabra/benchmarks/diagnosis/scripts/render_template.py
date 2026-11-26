"""Renderiza una plantilla de chat de Hugging Face con jinja2 (stdin → stdout).

Reproduce el entorno de ``transformers.apply_chat_template`` sin cargar
transformers: entorno sandbox con ``trim_blocks``/``lstrip_blocks``, el filtro
``tojson`` sin escapar HTML, ``raise_exception`` y ``strftime_now``. Se ejecuta
con ``uv run --with jinja2`` porque el api no lleva jinja2.

Entrada (JSON): ``template``, ``messages``, ``tools``, ``bos_token``,
``eos_token`` y ``kwargs`` (como ``chat_template_kwargs``).
Salida (JSON): ``{"prompt": "..."}``.
"""

import json
import sys
from datetime import datetime

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise(message):
    raise TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def main():
    data = json.loads(sys.stdin.read())
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    template = env.from_string(data["template"])
    variables = {
        "messages": data["messages"],
        "add_generation_prompt": True,
        "bos_token": data.get("bos_token") or "",
        "eos_token": data.get("eos_token") or "",
        **(data.get("kwargs") or {}),
    }
    if data.get("tools"):
        variables["tools"] = data["tools"]
    print(json.dumps({"prompt": template.render(**variables)}))


if __name__ == "__main__":
    main()
