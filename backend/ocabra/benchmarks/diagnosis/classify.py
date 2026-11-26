"""Categoría de cada caso fallido a partir de la evidencia reunida.

Orden de las reglas (la primera que aplica gana):

1. **servido**: el arnés del benchmark o el servido lo explican sin mirar al
   modelo (p. ej. BFCL sin ``underscore_to_dot``), o el mismo caso pasa con otra
   cuantización/vía y no con esta.
2. **limites**: salida cortada por el tope (``finish_reason=length`` o justo en
   el tope), o el prompt no cabe en el contexto por slot.
3. **plantilla**: el prompt servido difiere de verdad del que produce la
   plantilla oficial con los mismos mensajes.
4. **sampling**: con el muestreo del fabricante pasa la mayoría de las
   repeticiones y lo efectivo era distinto.
5. **no_reproduce**: vuelve a pasar con los mismos parámetros (ruido o algo
   que ya se arregló desde la ejecución).
6. **modelo**: falla también con el muestreo recomendado. Solo esto pasa a
   candidato de entrenamiento.
7. **sin_verificar**: no se pudo repetir (app sin repetición, modelo sin
   cargar o caso fuera de la muestra).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

CATEGORIES = (
    "plantilla",
    "sampling",
    "limites",
    "servido",
    "modelo",
    "no_reproduce",
    "sin_verificar",
)
# Las que el encargo pide en el resumen; las otras dos se cuentan aparte.
CAUSES = ("plantilla", "sampling", "limites", "servido", "modelo")


def _majority(rows: list[dict[str, Any]]) -> bool | None:
    scored = [row for row in rows if row.get("passed") is not None]
    if not scored:
        return None
    return sum(1 for row in scored if row["passed"]) * 2 > len(scored)


def _any_length(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("finish_reason") == "length" for row in rows)


def classify_case(
    case: dict[str, Any],
    *,
    template: dict[str, Any] | None,
    sampling_diffs: dict[str, Any],
    replays: dict[str, list[dict[str, Any]]],
    context_per_slot: int,
    prompt_tokens: int | None,
) -> tuple[str, list[str]]:
    """Devuelve ``(categoría, evidencias)`` de un caso."""
    signals = case.get("signals") or {}
    evidence: list[str] = []

    if signals.get("harness_dot_to_underscore"):
        return "servido", [
            "arnés de BFCL: el modelo llamó a la función correcta con «_» en vez de «.» "
            "y el runner no registra underscore_to_dot"
        ]

    recommended = replays.get("recomendado") or []
    original = replays.get("original") or []
    compared = replays.get("comparado") or []

    if compared and _majority(compared) and _majority(recommended) is False:
        return "servido", ["pasa con la cuantización/vía comparada y no con esta"]

    if signals.get("hit_output_cap") or int(signals.get("length_finishes") or 0) > 0:
        evidence.append("la salida original llegó al tope de tokens")
    if _any_length(original):
        evidence.append("al repetir con sus parámetros se corta por finish_reason=length")
    if prompt_tokens and context_per_slot and prompt_tokens > context_per_slot:
        evidence.append(f"el prompt ({prompt_tokens} tok) no cabe en el slot ({context_per_slot})")
    if evidence:
        if recommended and _majority(recommended) and not _any_length(recommended):
            evidence.append("con el muestreo recomendado ya no se corta y pasa")
        return "limites", evidence

    if template and template.get("status") == "different":
        return "plantilla", ["el prompt servido difiere del de la plantilla oficial"]

    verdict = _majority(recommended)
    if verdict is None:
        reason = "no se pudo repetir el caso"
        if not case.get("replayable"):
            reason = (
                f"esta ejecución de {case.get('app')} no guarda el caso para repetirlo "
                "(mide con la versión actual de la app)"
            )
        return "sin_verificar", [reason]
    if verdict and _any_length(recommended):
        return "limites", ["con el muestreo recomendado se corta por finish_reason=length"]
    if verdict and sampling_diffs:
        changed = ", ".join(sorted(sampling_diffs))
        return "sampling", [f"pasa con el muestreo del fabricante (cambia: {changed})"]
    if verdict or _majority(original):
        return "no_reproduce", ["vuelve a pasar al repetirlo: ruido o algo ya arreglado"]
    return "modelo", ["falla también con el muestreo recomendado"]


def summarize(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["category"] for item in verdicts)
    total = len(verdicts)
    verified = sum(counts[name] for name in CAUSES)
    return {
        "total": total,
        "counts": {name: counts.get(name, 0) for name in CATEGORIES},
        "percent_of_failures": {
            name: round(100.0 * counts.get(name, 0) / total, 1) if total else 0.0
            for name in CATEGORIES
        },
        # Sobre los casos con causa atribuida (sin ruido ni sin verificar).
        "percent_of_attributed": {
            name: round(100.0 * counts.get(name, 0) / verified, 1) if verified else 0.0
            for name in CAUSES
        },
    }
