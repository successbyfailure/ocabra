# Encargos pendientes a otros proyectos (diagnóstico de fallos)

El diagnóstico de oCabra (`docs/benchmark-diagnosis.md`) solo puede repetir y
clasificar casos de IFEval y BFCL. Para Corrala, Fulcrum y Escriba hace falta el
mismo contrato en cada proyecto. Encargados el 26/11 a los tres (Corrala cd6f1da1,
Fulcrum 6cce4a1e, Escriba 56d7da2a); este fichero queda como referencia del contrato.

## Contrato común

1. **Por caso**, en `runs.jsonl` junto al `results.json`: `case_id` estable,
   `messages` exactos de la llamada decisiva (formato OpenAI, system incluido),
   `tools`, `response`, `finish_reason`, `prompt_tokens`, `completion_tokens`,
   `score` (0–1) y `passed`.
2. **Repetición de casos sueltos**: `ocabra_replay.py` solo-stdlib junto al
   `ocabra_suite.py`:
   `python -I .../ocabra_replay.py --model <perfil> --cases cases.json --repeats N --output out.jsonl`
   con `cases.json = {"case_ids": [...]}`. Una línea por repetición, según
   termina: `{"case_id", "rep", "passed", "score", "finish_reason",
   "prompt_tokens", "completion_tokens", "error"}`; `error` solo para
   infraestructura (`passed: null`).
3. Sin muestreo propio distinto del de la suite: oCabra aplica en el servidor el
   muestreo a comparar (ejecuciones con «muestreo del fabricante»).
4. Variables: `OPENAI_COMPATIBLE_URL` (raíz sin `/v1`), `OPENAI_COMPATIBLE_KEY`.

## Fulcrum (`benchmarks/orchestration/`)

- **Aprobado por omisión.** `flint:40b`: 5/5 repeticiones en bucle, `cola` y
  `completo` a 0, y aun así `capability_rate` 0,778. Cada capacidad debe exigir
  evidencia mínima y, sin ella, contar como suspendida: `despacho`/`texto_fijo`
  al menos un `run_subagent` aceptado (y un `send_to_agent` para el texto de
  fase); `estado_max`/`orden_fases`/`no_fase_fantasma` todas las runs lanzadas;
  `honestidad` al menos un turno con texto y una acción; `aviso_fase3` al menos
  una run terminada. Test: transcripción tipo flint → `capability_rate` ≤ 0,2.
- Contrato común con «caso» = repetición de escenario: `case_id`
  `<escenario>:<rep>` y `scenario` en cada línea, `finish_reasons` por tipo;
  `ocabra_replay.py` corre N repeticiones nuevas del escenario; `passed` = todas
  las capacidades, `score` = `capability_rate`, `caps`, `finish_reason` el más
  grave (length > tool_loop > stop).

## Escriba (suite `llm`, `eval/datasets/llm/`)

- Contrato común con `case_id` = transcripción (o `transcripción:tamaño`),
  `response` (texto/JSON), `expected` (claims de referencia por id), `score`
  (p. ej. media armónica de cobertura y precisión) y `passed` por un umbral
  documentado (p. ej. JSON válido y score ≥ 0,6). Las transcripciones que no
  deban salir del repo, solo por id.
- `eval/datasets/llm/ocabra_replay.py` según el contrato común.
