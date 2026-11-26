# Diagnóstico de fallos de benchmark: ¿es el modelo o es cómo lo servimos?

Antes de entrenar (LoRA/QLoRA) un modelo para los casos en los que falla, hay
que descartar que el fallo venga del servido. Esta herramienta toma un intento
completado de un benchmark y clasifica cada caso fallido.

## Cómo se lanza

- **UI:** Benchmarks → una ejecución → «¿Es el modelo o es cómo lo servimos?» →
  *Diagnosticar &lt;modelo&gt;*.
- **API:** `POST /ocabra/benchmarks/attempts/{attempt_id}/diagnoses` con
  `{"repeats": 3, "max_cases": 30, "compare_model": null, "reference_repo": null, "allow_load": false}`.
- **CLI** (cliente de la API, solo stdlib):

  ```bash
  OCABRA_URL=http://127.0.0.1:8484 OCABRA_API_KEY=<clave model_manager> \
    python -m ocabra.benchmarks.diagnosis --run <run_id> --model gemma4:12b --out ./diag
  ```

Lo ejecuta la cola de benchmarks cuando no hay ninguno pendiente: las
repeticiones son tráfico y no deben coincidir con una medida. Entran por la cola
normal de oCabra con la identidad de servicio de los benchmarks. Si el modelo
no está cargado, el diagnóstico no lo carga (podría expulsar un PIN) salvo con
`allow_load: true`; sin él hace solo las comprobaciones que no necesitan GPU.

## Qué comprueba

1. **Plantilla.** Prompt renderizado por el llama-server (`/apply-template`)
   frente al de la plantilla oficial de Hugging Face (`chat_template.jinja` o
   `tokenizer_config.json`, renderizada con jinja2 como `apply_chat_template`).
   Sondas: si `reasoning_effort` y `enable_thinking` cambian de verdad el
   prompt, `content` como lista frente a string, rol system, BOS duplicado y una
   conversación con tool call + resultado. Por caso, el diff con sus mensajes.
2. **Parámetros efectivos.** Lo que llega al backend tras `request_defaults`,
   presets y el cuerpo del runner (`merge_profile_defaults` →
   `apply_declared_output_limit` → `to_backend_body`), con los valores por
   defecto del llama-server debajo, frente al `generation_config.json` del
   fabricante. También avisa de `suppress_tokens` que llama.cpp no aplica.
3. **Repetición.** Una vez con los parámetros del benchmark y N veces con el
   muestreo del fabricante (sin min_p, repetición, DRY ni penalizaciones),
   puntuado con el checker oficial de la app dentro de su entorno.
4. **Límites.** Prompt frente al contexto por slot; salida cortada
   (`finish_reason=length` o justo en el tope).
5. **Cuantización / servido.** Opcional (`compare_model`): las mismas
   repeticiones con otro perfil (p. ej. Q8). Además, el arnés: BFCL se
   re-puntúa con `underscore_to_dot`.

Apps: IFEval y BFCL completas. Corrala y Fulcrum no guardan los mensajes de cada
caso: solo límites y señales (bucles, salidas cortadas), el resto queda «sin
verificar». Escriba aún no.

## Categorías

`servido` (arnés o cuantización), `limites`, `plantilla`, `sampling`,
`no_reproduce` (vuelve a pasar con los mismos parámetros), `modelo` (falla
también con el muestreo recomendado) y `sin_verificar`. Solo `modelo` pasa a
candidato de entrenamiento. Orden de las reglas en
`backend/ocabra/benchmarks/diagnosis/classify.py`.

## Ficheros

En `<benchmark_dir>/diagnoses/<id>/`: `report.json`, `report.md`,
`replays.jsonl` (cada repetición), `candidates.train.jsonl` y
`candidates.eval.jsonl`.

### Formato de los candidatos

Una línea JSON por caso «modelo»:

| Campo | Contenido |
|---|---|
| `id` | id estable del caso (`bfcl:simple_python_12`, `ifeval:1122`…) |
| `app`, `suite`, `category`, `model` | de dónde sale |
| `messages`, `tools` | lo que vio el modelo (formato OpenAI) |
| `response` | lo que respondió en el benchmark |
| `expected` | respuesta esperada o restricciones (BFCL: `possible_answer`; IFEval: instrucciones y kwargs) |
| `failure`, `score` | por qué falla y su puntuación |
| `replays` | las repeticiones del diagnóstico |
| `split` | `train` o `eval` |
| `source` | `run_id`, `attempt_id`, `revision` |

**Separación de evaluación.** El lado de cada caso sale de un hash estable de
su id (20 % a `eval`), así que un caso cae siempre en el mismo lado sea cual sea
el modelo o la ejecución. Los de `eval` no se usan para entrenar: son los que
dirán si el fine-tune mejora. Todos salen del propio benchmark, así que para
medir sin sesgo compara el modelo entrenado sobre los casos `eval` (o sobre
datos nuevos), no sobre la suite completa.
