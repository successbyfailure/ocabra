# Benchmark de Backends Qwen3 — 2026-04-03

## Objetivo

Establecer una referencia reproducible para comparar `vllm`, `tensorrt_llm` y `ollama`
sirviendo variantes Qwen3 en la RTX 3090 del host objetivo.

Este documento sirve como baseline operativo para comparar:
- tiempo de carga (`load`)
- latencia al primer token (`TTFT`)
- throughput de generación (`tok/s`)
- tiempo de descarga (`unload`)
- VRAM retenida tras carga
- comportamiento warm/cold entre primera y segunda request

## Entorno

- Host objetivo:
  - GPU 0: RTX 3060 12 GB
  - GPU 1: RTX 3090 24 GB
- Fecha de prueba: `2026-04-03`
- Stack: `docker compose` del proyecto
- API de control: `http://127.0.0.1:8000`
- GPU objetivo de los modelos LLM probados: RTX 3090

## Modelos probados

No existe en el inventario actual una variante `~8B` exactamente equivalente en
cuantización para todos los backends. Por eso la lectura correcta es en dos carriles:

### Carril fp16 / pesos grandes

- `vllm/Qwen/Qwen3-8B`
- `tensorrt_llm/Qwen3-8B-fp16`

### Carril comprimido

- `vllm/Qwen/Qwen3-8B-AWQ`
- `ollama/qwen3:8b`

Referencias públicas de modelo:
- `Qwen/Qwen3-8B-AWQ`: <https://huggingface.co/Qwen/Qwen3-8B-AWQ>
- `Qwen/Qwen3-32B-AWQ`: <https://huggingface.co/Qwen/Qwen3-32B-AWQ>
- `ollama qwen3:8b`: <https://ollama.com/library/qwen3:8b>

## Metodología

### Flujo por caso

Cada caso se ejecuta de forma aislada con el ciclo:

1. `unload`
2. `load`
3. request 1
4. request 2
5. `unload`

Entre pasos se introducen pausas cortas para evitar medir estados intermedios.

### Prompt

Se usa el mismo prompt para todos los casos:

```text
Output the word BENCH exactly 64 times separated by a single space. No punctuation. No extra text.
```

### Parámetros de inferencia

#### `vllm` y `tensorrt_llm`

Se usa `POST /v1/chat/completions` con:

- `temperature=0`
- `top_p=1`
- `seed=123`
- `max_tokens=160`
- `stream=true`
- `stream_options.include_usage=true`
- `chat_template_kwargs.enable_thinking=false`
- prompt prefijado con `/no_think`

#### `ollama`

Se usa la API nativa `POST http://ollama:11434/api/chat` desde dentro del contenedor `api`,
no la pasarela OpenAI ni la compatibilidad `/api/chat` de oCabra, para obtener métricas
nativas consistentes:

- `think=false`
- `temperature=0`
- `top_p=1`
- `seed=123`
- `num_predict=160`
- `stream=false`

### Métricas registradas

- `load`: tiempo hasta que el backend queda listo para servir
- `TTFT`: tiempo hasta el primer token visible
- `decode_tps`: tokens de completion divididos por la ventana efectiva de decodificación
- `e2e_tps`: tokens de completion divididos por la duración completa de la request
- `unload`: tiempo hasta liberar el modelo
- `VRAM`: VRAM reportada por el estado del modelo cargado (`model.vram_used_mb`)

### Criterio sobre VRAM

La VRAM de referencia para comparación es la del modelo cargado (`vram_used_mb` del
modelo), no el delta bruto de `/ocabra/gpus`.

Motivo:
- el delta de `/ocabra/gpus` incluye procesos transitorios, overhead del runtime y ruido
  del driver
- en modelos cuantizados o con runtimes externos puede sobreestimar de forma fuerte el
  consumo atribuible al modelo

## Resultados

| Backend | Modelo | Load | VRAM | Req1 TTFT | Req1 tok/s | Req2 TTFT | Req2 tok/s | Unload |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm` | `Qwen3-8B` | `61.31 s` | `18.75 GB` | `279 ms` | `48.47` | `88.7 ms` | `48.44` | `1.34 s` |
| `tensorrt_llm` | `Qwen3-8B-fp16` | `109.04 s` | `17.23 GB` | `612 ms` | `49.77` | `78.5 ms` | `51.26` | `7.55 s` |
| `vllm` | `Qwen3-8B-AWQ` | `51.27 s` | `6.98 GB` | `213 ms` | `47.04` | `103 ms` | `47.43` | `2.62 s` |
| `ollama` | `qwen3:8b` | `4.21 s` | `9.81 GB` | `177 ms` aprox | `122.70` | `124 ms` aprox | `114.07` | `61 ms` |

### `e2e tok/s`

- `vllm/Qwen3-8B`: `44.69 -> 47.18`
- `tensorrt_llm/Qwen3-8B-fp16`: `41.81 -> 50.00`
- `vllm/Qwen3-8B-AWQ`: `44.27 -> 46.02`
- `ollama/qwen3:8b`: `87.94 -> 90.14`

## Interpretación

### 1. `vllm` vs `tensorrt_llm` en 8B grande

- El throughput de decode es muy parecido entre ambos.
- `tensorrt_llm` gana algo en la segunda request, pero paga más en `load` y `unload`.
- Con este setup, la experiencia total favorece a `vllm` si importa el tiempo de
  disponibilidad del modelo.

### 2. `vllm/AWQ` es el punto más atractivo del conjunto actual

- Baja a `~7 GB` de VRAM retenida.
- Mantiene throughput muy parecido al carril fp16.
- Reduce de forma significativa la presión de memoria sobre la RTX 3090.

### 3. `ollama/qwen3:8b` es el más rápido operativamente

- `load` muy corto
- `unload` prácticamente inmediato
- throughput claramente superior en esta prueba

Pero no debe compararse como equivalente exacto a `fp16`; es un carril comprimido con
runtime y formato diferentes.

## Problemas y observaciones detectados durante las pruebas

### 1. La comparación entre backends no es estrictamente isocuanta

No hay hoy en el inventario un trío `~8B` equivalente en cuantización para
`vllm`, `tensorrt_llm` y `ollama`.

Impacto:
- las cifras son útiles operativamente
- no deben interpretarse como benchmark académico “same weights, same quant, same runtime budget”

### 2. `/ocabra/gpus` no sirve como fuente única de memoria atribuible por modelo

Durante las pruebas aparecieron deltas de GPU muy superiores a la VRAM retenida por el
modelo, especialmente en runtimes cuantizados o con procesos auxiliares.

Impacto:
- para benchmark, `gpu_delta` es útil como señal bruta de presión
- no es fiable como cifra exacta de memoria del modelo

Acción recomendada:
- usar `model.vram_used_mb` como referencia principal
- considerar métricas separadas de “VRAM total runtime” frente a “VRAM atribuida al modelo”

### 3. `ollama` vía `/api/chat` de oCabra no fue una base robusta para la primera tanda

En una primera tanda, medir `ollama` a través de la compatibilidad `/api/chat` de oCabra
acabó en una request demasiado larga por no fijar explícitamente `num_predict`.

Impacto:
- la medición quedó contaminada
- hubo que repetir el caso usando la API nativa de Ollama para obtener métricas limpias

Acción recomendada:
- para benchmarking de `ollama`, usar API nativa o asegurar en la pasarela que el límite
  de generación queda fijado explícitamente

Estado posterior:
- corregido en oCabra: la compatibilidad Ollama ya promociona `max_tokens -> num_predict`
  y reenvía el `backend_model_id` nativo al runtime
- validado después del benchmark con `POST /api/chat` contra `ollama/qwen3:8b`, devolviendo
  `OK` con límite corto de generación

### 4. Inconsistencia detectada de `context_length` en `tensorrt_llm/Qwen3-8B-fp16`

Durante la prueba, el modelo respondió con:

- `extra_config.context_length = 8192`
- `capabilities.context_length = 512`

Eso es inconsistente y debe tratarse como bug o deriva contractual.

Impacto:
- la UI o clientes podrían interpretar mal el contexto efectivo del modelo
- complica benchmarking y validación automática

Estado posterior:
- corregido en oCabra: `TensorRT-LLM` ahora deriva `context_length` desde el `config.json`
  real del engine y aplica fallback desde `extra_config` en el estado cargado
- validado después del benchmark: `capabilities.context_length = 8192` y
  `extra_config.context_length = 8192`

### 5. Warmup visible en primera request

`vllm` y, sobre todo, `tensorrt_llm` mostraron diferencia clara entre primera y segunda
request, especialmente en TTFT.

Impacto:
- para producción tiene sentido distinguir métricas “cold after load” y “warm steady-state”
- una sola request no representa bien el comportamiento real tras estabilización

## Recomendaciones para benchmarks futuros

- Mantener siempre el patrón `load -> req1 -> req2 -> unload`.
- Registrar siempre:
  - `load`
  - `req1 TTFT`
  - `req1 tok/s`
  - `req2 TTFT`
  - `req2 tok/s`
  - `unload`
  - `vram_used_mb`
  - `prompt_tokens`
  - `completion_tokens`
- Separar explícitamente carriles `fp16`, `AWQ/GPTQ`, `GGUF`, etc.
- Para `ollama`, fijar `num_predict` en benchmark.
- Para `vllm`, mantener `enable_thinking=false` y `/no_think` cuando el modelo lo soporte.
- Guardar siempre si el benchmark fue:
  - `heuristic`
  - `runtime real`
  - `cold`
  - `warm`

## Siguiente benchmark recomendado

El siguiente baseline útil y más exigente es el carril grande:

- `vllm/Qwen/Qwen3-32B-AWQ`
- `tensorrt_llm/Qwen3-32B-AWQ-tp2-fp16`
- `ollama/qwen3:32b`

Eso permitirá comparar:
- coste real de carga en modelos grandes
- impacto de tensor parallel en `tensorrt_llm`
- viabilidad operativa en la RTX 3090
