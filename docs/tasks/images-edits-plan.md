# Plan: `/v1/images/edits` — image editing endpoint

**Fecha:** 2026-06-05
**Estado:** Fase 1 entregada y en producción · Fase 2 (mejoras) pendiente
**Objetivo:** Añadir el endpoint OpenAI-compat `/v1/images/edits` (modificar
imagen existente con prompt + máscara opcional) a oCabra, sin coste extra
de VRAM y compatible con los pipelines de Diffusers ya soportados.

**Contexto:** El stack ya servía `/v1/images/generations` para text2img sobre
SD1.5/SDXL/SD3.5/FLUX.1/FLUX.2-Klein/Z-Image. Faltaba el segundo endpoint
oficial de OpenAI para que clientes (SDK oficial, librerías third-party,
Playground) puedan editar imágenes existentes — img2img + inpainting.

---

## Fase 1 — Endpoint funcional + UI Playground · ✅ entregada

### F1-1: Worker `POST /edit` · ✅

`backend/workers/diffusers_worker.py` añade `POST /edit` que recibe
`prompt`, `image_b64`, opcional `mask_b64`, `strength`, `width/height`,
`negative_prompt`, `num_inference_steps`, `guidance_scale`, `seed`,
`num_images`.

La pipeline de edición se deriva *lazily* de la text2img cargada vía:

```python
edit_pipeline = AutoPipelineForImage2Image.from_pipe(state.pipeline)
inpaint_pipeline = AutoPipelineForInpainting.from_pipe(state.pipeline)
```

`from_pipe()` comparte pesos con la pipeline base — **no carga modelo
extra ni consume VRAM adicional**. Las variantes derivadas se cachean en
`WorkerState`.

Conversión de máscara: OpenAI usa transparencia (alpha=0 → editar);
Diffusers espera grayscale donde blanco = editar. Se invierte el canal
alfa con `_mask_from_alpha()`. Máscaras grayscale pasan tal cual.

Cuando la pipeline base no está en el auto-mapping (FLUX.2 Klein y
Z-Image-Turbo solo tienen text2img), `from_pipe` levanta `ValueError`
y devolvemos `HTTPException(400)` con mensaje específico.

### F1-2: API `POST /v1/images/edits` · ✅

`backend/ocabra/api/openai/images.py` añade el endpoint con
`multipart/form-data`:

| Campo | Tipo | Notas |
|--|--|--|
| `image` | UploadFile | PNG/JPEG/WEBP, obligatorio |
| `mask` | UploadFile | PNG con alfa, opcional |
| `model` | str | profile_id |
| `prompt` | str | obligatorio |
| `n`, `size`, `response_format`, `user` | varios | igual que generations |
| `strength`, `seed`, `negative_prompt`, ... | varios | opcionales |

`response_format` soporta `b64_json` (default) y `url` (guarda PNG en
`image_outputs_dir` y devuelve URL pública firmada por TTL — mismo
janitor que `/generations` ya respeta).

Reusa la capability `image_generation` (no creé una nueva — el mismo
modelo que genera también puede editar si tiene img2img/inpainting en
diffusers). Cuando el worker devuelve 400, la API traduce a códigos
estables:

- `mask_unsupported` — había máscara pero la pipeline no tiene
  inpainting (Z-Image-Turbo, FLUX.2 Klein).
- `edit_unsupported` — no hay máscara y la pipeline tampoco tiene img2img.

### F1-3: Stats + setting · ✅

- `stats/collector.py` mapea `/v1/images/edits` → `image_generation` para
  métricas y cost calculator.
- `config.py` nueva setting `openai_image_max_part_size_mb`
  (`Field(default=25, ge=1)`) — espejo del límite oficial de OpenAI.

### F1-4: Refactor — `_federation.py` · ✅

Los 4 endpoints non-stream (`chat`, `completions`, `embeddings`,
`images/generations`, `images/edits`, `audio/transcriptions`,
`audio/generate`) repetían el mismo bloque de ~50 líneas para hablar con
los peers. Extracto a `backend/ocabra/api/openai/_federation.py` dos
helpers:

```python
fed_resp = await try_proxy_json(request, model_id=..., body=..., ...)
if fed_resp is not None:
    return fed_resp
# fall through to local

fed_resp = await try_proxy_multipart(request, model_id=..., files=..., data=..., ...)
if fed_resp is not None:
    return Response(content=fed_resp.content, ...)
```

Neto: ~70 líneas menos, una sola sede para tunear
`PEER_FALLBACK_STATUSES` o añadir telemetría.

`audio/speech` y la rama streaming de `completions`/`chat` siguen inline
(el helper actual no cubre streaming sin complicar la firma).

### F1-5: UI Playground · ✅

`frontend/src/components/playground/ImageInterface.tsx` añade un toggle
Generar/Editar:

- Modo **Editar**: dos drop-tiles (Imagen base obligatoria + Máscara
  opcional con hint sobre alfa transparente) y slider `strength`
  (0.1–1.0). Width/height se ocultan — la salida mantiene el tamaño de
  la entrada.
- Construye `FormData` y POSTea a `/v1/images/edits`. Soporta respuesta
  `b64_json` o `url`.

### F1-6: Tests · ✅

- **`tests/test_openai_api.py::TestImageEdits`** (9 tests): capability,
  prompt requerido, b64 round-trip, máscara forward, URL persist,
  response_format inválido, los dos códigos 400 (mask/edit unsupported),
  y test de federación con `proxy_multipart`.
- **`tests/workers/test_diffusers_worker.py`** (3 tests): inversión del
  alfa de máscara, passthrough grayscale, y propagación de
  `HTTPException(400)` cuando `from_pipe` falla.

Resultado: 69 passed, 2 skipped (skips solo por falta de PIL en venv
local — corren en el venv del worker que sí lo tiene).

### F1-7: Verificación en producción · ✅

- `docker compose build api frontend` OK.
- `docker compose up -d api frontend` OK; logs limpios, `ocabra_ready`,
  `image_outputs_cleanup_loop_started ttl_s=3600`.
- `GET /openapi.json` (vía caddy:8484) lista `/v1/images/edits`.
- POST sin auth → 401 (correcto: auth antes de validar body).

---

## Avances

- Endpoint funcional, registrado en OpenAPI y desplegado en el stack
  local de producción.
- Pipeline derivation con compartición de pesos — cero coste de VRAM
  para añadir editing a cualquier modelo de imagen ya cargado.
- Códigos de error estables (`mask_unsupported`, `edit_unsupported`)
  para que el cliente pueda degradar elegantemente cuando el modelo
  no soporta máscara.
- UI en Playground para probarlo sin tocar curl.
- Refactor lateral: 4 federation hooks duplicados eliminados, base de
  cara a añadir más endpoints.

---

## Deudas / cuestiones pendientes (Fase 2)

### D-1: Timeout fijo de 300s en `worker_pool.forward_request`

`backend/ocabra/core/worker_pool.py` usa `httpx.AsyncClient(timeout=300.0)`
para todas las llamadas. Inpainting con FLUX en GPU contendida fácilmente
supera ese tiempo → cliente recibe 504 mientras el worker sigue
trabajando.

**Sugerencia:** o bien parametrizar `timeout` por endpoint (image edits
y generations a 600s), o subirlo a 600s global. Afecta también a
generations/embeddings/etc.

### D-2: Combo `from_pipe()` + `enable_model_cpu_offload()` sin validar

Cuando la text2img base tiene offload activo (`DIFFUSERS_OFFLOAD_MODE=model`
/ `sequential`) y derivamos img2img/inpainting con `from_pipe`, los hooks
de offload son compartidos. Puede funcionar correctamente o duplicar
registros y provocar OOM/cuelgue en la primera edit.

**Sugerencia:** probar en un Frankenstein con SDXL + sequential offload
en GPU pequeña, y si hay problema, llamar `enable_model_cpu_offload()` /
`enable_sequential_cpu_offload()` también sobre la pipeline derivada
tras `from_pipe`.

### D-3: Streaming federation en `chat.py` sigue inline

`try_proxy_json` no cubre streaming. La rama remota de chat
completions (la más usada) sigue siendo el bloque copy-paste de antes.

**Sugerencia:** añadir `try_proxy_stream(request, ...)` que devuelva un
`StreamingResponse | None`. El patrón es paralelo a las otras dos
helpers pero la firma necesita pensarse para no atragantar la API con
`async for` anidados.

### D-4: `audio.py` — la setting `openai_audio_max_part_size_mb` no usa `Field`

Pre-existente. Mientras tocaba el config.py para añadir
`openai_image_max_part_size_mb` con `Field(ge=1)`, la setting hermana
sigue como `int = 256` sin validación. Trivial de homogeneizar.

### D-5: No hay tests del path streaming de federación

Ninguno de los endpoints con `proxy_stream` tiene test. Si el patrón
del helper streaming (D-3) llega, vendrá con tests también.

### D-6: `user` field no se reenvía al worker ni separa stats por usuario

Pre-existente en todos los endpoints OpenAI. Para auditoría /
cost-by-user habría que propagar `user` desde el body al worker y a la
fila de stats. Image edits hereda el mismo gap.

### D-7: Validación de tamaño/formato up-front

El endpoint acepta cualquier formato que PIL pueda abrir y cualquier
tamaño hasta `openai_image_max_part_size_mb` (25 MB). OpenAI cap real
es 4 MB y exige PNG. Nuestro permisivismo es probablemente deseable —
pero no documentado.

### D-8: Pipeline cache no se invalida en hot-reload del modelo base

`state.img2img_pipeline` / `state.inpaint_pipeline` se cachean una vez
y nunca se invalidan. El worker actual no recarga el modelo base en
caliente, así que no hay bug — pero si en el futuro se añade reload
de la pipeline base sin reiniciar el worker, las derivadas quedarían
pegadas al objeto antiguo.

**Sugerencia:** invalidar derivadas (`state.img2img_pipeline = None`,
`state.inpaint_pipeline = None`) cuando `load_pipeline()` se ejecute
una segunda vez.

### D-9: Pre-existing — `audio.py:openai_audio_max_part_size_mb` sin Field, B008/I001 en otros endpoints

Auditoría lateral durante el lint:

- `ocabra/api/openai/models.py` — 2× `B008 Depends in defaults`
- `ocabra/api/openai/files.py` — 1× `S110 try/except/pass`
- `ocabra/api/openai/chat.py` — `I001 import block`

No los toqué (fuera de scope). Lista para limpiar en una pasada
dedicada de lint cuando proceda.

---

## Referencias

- Commit: `95fbc6e` — *feat(openai): /v1/images/edits endpoint + federation hook helpers*
- OpenAI spec: https://platform.openai.com/docs/api-reference/images/createEdit
- Diffusers AutoPipeline mapping:
  https://huggingface.co/docs/diffusers/api/pipelines/auto_pipeline
