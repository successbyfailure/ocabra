# Plan: Compilación de engines TensorRT-LLM desde la UI

**Fecha:** 2026-03-31
**Estado:** PENDIENTE DE IMPLEMENTACIÓN

---

## Objetivo

Permitir al usuario seleccionar un modelo HuggingFace ya descargado y lanzar la compilación
de un engine TensorRT-LLM directamente desde la UI de oCabra, con soporte para 1 GPU o 2 GPUs
en tensor parallelism.

Una vez compilado, el engine queda registrado como modelo `tensorrt_llm/*` listo para cargar.

---

## Contexto de hardware

| GPU | VRAM | Rol |
|-----|------|-----|
| GPU 0 — RTX 3060 | 12 GB | Secondary / TP slave |
| GPU 1 — RTX 3090 | 24 GB | Primary |

**Caso de uso principal:** `Qwen/Qwen3.5-27B-GPTQ-Int4` o `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
en GPU 1 sola (cabe en 24 GB). La opción de 2 GPUs se reserva para modelos que no caben
en la 3090 individualmente (>24 GB en la precisión elegida).

---

## Flujo de usuario

```
Página Explore / Models
  → modelo descargado seleccionado
  → botón "Compilar para TensorRT-LLM"
  → modal de configuración:
      - GPU target: [ GPU 1 - 3090 (recomendado) | GPU 0 - 3060 | Ambas GPUs (TP=2) ]
      - Dtype: [ fp16 | bf16 | int8 | fp8 ]
      - Max batch size: [ 1 | 4 | 8 | 16 ]  (default: 1 para uso interactivo)
      - Max input len: [ 512 | 2048 | 8192 | 32768 ]
      - Max seq len: [ 1024 | 4096 | 16384 | 65536 ]
      - Nombre del engine (editable, default: nombre del modelo)
      - AVISO si se eligen 2 GPUs asimétricas: "La 3060 es más lenta que la 3090.
        Usar ambas GPUs sólo si el modelo no cabe en la 3090 sola."
  → botón "Iniciar compilación"
  → progreso en tiempo real (SSE):
      Paso 1/2: Convirtiendo pesos HF → formato TRT-LLM...  [barra]
      Paso 2/2: Compilando engine para GPU X...             [barra + tiempo estimado]
      ✅ Engine listo — disponible como tensorrt_llm/Qwen/Qwen3.5-27B
  → botón "Cargar modelo ahora"
```

---

## Arquitectura técnica

### 1. Backend — nuevo endpoint

```
POST /ocabra/trtllm/compile
Body: {
  "model_id": "vllm/Qwen/Qwen3.5-27B-GPTQ-Int4",   // modelo HF origen
  "gpu_indices": [1],                                // [1] o [0, 1] para TP
  "dtype": "fp16",
  "max_batch_size": 1,
  "max_input_len": 2048,
  "max_seq_len": 4096,
  "engine_name": "Qwen3.5-27B-fp16"                 // nombre carpeta destino
}
Response: { "job_id": "...", "stream_url": "/ocabra/trtllm/compile/{job_id}/stream" }

GET /ocabra/trtllm/compile/{job_id}/stream          // SSE con progreso
GET /ocabra/trtllm/compile                          // listar compilaciones activas/historial
DELETE /ocabra/trtllm/compile/{job_id}              // cancelar compilación en curso
```

### 2. Backend — CompileManager

Nuevo módulo `backend/ocabra/core/trtllm_compile_manager.py`:

- Recibe job de compilación, lo encola (un job a la vez — bloquea GPU)
- Verifica que la GPU no esté en uso por otro modelo cargado antes de empezar
- Ejecuta en background dos fases via `asyncio.create_subprocess_exec`:
  - **Fase 1 — convert:** `trtllm-convert` o equivalente Docker
  - **Fase 2 — build:** `trtllm-build` con los parámetros configurados
- Publica progreso en Redis canal `trtllm:compile:{job_id}`
- Al terminar, registra automáticamente el engine en `ModelManager` como
  `tensorrt_llm/{nombre}`
- Almacena historial de compilaciones en tabla nueva `trtllm_compile_jobs`

### 3. Ejecución via Docker (igual que el runtime)

La compilación se lanza en el mismo contenedor TRT-LLM que el runtime:

```bash
# Fase 1 — convert
docker run --rm --gpus "device=1" \
  -v /docker/ai-models/ocabra/models:/data/models \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  trtllm-convert \
    --model_dir /data/models/huggingface/Qwen--Qwen3.5-27B-GPTQ-Int4 \
    --output_dir /data/models/tensorrt_llm/Qwen3.5-27B-fp16/tllm_ckpt \
    --dtype fp16 --tp_size 1

# Fase 2 — build
docker run --rm --gpus "device=1" \
  -v /docker/ai-models/ocabra/models:/data/models \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  trtllm-build \
    --checkpoint_dir /data/models/tensorrt_llm/Qwen3.5-27B-fp16/tllm_ckpt \
    --output_dir /data/models/tensorrt_llm/Qwen3.5-27B-fp16/engine \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --tp_size 1
```

Para 2 GPUs: `--gpus "device=0,1"` y `--tp_size 2` en ambas fases.

### 4. Tabla BD nueva: `trtllm_compile_jobs`

```sql
id            UUID PK
source_model  TEXT    -- modelo HF origen
engine_name   TEXT    -- nombre del engine resultante
gpu_indices   JSON    -- [1] o [0, 1]
dtype         TEXT
config        JSON    -- max_batch_size, max_input_len, max_seq_len
status        TEXT    -- pending | running | done | failed | cancelled
phase         TEXT    -- convert | build | null
progress_pct  INT
error_detail  TEXT
engine_dir    TEXT    -- ruta final del engine
started_at    TIMESTAMPTZ
finished_at   TIMESTAMPTZ
```

### 5. Frontend — CompileModal

Nuevo componente `frontend/src/components/models/CompileModal.tsx`:

- Formulario con los parámetros descritos en el flujo de usuario
- Lógica de validación: estimar VRAM necesaria según dtype + tamaño del modelo
  y advertir si no cabe en la GPU seleccionada
- Progreso via `useSSE` al stream endpoint
- Integración en `Models.tsx` y `Explore.tsx` (botón en modelos descargados)

---

## Estimación de VRAM por configuración (referencia para validación en UI)

| Modelo | fp16 | int8 | fp8 | int4 |
|--------|------|------|-----|------|
| Qwen3.5-27B | ~54 GB | ~27 GB | ~27 GB | ~14 GB |
| Qwen3.5-35B-A3B (MoE) | ~14 GB activos | ~8 GB | ~8 GB | ~6 GB |
| Qwen3.5-9B | ~18 GB | ~9 GB | ~9 GB | ~5 GB |

> Para MoE el cálculo es sobre parámetros activos (A3B = ~3B activos).
> La UI debe mostrar estimación y advertir si supera la VRAM de la GPU seleccionada.

---

## Restricciones y advertencias a implementar

1. **Un job a la vez:** la compilación bloquea la GPU. Si hay un modelo cargado en esa GPU,
   pedir confirmación para descargarlo antes de compilar.
2. **GPUs asimétricas:** si se eligen GPU 0 + GPU 1, mostrar aviso de que la velocidad
   de inferencia estará limitada por la 3060.
3. **Compatibilidad de versión:** el engine compilado solo funciona con la misma versión
   de TRT-LLM con la que fue compilado. Si se actualiza la imagen Docker, los engines
   existentes pueden quedar incompatibles. Mostrar la versión TRT-LLM usada en el historial.
4. **Tiempo de compilación:** avisar al usuario que puede tardar 15-60 minutos dependiendo
   del modelo y la GPU. La UI debe permanecer usable durante la compilación.
5. **Espacio en disco:** verificar espacio libre antes de iniciar (el engine puede ocupar
   tanto como los pesos originales o más).

---

## Dependencias previas

- [x] `TensorRTLLMBackend` implementado en `backend/ocabra/backends/tensorrt_llm_backend.py`
- [x] `tensorrt_llm_worker.py` implementado con soporte docker mode
- [x] `scripts/smoke_trtllm.py` disponible para validación post-compilación
- [ ] Imagen Docker `nvcr.io/nvidia/tensorrt-llm/release:latest` descargada en el host
- [ ] Smoke test ejecutado con el engine existente de TinyLlama (validación previa)

---

## Fases de implementación

### Fase 1 — Backend
1. Migración Alembic: tabla `trtllm_compile_jobs`
2. `CompileManager`: gestión de jobs, ejecución Docker en 2 fases, progreso Redis
3. Endpoints REST + SSE en `api/internal/trtllm.py`
4. Tests: mock Docker, ciclo completo convert→build→registro

### Fase 2 — Frontend
1. `CompileModal` con formulario, estimación VRAM y advertencias
2. Progreso en tiempo real via SSE
3. Integración en `Models.tsx` y `Explore.tsx`
4. Panel de historial de compilaciones (jobs pasados, estado, engine resultante)

### Fase 3 — Validación
1. Compilar `Qwen3.5-27B-GPTQ-Int4` en GPU 1 sola
2. Compilar un modelo pequeño en ambas GPUs (TP=2) para verificar ese flujo
3. Ejecutar smoke test sobre los engines compilados
4. Verificar registro automático y carga desde la UI
