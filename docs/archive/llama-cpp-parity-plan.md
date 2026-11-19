# Plan: llama.cpp loader parity con LM Studio

**Bloque del roadmap:** 17 (4 sprints)
**Owner del esquema compartido:** Sprint 17.1
**Última actualización:** 2026-10-27 (nota de "superseded" sobre `llama_cpp_estimator.py`; el resto del documento sigue describiendo el plan original de 2026-05-05)

## Objetivo

Llevar la cobertura de controles de carga del backend `llama.cpp` en oCabra desde
el ~27% actual hasta paridad funcional con LM Studio (15+ flags expuestos en UI
per-model, estimador VRAM en vivo, multi-GPU granular, speculative decoding,
runtime alterno).

## Estado de partida

**Backend** — `backend/ocabra/backends/llama_cpp_backend.py` + `workers/llama_cpp_worker.py`
pasan 8 flags a `llama-server`: `n-gpu-layers`, `ctx-size`, `batch-size`,
`ubatch-size`, `threads`, `flash-attn`, `mlock`, `embedding`. Los overrides
per-model viven en `ModelState.extra_config["llama_cpp"]` y se leen vía
`_get_option()` (línea 292). Cuatro de ellos son globales en `config.py`
(`llama_cpp_*`) sin path UI per-model.

**Frontend** — `frontend/src/components/models/ModelConfigModal.tsx:1121-1176`
expone sólo 4 campos para llama.cpp: `gpu_layers`, `ctx_size`, `flash_attn`,
`embedding`. Los otros 4 flags soportados en backend están encerrados en config
global.

**API** — `POST /ocabra/models/{model_id}/load` no acepta body; lee de
`extra_config` persistido en BD. Endpoint `POST /ocabra/models/{model_id}/memory-estimate`
ya existe (api/internal/models.py:331-352) y acepta `extraConfig` para
estimación sin persistir; hoy es un probe genérico, no determinístico.

## Esquema compartido (`LlamaCppLoadConfig`)

**Crear en Sprint 17.1**, ampliar aditivamente en sprints siguientes. Vivirá
en `backend/ocabra/schemas/backend_load.py` y se serializará bajo
`extra_config["llama_cpp"]` con `model_dump(exclude_none=True)`. Todos los
campos son opcionales — `None` significa "usar default global / del binario".

```python
from pydantic import BaseModel, Field
from typing import Literal

KvCacheType = Literal["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]

# --- Sprint 17.1 (Tier 1) ---
class LlamaCppLoadConfig(BaseModel):
    # ya soportados pre-sprint (mantener nombres)
    gpu_layers: int | None = None
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    flash_attn: bool | None = None
    mlock: bool | None = None
    embedding: bool | None = None
    # nuevos en 17.1
    mmap: bool | None = None  # default True; False emite --no-mmap
    seed: int | None = None
    no_kv_offload: bool | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

# --- Sprint 17.2 ---
    cache_type_k: KvCacheType | None = None
    cache_type_v: KvCacheType | None = None  # requiere flash_attn=True si != f16

# --- Sprint 17.3 ---
    main_gpu: int | None = None
    tensor_split: list[float] | None = None
    split_mode: Literal["layer", "row", "none"] | None = None
    disabled_gpus: list[int] | None = None
    split_strategy: Literal["evenly", "favor_main"] | None = None
    n_cpu_moe: int | None = None
    override_tensor: str | None = None

# --- Sprint 17.4 ---
    speculative: SpeculativeConfig | None = None
    runtime: Literal["cuda", "rocm", "vulkan", "cpu"] | None = None
    parallel_slots: int | None = None
    cont_batching: bool | None = None
    keep_alive_seconds: int | None = None

class SpeculativeConfig(BaseModel):
    draft_model_id: str
    draft_n: int | None = None
    draft_min: int | None = None
    draft_p_min: float | None = None
```

**Regla de coordinación**: Sprints 17.2-17.4 sólo AÑADEN campos al schema,
nunca renombran ni borran. El merge será una unión de campos pydantic, conflicto
trivial.

## Mapeo schema → flags llama-server

| Schema | Flag llama-server | Sprint |
|---|---|---|
| `gpu_layers` | `--n-gpu-layers N` | base |
| `ctx_size` | `--ctx-size N` | base |
| `batch_size` | `--batch-size N` | base |
| `ubatch_size` | `--ubatch-size N` | base |
| `threads` | `--threads N` | base |
| `flash_attn=True` | `--flash-attn on` | base |
| `mlock=True` | `--load-mode mmap+mlock` (ver nota 2026-10-25 abajo) | base |
| `embedding=True` | `--embedding` | base |
| `mmap=False` | `--load-mode none` (ver nota 2026-10-25 abajo) | 17.1 |
| `seed` | `--seed N` | 17.1 |
| `no_kv_offload=True` | `--no-kv-offload` | 17.1 |
| `rope_freq_base` | `--rope-freq-base F` | 17.1 |
| `rope_freq_scale` | `--rope-freq-scale F` | 17.1 |
| `cache_type_k` | `--cache-type-k STR` | 17.2 |
| `cache_type_v` | `--cache-type-v STR` | 17.2 |
| `main_gpu` | `--main-gpu N` | 17.3 |
| `tensor_split` | `--tensor-split a,b,c` | 17.3 |
| `split_mode` | `--split-mode STR` | 17.3 |
| `disabled_gpus` | composición `CUDA_VISIBLE_DEVICES` | 17.3 |
| `n_cpu_moe` | `--n-cpu-moe N` | 17.3 |
| `override_tensor` | `--override-tensor STR` | 17.3 |
| `speculative.draft_model_id` | `--model-draft PATH` | 17.4 |
| `speculative.draft_n` | `--spec-draft-n-max N` (renombrado, ver nota abajo) | 17.4 |
| `speculative.draft_min` | `--spec-draft-n-min N` (renombrado, ver nota abajo) | 17.4 |
| `speculative.draft_p_min` | `--spec-draft-p-min F` | 17.4 |
| `speculative.spec_type` | `--spec-type STR` (p.ej. `draft-mtp`) | 17.4 |
| `parallel_slots` | `--parallel N` | 17.4 |
| `cont_batching=True` | `--cont-batching` | 17.4 |
| `mmproj_model_id` | `--mmproj PATH` (visión; añadido fuera de los 4 sprints originales, ver nota 2026-10-27 abajo) | — |

**Nota 2026-10-25 — modernización del backend + hallazgos de CLI:**
El backend `llama_cpp` se reconstruyó desde `ggml-org/llama.cpp` master
(el build anterior era de abril 2026). Dos cambios de CLI reales entre esa
versión y la actual, ya corregidos en código (`llama_cpp_worker.py`):
- `--draft-max`/`--draft-min` → renombrados a `--spec-draft-n-max`/
  `--spec-draft-n-min` (ya corregido antes de hoy).
- `--mlock`/`--no-mmap` (flags booleanos sueltos) → unificados en un único
  `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`. Pasar el flag viejo
  ahora aborta con `error: invalid argument: --mlock`. `llama_cpp_worker.py`
  traduce internamente; la interfaz `extra_config.llama_cpp.{mlock,mmap}`
  no cambia de cara a quien registra el modelo.

**Self-contained MTP — antes bloqueado, ahora funciona.** El build de abril
rechazaba `--spec-type draft-mtp` sin `--model-draft` con
`"unknown speculative decoding type without draft model"` incluso con los
tensores `nextn` embebidos en el GGUF (ver `gpu-bench/README.md`). Con el
build de hoy, el mismo GGUF (`qwen3.8:27b`, blob de Ollama) carga y sirve
inferencia real con MTP self-speculative activo — confirmado vía
`draft_n`/`draft_n_accepted` en la respuesta de `/v1/chat/completions`
(26/36 tokens de borrador aceptados). Probado solo en CPU (sin contender
por la GPU de producción); pendiente validar rendimiento real en GPU.

**Nota 2026-10-27 — soporte de visión (`mmproj`), fuera del plan original.**
Ningún sprint de este plan cubría multimodal; se añadió end-to-end
(schema, `_build_options()`, worker, heurística de `BackendCapabilities.vision`)
al migrar `gemma4:26b` a este backend, que necesita su projector `gemma4v`
para servir imágenes. Mismo patrón de resolución que
`speculative.draft_model_id` (ruta directa o model_id canónico vía
`_resolve_mmproj_path()`). Detalle completo en `gpu-bench/README.md`.

## Sprints

### Sprint 17.1 — Tier 1 (foundation + flags triviales)

**Branch sugerida:** `feat/llama-cpp-tier1`

**Scope backend:**
- Crear `backend/ocabra/schemas/backend_load.py` con `LlamaCppLoadConfig` (sólo
  campos del Tier 1 — ver schema arriba).
- Ampliar `_build_options()` en `llama_cpp_backend.py:257-290` para incluir los
  5 flags nuevos: `mmap`, `seed`, `no_kv_offload`, `rope_freq_base`,
  `rope_freq_scale`.
- Ampliar `llama_cpp_worker.py` (líneas 15-23) con los args correspondientes.
- Mantener compatibilidad: si el campo es `None` no se pasa el flag.
- `mmap` semántica: default `True` (no flag); `False` → `--no-mmap`.

**Scope frontend:**
- En `ModelConfigModal.tsx:1121-1176` añadir inputs:
  - `threads` (number, placeholder "auto")
  - `batch_size` y `ubatch_size` (number)
  - `mlock` (checkbox)
  - `mmap` (checkbox, default checked)
  - `seed` (number, placeholder "random")
  - `no_kv_offload` (checkbox)
  - `rope_freq_base` y `rope_freq_scale` (number, step 0.1)
- Agruparlos en una sub-sección "Avanzado" colapsable para no saturar la UI
  básica.

**Tests:**
- Ampliar `backend/tests/test_llama_cpp_backend.py`: cada flag nuevo en su test
  unitario verificando que aparece/desaparece en el comando construido.

**Definition of done:**
- 5 flags nuevos navegables backend → llama-server.
- 4 flags ya soportados (threads, batch, ubatch, mlock) editables per-model en UI.
- Tests verdes.

---

### Sprint 17.2 — KV-quant + estimador VRAM determinístico

**Branch sugerida:** `feat/llama-cpp-kvquant-vram-estim`
**Asume:** Sprint 17.1 mergeado o se merge con conflicto trivial sobre el schema.

**Scope backend:**
- Añadir `cache_type_k` y `cache_type_v` al schema `LlamaCppLoadConfig` con
  enum `KvCacheType`.
- Validador pydantic: si `cache_type_v` ≠ `f16`, exigir `flash_attn=True`
  (mismo gating que LM Studio). Mensaje de error explícito.
- Cablear flags `--cache-type-k` y `--cache-type-v` en `_build_options()` y worker.
- Crear `backend/ocabra/core/llama_cpp_estimator.py`:
  - Función `estimate_vram(gguf_path, config: LlamaCppLoadConfig) -> dict`
    que devuelva `{model_bytes, kv_bytes, compute_buffer_bytes, total_bytes}`.
  - Parsear header GGUF (struct mágico `GGUF` + clave-valor metadata) sin cargar
    pesos. Extraer `n_layers`, `n_kv_heads`, `head_dim`, dtype del modelo.
  - Fórmula KV-cache: `n_layers × 2 × n_kv_heads × head_dim × n_ctx ×
    bytes_per_element(cache_type)`.
  - Buffer de cómputo: aproximar como `batch_size × hidden_dim × 4` (suficiente
    para UI; no busca exactitud al byte).
- Ampliar endpoint `POST /ocabra/models/{model_id}/memory-estimate`
  (api/internal/models.py:331) para usar el nuevo estimador cuando el backend
  sea llama.cpp; mantener el probe genérico actual como fallback.

  > **Superseded (2026-10-27):** este módulo se construyó tal cual el plan,
  > pero quedó como un segundo estimador desincronizado de `vram_planner.py`
  > (sin arquitecturas híbridas SSM/SWA, sin rs_cache, sin overhead de draft
  > MTP) — causa directa de que los números de "Memoria y contexto" no
  > cuadraran entre paneles. Borrado y `_estimate_llama_cpp_memory()`
  > reescrito sobre `vram_planner.py` (`arch_from_gguf` +
  > `plan_llama_cpp_vram_mb` + `max_context_tokens`), que ya es la única
  > fuente de verdad para VRAM de llama.cpp en todo el backend. Detalle en
  > `gpu-bench/README.md` y memoria del agente.

**Scope frontend:**
- En `ModelConfigModal.tsx` añadir dos selects (`cache_type_k`, `cache_type_v`)
  con tooltip explicando el ahorro de VRAM. Disable `cache_type_v != f16` si
  `flash_attn` no está marcado, con mensaje.
- Panel "Estimación VRAM" que llame a `/memory-estimate` con debounce 300ms al
  cambiar cualquier campo (ctx_size, gpu_layers, cache_type_k/v). Mostrar
  desglose model / KV / compute en la sidebar del modal.

**Tests:**
- `test_llama_cpp_estimator.py`: parsing de un GGUF de ejemplo (se puede usar
  `qwen2.5-0.5b-gguf` ya validado en bloque 15) y verificación de fórmula con
  varios `cache_type_k/v` y `n_ctx`.
- Test E2E del endpoint `/memory-estimate` con backend llama.cpp.

**Definition of done:**
- KV-quant ahorra VRAM medida en estimador.
- UI refleja la estimación en vivo al cambiar sliders.
- Validación cache_type_v ≠ f16 ⟹ flash_attn enforced.

---

### Sprint 17.3 — Multi-GPU granular + MoE CPU offload

**Branch sugerida:** `feat/llama-cpp-multigpu-moe`
**Asume:** Sprint 17.1 mergeado.

**Scope backend:**
- Ampliar `LlamaCppLoadConfig` con: `main_gpu`, `tensor_split`, `split_mode`,
  `disabled_gpus`, `split_strategy`, `n_cpu_moe`, `override_tensor`.
- Cablear flags en `_build_options()` y worker. `tensor_split` se serializa como
  CSV: `--tensor-split 3,1`.
- Refactor de la sección de `CUDA_VISIBLE_DEVICES` en
  `llama_cpp_backend.py:141-144` para componer también con `disabled_gpus`.
- Cuando `split_strategy="evenly"` y no hay `tensor_split`, autocalcular ratios
  proporcionales al VRAM total de cada GPU (consultar `gpu_manager`).

**Scope frontend:**
- Sub-sección "Multi-GPU" en el modal con:
  - Lista drag-and-drop de GPUs detectadas (consume `/ocabra/gpus`).
  - Toggle enable/disable per-GPU → `disabled_gpus`.
  - Radio: "Even split" / "Custom ratios" / "Single GPU".
  - Si custom: inputs numéricos por GPU para `tensor_split` ratios.
  - Select `main_gpu` (filtrado por GPUs habilitadas).
  - Select `split_mode` (layer/row/none).
- Sub-sección "MoE": slider `n_cpu_moe` (0 a `n_layers`), input `override_tensor`
  (avanzado, free-text con tooltip).

**Tests:**
- `test_llama_cpp_backend.py`: cada flag multi-GPU nuevo, composición de
  `CUDA_VISIBLE_DEVICES` con `disabled_gpus`.
- Test del cálculo automático de ratios "evenly" con mock GPU manager.

**Definition of done:**
- Carga multi-GPU configurable per-model con tres estrategias.
- MoE CPU offload aplicable.

---

### Sprint 17.4 — Speculative decoding + runtime alterno + concurrent slots

**Branch sugerida:** `feat/llama-cpp-speculative-runtime`
**Asume:** Sprint 17.1 mergeado.

**Scope backend:**
- Ampliar `LlamaCppLoadConfig` con: `speculative: SpeculativeConfig | None`,
  `runtime`, `parallel_slots`, `cont_batching`, `keep_alive_seconds`.
- Speculative: cuando `speculative.draft_model_id` está definido, resolver el
  path local del modelo borrador (consultar registry/local_scanner) y pasar
  `--model-draft PATH` + flags `--draft-*`.
- Validador: el draft model debe tener mismo `vocab_size` y mismos token IDs
  bos/eos que el modelo principal. Implementar parsing de tokenizer en
  `registry/local_scanner.py` para indexar `vocab_size`, `bos_id`, `eos_id`
  por modelo.
- Endpoint nuevo: `GET /ocabra/models/{id}/speculative-candidates` que devuelva
  modelos compatibles con el `id` dado.
- Runtime selection: añadir `BackendInstallSpec` para variantes
  `llama_cpp:cuda`, `llama_cpp:rocm`, `llama_cpp:vulkan`, `llama_cpp:cpu`;
  cuando `runtime` está definido en el load config, escoger el binario
  correspondiente en `_get_binary_path()` (puede ser nuevo método). Default
  actual = `cuda`.
- `parallel_slots` y `cont_batching` → flags `--parallel N` y `--cont-batching`.
- `keep_alive_seconds`: integrarlo en `model_manager.py` como override sobre
  `idle_eviction_seconds` global.

**Scope frontend:**
- Sub-sección "Speculative decoding": select de draft model (filtrado por
  `/speculative-candidates`), inputs `draft_n` / `draft_min` / `draft_p_min`.
- Sub-sección "Runtime": select cuda/rocm/vulkan/cpu (mostrar sólo runtimes
  instalados consultando `/ocabra/backends`).
- Sub-sección "Concurrencia": `parallel_slots` (number), `cont_batching`
  (checkbox, default true), `keep_alive_seconds` (number, placeholder "global").

**Tests:**
- `test_llama_cpp_backend.py`: speculative flags, runtime path resolution.
- Test de validación de compatibilidad de tokenizer entre dos GGUFs.
- Test del endpoint `/speculative-candidates`.

**Definition of done:**
- Speculative decoding cargable en UI con validación.
- Runtime alterno seleccionable (al menos cuda/cpu reales; rocm/vulkan pueden
  quedarse como entrada lista sin binarios si no hay hardware de prueba).
- TTL per-model funciona.

## Reglas de coordinación entre sprints

1. **Schema additive only**: Sprints 17.2-17.4 sólo añaden campos. Si dos
   sprints añaden el mismo campo (no debería pasar dado el reparto), prevalece
   el de Sprint inferior.
2. **Worker args additive only**: cada sprint añade sus flags al final de la
   lista de args en `llama_cpp_worker.py`. Sin reordenar los existentes.
3. **UI: subsecciones nuevas**: cada sprint añade su propia sub-sección en el
   modal en lugar de mezclarse con las existentes. Esto minimiza conflictos.
4. **Tests**: cada sprint añade tests en bloques claramente delimitados con
   comentarios `# --- Sprint 17.X ---`.
5. **Commits**: prefijo `feat(llama-cpp):` y mencionar el sprint en el cuerpo
   del mensaje.
6. **No tocar otros backends**: vLLM, SGLang, TRT-LLM, Ollama, etc. fuera de
   alcance.
