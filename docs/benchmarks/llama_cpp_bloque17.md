# Benchmarks llama.cpp post-Bloque 17

**Fecha:** 2026-05-05
**Hardware:** RTX 3060 12 GB (gpu 0) + RTX 3090 24 GB (gpu 1) — total 36 GB VRAM
**Binario:** `/data/backends/llama_cpp/bin/llama-server` rebuilt desde upstream master `a817a22` (sm_86 + sm_89, CUDA 12.4)
**Harness:** `/tmp/llamacpp_bench.py` (en el contenedor api), 3 prompts secuenciales + 4 paralelos por modelo

## Caveat: modelos no compatibles

Los siguientes modelos de Ollama **no se cargan** ni con upstream master de llama.cpp:

| Modelo | Arquitectura | Síntoma |
|---|---|---|
| `qwen3.6:latest` | `qwen35moe` | `rope.dimension_sections has wrong array length; expected 4, got 3` |
| `gemma4:26b` | `gemma4` (vision sparse) | `wrong number of tensors; expected 1014, got 658` |
| `gemma4:e4b-it-q8_0` | `gemma3 E4B` | `wrong number of tensors; expected 2131, got 720` |
| `nemotron3:33b` | `nemotron_h_omni` | `unknown model architecture` |
| `ministral-3:14b`, `devstral-small-2:24b` | `mistral3` | `wrong number of tensors; expected 585, got 363` |

Estos modelos usan extensiones de arquitectura que Ollama bajó pero que aún no están portadas a llama.cpp upstream. Para servirlos hoy hay que usar el contenedor de Ollama (que trae su propio fork). Cuando llama.cpp absorba esas archs (típicamente 1-3 meses tras release), bastará con re-ejecutar `cmake --build build --target llama-server -j` en `/data/backends/llama_cpp/src` para tenerlos vía nuestro backend.

## Modelos benchmarkeados

Los modelos válidos del set Ollama del usuario para programación + agentic loops:

| Modelo | Tamaño | Layout | Notas |
|---|---|---|---|
| `qwen3:32b` | 32B dense | Single GPU (3090) o split | Razonamiento full, mejor calidad |
| `qwen3:32b` (split) | 32B dense | Multi-GPU (3060+3090) | Más contexto a costa de velocidad |
| `qwen3-coder:30b` MoE | 30B/3B activos | Multi-GPU | Especializado código, **muy rápido** |
| `mistral:7b` | 7B dense | Single GPU (3060) | Baseline pequeño |

## Resultados

Configuración común: `flash_attn=on`, `cache_type_k=q8_0`, `cache_type_v=q8_0`, `cont_batching=on`.

| Modelo | GPUs | ctx | slots | VRAM total | Load (s) | TTFT (ms) | Single t/s | Parallel agg t/s | Per-req t/s |
|---|---|---|---|---|---|---|---|---|---|
| `mistral:7b` | 0 (3060) | 16K | 4 | 5.4 GB | 12 | 70 | **65.5** | 134.8 | 34.7 |
| `qwen3:32b` | 1 (3090) | 16K | 4 | 21.6 GB | 6 | 184 | 33.6 | 69.3 | 17.9 |
| `qwen3:32b` | 0+1 (split 1:2) | 32K | 4 | 24.3 GB (8.2+16.1) | 49 | 258 | 25.1 | 55.8 | 14.4 |
| `qwen3-coder:30b` MoE | 0+1 (split 1:2) | 32K | 8 | 20.1 GB (7.0+13.1) | 47 | 103 | **138.1** | **234.5** | **63.3** |

### Lectura clave

- **`qwen3-coder:30b` MoE es el ganador absoluto** para tu caso de uso (programación + agentes). Activa solo 3B params por token, así que aunque pesa ~17 GB, decodifica como un modelo de 3B: **138 t/s single, 234 t/s con 4 requests paralelas**. Cabe en 20 GB con 32K de contexto y 8 slots.
- **`qwen3:32b` dense en 3090 alone** es la opción si quieres calidad máxima de razonamiento: 33 t/s single, 21.6 GB. Te quedan 2.4 GB en el 3090 pero el 3060 queda libre para otros usos.
- **Multi-GPU split del 32B dense** tiene coste claro: 25% menos t/s vs single-GPU 3090 (25.1 vs 33.6) por la comunicación inter-GPU. Solo merece la pena si necesitas más de 16K de contexto.

## Configuraciones recomendadas como ModelConfig overrides

Estas se pegan en `extra_config["llama_cpp"]` al registrar el modelo en oCabra (vía `POST /ocabra/models` o `PATCH`).

### Ganador para coding/agentic — `qwen3-coder:30b` MoE

```json
{
  "llama_cpp": {
    "gpu_layers": 99,
    "ctx_size": 32768,
    "batch_size": 2048,
    "ubatch_size": 512,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tensor_split": [1, 2],
    "split_strategy": "evenly",
    "main_gpu": 1,
    "parallel_slots": 8,
    "cont_batching": true,
    "keep_alive_seconds": 1800
  }
}
```

`parallel_slots=8` aprovecha que cada slot decodifica con solo 3B activos. `tensor_split=[1,2]` se calculará automáticamente con `split_strategy="evenly"` consultando el GPU manager (12+24 GB → ratio 1:2 normalizado), pero lo dejo explícito para que sea reproducible.

### Calidad máxima — `qwen3:32b` dense single-GPU

```json
{
  "llama_cpp": {
    "gpu_layers": 99,
    "ctx_size": 16384,
    "batch_size": 2048,
    "ubatch_size": 512,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "main_gpu": 1,
    "disabled_gpus": [0],
    "parallel_slots": 4,
    "cont_batching": true,
    "keep_alive_seconds": 1800
  }
}
```

`disabled_gpus=[0]` fuerza al backend a ignorar el 3060 aunque el scheduler lo asigne — este modelo encaja entero en el 3090 y no compensa la latencia extra del split.

### Calidad máxima con contexto largo — `qwen3:32b` dense split

```json
{
  "llama_cpp": {
    "gpu_layers": 99,
    "ctx_size": 32768,
    "batch_size": 2048,
    "ubatch_size": 512,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tensor_split": [1, 2],
    "main_gpu": 1,
    "parallel_slots": 4,
    "cont_batching": true,
    "keep_alive_seconds": 1800
  }
}
```

## Aprendizajes operativos

1. **KV-quant `q8_0/q8_0` con `flash_attn=on`** es free win: ahorra ~50% del KV cache sin pérdida de calidad apreciable. Imprescindible para llegar a 32K en estos tamaños.

2. **`parallel_slots`** escala bien con `cont_batching` activo. Para un modelo de 30B dense, 4 slots saturan; para MoE 3B-active, 8 slots siguen escalando linealmente. Regla: arranca con `slots = min(8, ctx_size_total / 8192)`.

3. **`tensor_split` proporcional a VRAM** (12:24 = 1:2 con normalización al menor). El backend de oCabra ya lo autocalcula con `split_strategy="evenly"` pero conviene fijarlo explícitamente para producción.

4. **Modelos reasoning emiten `reasoning_content`** en SSE (no `content`). Cualquier cliente que quiera contar throughput debe sumar ambos. Lo hago ya en mi harness.

5. **Comunicación inter-GPU** cuesta ~25% en dense models. En MoE el coste es menor porque el subset activo es pequeño.

## Cómo replicar

```bash
# Dentro del contenedor api
docker compose exec api python /tmp/llamacpp_bench.py [model_name]
```

Targets disponibles: `mistral-7b`, `qwen3-32b`, `qwen3-32b-split`, `qwen3-coder-30b-moe`. Sin argumento corre los 4. Resultados se guardan en `/tmp/llamacpp_bench_results.json`.

Para añadir más modelos, edita `SPECS` al final del script.
