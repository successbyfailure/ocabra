# Benchmark vLLM 0.26: Gemma 4, Qwen3.6 y Qwen3-Coder

Fecha: 2026-07-26

## Resumen

La RTX 3090 sola es la configuración recomendada para los tres modelos:

- vLLM supera a Ollama en generación individual entre **1,98× y 3,08×**.
- Con 12 peticiones concurrentes, vLLM alcanza entre **9,78× y 13,46×** el
  throughput agregado de Ollama.
- Añadir la RTX 3060 no acelera ningún modelo. Gemma 4 y Qwen3-Coder pierden
  rendimiento y contexto por la sincronización con la GPU lenta. Qwen3.6 no
  puede inicializar una configuración distribuida viable en la 3060.
- Qwen3.6 incluye una capa MTP, pero necesita unos 970 MiB adicionales. Con
  offload a CPU funciona, aunque queda mucho más lento que la ejecución base.

## Protocolo

- vLLM 0.26.0, PyTorch 2.11.0+cu130, compile y CUDA Graphs activos.
- Prefix caching y chunked prefill activos; sampler FlashInfer desactivado.
- Temperatura 0, semilla 42.
- Peticiones cortas de 64 tokens con concurrencia 1, 4, 8 y 12.
- Petición larga de hasta 256 tokens; prompt de código para Qwen3-Coder.
- Cada cifra es la mediana de tres ejecuciones después de un warm-up.
- Throughput concurrente = suma de tokens generados / tiempo de pared del lote.
- Ollama utiliza `OLLAMA_NUM_PARALLEL=1`, por lo que encola solicitudes.

El benchmark reproducible está en
`scripts/benchmark_vllm_concurrency.py`. El script fija explícitamente
`enforce_eager=false` para que los valores globales antiguos del contenedor no
desactiven accidentalmente compile y CUDA Graphs.

## Checkpoints

| Modelo | vLLM | Cuantización | Tamaño |
|---|---|---|---:|
| Gemma 4 26B-A4B | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | AWQ 4-bit, group 32 | 17,19 GB |
| Gemma 4 12B | `mattbucci/gemma-4-12B-AWQ` | AWQ 4-bit, group 128 | 7,80 GB |
| Qwen3.6 35B-A3B | `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | GPTQ Int4, group 128 | 24,42 GB |
| Qwen3-Coder 30B-A3B | `stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ` | AWQ 4-bit, group 128 | 16,69 GB |

Los modelos de Ollama son `gemma4:26b`, `qwen3.6:latest` y
`qwen3-coder:30b`. No es una comparación bit a bit: Ollama usa checkpoints
GGUF y cuantizadores distintos, y puede repartirlos automáticamente entre
ambas GPU. La comparación representa las opciones desplegables reales en este
servidor.

## vLLM en RTX 3090 frente a Ollama

### Throughput agregado

| Modelo | Carga | vLLM RTX 3090 | Ollama | Ventaja vLLM |
|---|---:|---:|---:|---:|
| Gemma 4 | 1 × 64 | 135,62 tok/s | 43,96 tok/s | 3,08× |
| Gemma 4 | 4 × 64 | 428,89 tok/s | 65,80 tok/s | 6,52× |
| Gemma 4 | 8 × 64 | 777,15 tok/s | 71,64 tok/s | 10,85× |
| Gemma 4 | 12 × 64 | 984,81 tok/s | 73,19 tok/s | 13,46× |
| Gemma 4 | 1 × hasta 256 | 137,61 tok/s | 66,44 tok/s | 2,07× |
| Qwen3.6 | 1 × 64 | 108,03 tok/s | 54,68 tok/s | 1,98× |
| Qwen3.6 | 4 × 64 | 325,81 tok/s | 70,55 tok/s | 4,62× |
| Qwen3.6 | 8 × 64 | 605,82 tok/s | 75,14 tok/s | 8,06× |
| Qwen3.6 | 12 × 64 | 743,19 tok/s | 75,98 tok/s | 9,78× |
| Qwen3.6 | 1 × hasta 256 | 120,55 tok/s | 80,23 tok/s | 1,50× |
| Qwen3-Coder | 1 × 64 | 173,54 tok/s | 80,19 tok/s | 2,16× |
| Qwen3-Coder | 4 × 64 | 574,84 tok/s | 103,52 tok/s | 5,55× |
| Qwen3-Coder | 8 × 64 | 955,73 tok/s | 110,16 tok/s | 8,68× |
| Qwen3-Coder | 12 × 64 | 1.104,88 tok/s | 111,45 tok/s | 9,91× |
| Qwen3-Coder | 1 × hasta 256 | 178,03 tok/s | 104,01 tok/s | 1,71× |

La diferencia crece con la concurrencia: vLLM incorpora solicitudes nuevas al
batch continuo, mientras Ollama procesa esencialmente una detrás de otra con
la configuración actual.

### Carga y memoria

| Modelo | Contexto vLLM | VRAM RTX 3090 | Carga vLLM | Carga Ollama |
|---|---:|---:|---:|---:|
| Gemma 4 | 65.536 | 23.037 MiB | 161,83 s | no medida; ya estaba warm |
| Qwen3.6 | 32.768 | 23.481 MiB | 91,55 s con caché | 23,36 s |
| Qwen3-Coder | 65.536 | 23.821 MiB | 71,80 s con caché | 46,65 s |

El primer arranque que pobló la caché de Qwen3.6 tardó 262,42 s. vLLM paga
más carga inicial por compilación y captura de grafos, pero la política `warm`
permite amortizarla manteniendo el worker residente hasta que haya presión de
VRAM.

Gemma 4 es además una mejora funcional de vLLM 0.26: el mismo checkpoint no
podía cargarse con el runtime 0.19.1 y ahora completa inferencia.

## Una GPU frente a dos GPU

Gemma admite tensor parallel 2 a 32K. Qwen3-Coder no admite TP=2 con este AWQ,
pero sí pipeline parallel 2 a 32K. Los intentos de 64K no caben en la RTX 3060:

- Gemma necesitaba 1,90 GiB de KV y solo disponía de 1,81 GiB.
- Qwen3-Coder necesitaba 3,00 GiB de KV y disponía de 2,19 GiB.

| Modelo | Carga | RTX 3090 | 3060 + 3090 | Ventaja 3090 |
|---|---:|---:|---:|---:|
| Gemma 4 | 1 × 64 | 135,62 tok/s | 103,71 tok/s | 1,31× |
| Gemma 4 | 4 × 64 | 428,89 tok/s | 288,37 tok/s | 1,49× |
| Gemma 4 | 8 × 64 | 777,15 tok/s | 529,17 tok/s | 1,47× |
| Gemma 4 | 12 × 64 | 984,81 tok/s | 625,56 tok/s | 1,57× |
| Gemma 4 | larga | 137,61 tok/s | 105,08 tok/s | 1,31× |
| Qwen3-Coder | 1 × 64 | 173,54 tok/s | 142,39 tok/s | 1,22× |
| Qwen3-Coder | 4 × 64 | 574,84 tok/s | 409,92 tok/s | 1,40× |
| Qwen3-Coder | 8 × 64 | 955,73 tok/s | 798,33 tok/s | 1,20× |
| Qwen3-Coder | 12 × 64 | 1.104,88 tok/s | 932,87 tok/s | 1,18× |
| Qwen3-Coder | larga | 178,03 tok/s | 147,52 tok/s | 1,21× |

La carga también empeora: Gemma tarda 201,94 s en TP=2 y Coder 131,76 s en
PP=2. La 3090 sola tarda 161,83 s y 71,80 s, respectivamente.

Qwen3.6 falló tanto con TP=2 a 32K como con PP=2 reducido hasta 8K. Un worker
muere durante la carga en la RTX 3060; el checkpoint ya utiliza 23,48 GiB en
la 3090 y no deja un reparto práctico en una GPU de 12 GiB. No existe por
tanto una cifra de dos GPU válida para este checkpoint y hardware.

## Qwen3.6 con MTP speculative

vLLM detecta `mtp_num_hidden_layers=1` y soporta el método `qwen3_5_mtp`.
Sin offload, la carga falla al reservar una cabeza de 970 MiB: quedaban solo
235 MiB libres en la 3090. Se validó con 2 GiB de offload a CPU:

| Carga | Base | MTP + offload | Ventaja base |
|---|---:|---:|---:|
| 1 × 64 | 108,03 tok/s | 56,48 tok/s | 1,91× |
| 4 × 64 | 325,81 tok/s | 179,06 tok/s | 1,82× |
| 8 × 64 | 605,82 tok/s | 332,97 tok/s | 1,82× |
| 12 × 64 | 743,19 tok/s | 261,52 tok/s | 2,84× |
| 1 × hasta 256 | 120,55 tok/s | 54,69 tok/s | 2,20× |
| Carga | 91,55 s | 292,43 s | 3,19× más rápida |

El tráfico PCIe del offload elimina cualquier ganancia especulativa. No debe
activarse MTP en esta máquina. Merece repetirse sin offload en una GPU con al
menos unos 25 GiB utilizables o con un checkpoint más pequeño.

## Gemma 4 12B

Gemma 4 12B admite en la RTX 3090 su contexto nativo completo de 262.144
tokens con TP=1, `max_num_seqs=32`, batch 8.192 y reserva VRAM 0,94.

| Carga | vLLM RTX 3090 | Ollama RTX 3090 | Ventaja vLLM |
|---|---:|---:|---:|
| 1 × 64 | 74,13 tok/s | 39,45 tok/s | 1,88× |
| 4 × 64 | 271,38 tok/s | 52,92 tok/s | 5,13× |
| 8 × 64 | 491,59 tok/s | 56,62 tok/s | 8,68× |
| 12 × 64 | 646,81 tok/s | 57,53 tok/s | 11,24× |
| 1 × hasta 256 | 75,05 tok/s | 53,95 tok/s | 1,39× |

vLLM tardó 161,89 s en el primer arranque y 81,45 s reutilizando la caché de
compilación. Con 256K reservados utiliza 22.443 MiB de la RTX 3090. Ollama
tardó 23,67 s en frío y utilizó 10.949 MiB en la misma GPU. La diferencia de
VRAM procede principalmente de que vLLM dedica la memoria restante a su pool
KV para batching, no de los pesos del modelo, cuyo estimador es 8.906 MiB.

El modelo queda registrado como `Gemma 4 12B — vllm-ctx256k`, con política
`warm`: carga en la primera petición y permanece residente hasta que la presión
de VRAM requiera desalojarlo.

## Recomendación operativa

1. Ejecutar los tres modelos en la RTX 3090 con TP=1.
2. Mantener compile, CUDA Graphs, prefix caching y chunked prefill activos.
3. Usar 65K para Gemma 4 y Qwen3-Coder; limitar Qwen3.6 a 32K.
4. No activar MTP/offload para Qwen3.6.
5. Reservar la RTX 3060 para otro worker independiente. Solo usar TP/PP cuando
   un modelo no quepa en la 3090, aceptando menor contexto y throughput.

## Actualización KV FP8

Una ronda posterior amplió Qwen3.6 a 65.536 tokens y Qwen3-Coder a 114.688
tokens mediante KV FP8 sobre FlashInfer. Gemma 4 conserva KV BF16 porque sus
kernels de atención no admiten FP8 en la RTX 3090 SM86. Véase
`docs/benchmarks/vllm_026_fp8_kv_expansion.md`.
