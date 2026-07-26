# Qwen3.5 27B: ajuste de contexto en RTX 3090

Fecha: 2026-07-26

## Resultado

El perfil productivo inicial con KV BF16 era:

- `cyankiwi/Qwen3.5-27B-AWQ-4bit`, compressed-tensors AWQ int4 group-32.
- RTX 3090, TP=1 y solo el language model.
- Contexto 24.576 (`vllm-ctx24k`).
- KV BF16, `gpu_memory_utilization=0.96`.
- Compile, CUDA Graphs, prefix caching y chunked prefill activos.
- `max_num_seqs=16`, batch de 8.192 tokens.
- Parser de razonamiento `qwen3` y parser de herramientas `qwen3_coder`.

El modelo queda registrado como `Qwen3.5 27B — vllm-ctx24k`, con política
`on_demand`.

## Actualización: KV FP8 y contexto 64K

Se corrigió el JIT de FlashInfer y se validó el perfil de 65.536 tokens:

- KV cache FP8 E4M3 con backend `FLASHINFER`.
- `gpu_memory_utilization=0.98`, con 2,30 GiB de KV fijados mediante
  `--kv-cache-memory-bytes 2469606195`.
- Cálculo dinámico de escalas K/V activo para no degradar precisión usando
  las escalas 1.0 de fallback.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para evitar OOM por
  fragmentación durante prefills largos.
- `max_num_seqs=16` y batch de 8.192 tokens.

El fallo PTX procedía de una instalación parcialmente alineada: `nvcc` y
`ptxas` eran 13.0.88, pero sus dependencias sin pin `nvidia-nvvm` y
`nvidia-cuda-crt` habían subido a 13.3.73. NVVM generaba PTX 9.3 y `ptxas`
13.0 solo acepta hasta PTX 9.0. Se fijaron ambos paquetes a 13.0.88.

FlashInfer también presupone el layout tradicional
`CUDA_HOME/lib64/libcudart.so`; los wheels CUDA 13 proporcionan
`lib/libcudart.so.13`. El backend prepara automáticamente los alias
compatibles `lib64 -> lib` y `libcudart.so -> libcudart.so.13`.

Validación de inferencia real:

| Entrada | Salida | Tiempo | Resultado |
|---:|---:|---:|---|
| 10.014 tokens | 8 tokens | 8,49 s | Correcto |
| 60.014 tokens | 8 tokens | 53,52 s | Correcto |

Antes de activar `expandable_segments`, PyTorch mantenía 668,5 MiB reservados
pero no utilizables y fallaba al solicitar un bloque contiguo de 262 MiB.
Después del ajuste ambas entradas completaron sin OOM.

El perfil público registrado pasa a llamarse `qwen3.5:27b-vllm-ctx64k`.

## Selección del checkpoint

El checkpoint oficial `Qwen/Qwen3.5-27B-GPTQ-Int4` ocupa 30,3 GB y no cabe en
la RTX 3090. El AWQ seleccionado ocupa 20,06 GB en disco y permite mantener
pesos y KV completamente en GPU, sin offload.

El contexto nativo declarado por el modelo es 262.144 tokens.

## Búsqueda del límite

| Configuración | Resultado |
|---|---|
| 256K, KV BF16, reserva 0,98 | No cabe: requiere 16,32 GiB de KV y dispone de 2,35 GiB |
| Techo calculado, KV BF16, reserva 0,98 | 33.712 tokens |
| 32K, KV BF16, reserva 0,98 | Carga e inferencia correctas en vLLM directo |
| 64K, KV FP8, FlashInfer | Corregido y validado con una entrada real de 60.014 tokens |
| 64K, KV FP8, FlashAttention | Backend inválido: no soporta KV FP8 |
| 64K, KV FP8, Triton | Backend inválido: KV FP8 requiere SM89+; la RTX 3090 es SM86 |
| 24K, KV BF16, reserva 0,96 | Carga directa y carga completa mediante oCabra correctas |

32K es el máximo directo con la VRAM completa. Sin embargo, oCabra conserva
además un buffer global de 512 MiB para absorber memoria de CUDA, JIT y otros
procesos. Con esa protección, la mayor configuración redonda y estable es 24K.

Reducir el buffer global a 256 MiB permitiría aproximadamente reserva 0,97 y un
contexto práctico de 28–30K, pero no 32K. No se cambió porque el buffer protege
también backends que no tienen un límite propio equivalente a
`gpu_memory_utilization`.

## Rendimiento del perfil productivo

Mediana de tres ejecuciones, temperatura 0, semilla 42:

| Carga | Tiempo de pared | Throughput agregado |
|---|---:|---:|
| 1 × 64 tokens | 1,522 s | 42,04 tok/s |
| 4 × 64 tokens | 1,798 s | 142,35 tok/s |
| 8 × 64 tokens | 1,931 s | 265,21 tok/s |
| 12 × 64 tokens | 2,248 s | 341,66 tok/s |
| 1 × hasta 256 tokens | 5,956 s | 42,98 tok/s |

- Primera carga con compilación: 171,8 s.
- Carga con caché AOT poblada: 51,4 s.
- VRAM residente medida mediante oCabra: 22.291 MiB.
- Estimación conservadora registrada: 22.954 MiB.
