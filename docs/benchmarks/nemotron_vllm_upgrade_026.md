# Benchmark: Nemotron 3 Nano — actualización vLLM 0.19.1 → 0.26.0

Fecha: 2026-07-26

## Resultado

La configuración ganadora es vLLM 0.26.0 en la RTX 3090, sin
`--enforce-eager` y sin speculative decoding:

- 211,9 tokens/s en generación individual sostenida, frente a 22,55 tokens/s
  con 0.19.1: **9,40×**.
- 983,25 tokens/s agregados con 12 peticiones, frente a 319,78: **3,07×**.
- La 3090 sola supera el pipeline 3060 + 3090 porque la GPU lenta introduce
  burbujas y limita todas las etapas.

## Entornos

| Componente | Antes | Actualizado |
|---|---:|---:|
| vLLM | 0.19.1 | 0.26.0 |
| PyTorch | 2.10.0+cu128 | 2.11.0+cu130 |
| Transformers | 5.7.0 | 5.14.1 |
| Ejecución | eager | compile + CUDA Graphs |
| Sampler | nativo | nativo |

Checkpoint en todas las pruebas:
`stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ`, contexto 65.536,
temperatura 0 y semilla 42.

Para una GPU se utilizó la RTX 3090 con `max_num_seqs=32`,
`max_num_batched_tokens=8192` y reserva de VRAM 0,94. Para dos GPU se empleó
pipeline parallel 2, no tensor parallel: el AWQ tiene dimensiones que el kernel
Marlin no puede dividir entre dos ranks TP. El pipeline utilizó
`max_num_seqs=12`, batch 6.144 y reserva 0,97 para que la RTX 3060 pudiera
mantener el contexto completo.

Los valores actualizados son la mediana de tres ejecuciones después de un
warm-up. Los valores anteriores proceden de la sesión de benchmark 0.19.1.

## Throughput

### RTX 3090

| Carga | 0.19.1 | 0.26.0 | Mejora |
|---|---:|---:|---:|
| 1 × 64 tokens | 23,26 tok/s | 200,13 tok/s | 8,60× |
| 4 × 64 tokens | 103,65 tok/s | 497,91 tok/s | 4,80× |
| 8 × 64 tokens | 214,02 tok/s | 887,75 tok/s | 4,15× |
| 12 × 64 tokens | 319,78 tok/s | 983,25 tok/s | 3,07× |
| 1 × hasta 256 tokens | 22,55 tok/s | 211,90 tok/s | 9,40× |

### RTX 3060 + RTX 3090, pipeline parallel

| Carga | 0.19.1 | 0.26.0 | Mejora |
|---|---:|---:|---:|
| 1 × 64 tokens | 18,91 tok/s | 142,31 tok/s | 7,53× |
| 4 × 64 tokens | 127,07 tok/s | 326,38 tok/s | 2,57× |
| 8 × 64 tokens | 319,99 tok/s | 595,06 tok/s | 1,86× |
| 12 × 64 tokens | 475,42 tok/s | 632,22 tok/s | 1,33× |
| 1 × hasta 256 tokens | 42,76 tok/s | 160,19 tok/s | 3,75× |

### Configuración actualizada: una frente a dos GPU

| Carga | RTX 3090 | Pipeline 3060 + 3090 | Ventaja 3090 |
|---|---:|---:|---:|
| 1 × 64 tokens | 200,13 tok/s | 142,31 tok/s | 1,41× |
| 4 × 64 tokens | 497,91 tok/s | 326,38 tok/s | 1,53× |
| 8 × 64 tokens | 887,75 tok/s | 595,06 tok/s | 1,49× |
| 12 × 64 tokens | 983,25 tok/s | 632,22 tok/s | 1,56× |
| 1 × hasta 256 tokens | 211,90 tok/s | 160,19 tok/s | 1,32× |

## Carga en frío

| Entorno | 0.19.1 | 0.26.0 |
|---|---:|---:|
| RTX 3090, sesión de benchmark | 41–93 s | ~118 s |
| RTX 3090, API recién recreada | — | 181,8 s |
| RTX 3090, caché persistente poblada | — | 121,6 s |
| Pipeline 3060 + 3090 | ~64 s | ~111 s |

En 0.26.0 los pesos tardaron unos 11 s en una GPU y 6 s por etapa en pipeline.
El resto del arranque lo dominan `torch.compile`, captura de CUDA Graphs y un
warm-up de kernels Mamba de unos 57–58 s. Se acepta el arranque más lento porque
la política `on_demand` mantiene el worker residente tras la primera petición
hasta que sea necesario desalojarlo, y la ganancia sostenida es muy superior.
La caché AOT redujo la carga productiva de 161,8 s al poblarla a 121,6 s al
reutilizarla (24,8%). Frente al primer arranque de un contenedor limpio,
181,8 s, la reducción fue del 33,1%. El warm-up Mamba sigue siendo costoso.
oCabra fija `VLLM_CACHE_ROOT=/data/backends/vllm/cache` para conservar grafos
AOT y planes de arranque aunque se recree el contenedor API.

## Speculative decoding

El checkpoint Nano AWQ no contiene capas ni pesos MTP
(`mtp_num_layers`/`num_nextn_predict_layers` no existen), por lo que el método
`nemotron_h_mtp` no es aplicable.

Se validó `ngram_gpu` con cinco tokens especulativos y lookup máximo 4. El
servidor carga, pero la primera inferencia termina el engine con una aserción en
el estado Mamba (`mamba_mixer2.py`, estado del último token planificado). No se
activa en producción. El backend acepta `speculative_config` para modelos
compatibles, y esta combinación debe revalidarse cuando vLLM corrija el soporte
speculative de modelos híbridos o exista un checkpoint Nano con MTP real.

## Configuración desplegada

- Runtime activo: vLLM 0.26.0; el venv 0.19.1 se conserva como rollback.
- RTX 3090, TP=1, contexto 262.144 con KV FP8 sobre FlashInfer.
- Compile y CUDA Graphs activos (`enforce_eager=false`).
- Prefix caching y chunked prefill activos.
- `max_num_seqs=16`, batch de 8.192 tokens y KV fijada en 4,1 GiB.
- Sampler FlashInfer desactivado: sus paquetes CUDA 13 pueden intentar JIT; el
  sampler nativo fue estable y rápido.
- Caché de compilación y startup persistente en `/data/backends/vllm/cache`.
- `nvidia-cuda-nvcc` fijado a 13.0.88 para coincidir con las cabeceras CUDA 13.0
  de PyTorch/vLLM.
