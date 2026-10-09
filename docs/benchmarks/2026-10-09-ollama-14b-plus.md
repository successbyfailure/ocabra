# Benchmark 2026-10-09 — modelos Ollama ≥14 B

Benchmark de los modelos registrados en oCabra con ≥14 B parámetros,
servidos por el backend **Ollama** (contexto por defecto, un slot por
modelo según el perfil activo).

## Entorno

| Pieza | Valor |
|---|---|
| Host | 2× NVMe (ZFS stripe, pool 67 % CAP) |
| GPUs | GPU 0 RTX 3060 12 GB · GPU 1 RTX 3090 24 GB |
| Backend | Ollama 0.35.1, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=2` |
| Transporte | Peticiones vía oCabra `/v1/chat/completions` con API key, 128 tokens Phase 1 / 80 tokens Phase 2 |
| Prompt | `"Explain quantum entanglement in one paragraph. Be concise, no more than 150 words."` |
| Fragmentación disco | FRAG 49 %, CAP 67 % (tras cleanup previo + rewrite de blobs calientes) |

Observación importante sobre tiempos: todos los modelos ≥20 GB **se splittean
entre 3060 + 3090**, lo que añade overhead PCIe por capa y limita t/s. El
GGUF warm reload mide "page cache hit" tras un unload inmediato, no `drop_caches`.

## Phase 1 — Secuencial, un modelo cargado solo

Flujo por modelo: unload-all → chat `max_tokens=128` cold → unload → chat warm.
`wall_s` es tiempo total end-to-end (load + inference) medido por el cliente.

| Modelo | VRAM peak | Cold wall | Warm wall | t/s warm | Notas |
|---|---:|---:|---:|---:|---|
| nemotron3:33b | 25.6 GB | 62.7 s | 17.6 s | **6.8** | split 3060+3090 |
| nemotron-3-nano:30b | 25.1 GB | 68.0 s | 12.0 s | **10.0** 🥇 | mejor t/s de la ronda |
| qwen3-coder:30b | 26.3 GB | 52.6 s | 13.0 s | **7.4** | split |
| qwen3.8:27b (Q4_K_M) | 21.8 GB | 89.9 s | 21.2 s | **5.6** | — |
| qwen3.8:27b-abliterated (Q5_K_XL) | 23.6 GB | 88.3 s | 64.9 s | **1.9** 🐢 | 5× más lenta que el Q4_K_M por quant |
| gemma4:26b (vía `gemma4:26b-ollama`) | 24.1 GB | 57.1 s | 20.3 s | **5.9** | — |
| qwen3.6:latest | — | — | — | — | sin profile → 404 (saltado) |
| qwen3:32b | — | — | — | — | sin profile en el momento del bench; configurado después |

Highlights:
- `nemotron-3-nano:30b` es el ganador en t/s warm pese al split.
- `qwen3.8:27b-abliterated` (Q5_K_XL de huihui-ai) es ~5× más lento que la
  variante Q4_K_M del mismo modelo: la cuantización mayor amplifica la
  penalización PCIe del split.
- Diferencia cold vs warm = ~45–75 s, aproximadamente el coste de
  lectura secuencial de 20 GB de blob desde ZFS.

## Phase 2 — 7 modelos ≥14 B concurrentes

Las 7 peticiones se lanzan **en paralelo** a `t=0` contra
`/v1/chat/completions` (80 tokens), timeout de cliente HTTP 900 s.
oCabra encola; Ollama serializa cargas.

| # | Modelo | Terminó a `t+` | Status | Tokens |
|---|---|---:|---|---:|
| 1 | nemotron-3-nano:30b | 215 s | ✅ 200 | 80 |
| 2 | qwen3-coder:30b | 393 s | ✅ 200 | 73 |
| 3 | nemotron3:33b | 539 s | ✅ 200 | 80 |
| 4 | gemma4:26b-ollama | 647 s | ✅ 200 | 80 |
| 5 | qwen3.8:27b | 753 s | ✅ 200 | 80 |
| 6 | qwen3.8:27b-abliterated | 878 s | ✅ 200 | 80 |
| 7 | qwen3:32b | 900 s | ⚠️ timeout cliente | 0 (Ollama acabó cargando, 30 GB residentes al terminar la cola) |

Highlights:
- Serie estricta — ningún 503 por VRAM pressure, ningún salto de cola.
- Cada carga + inferencia libera el slot al cabo de **~100–125 s**.
- El **último de 7** tarda ~15 min en responder al cliente: con colas
  largas de modelos pesados, un cliente HTTP con timeout 900 s se queda
  corto. Dos mitigaciones posibles cuando se vean en producción:
  1. Elevar el timeout en los clientes conocidos (coder, portal).
  2. Que oCabra devuelva 503 con `Retry-After` cuando el estimado de
     tiempo-en-cola exceda un umbral (`model_load_wait_timeout_s`).

## Cómo rehacer este bench

Scripts usados (archivados en `/tmp/ocabra-bench/`):

- `bench.py` — Phase 1: por cada modelo de `MODELS`, unload-all → chat cold
  → unload → chat warm; captura `total_duration`, `load_duration`,
  `eval_duration` y VRAM via `nvidia-smi`.
- `phase2.py` — Phase 2: lanza `len(MODELS)` chats concurrentes con
  `ThreadPoolExecutor` y timeout 900 s; registra `finished_at_s` por
  tarea.

Para reproducir:

```bash
docker cp bench.py ocabra-api-1:/tmp/bench.py
docker exec -d ocabra-api-1 python3 /tmp/bench.py
# resultados en /tmp/bench.out (stdout)

# luego Phase 2:
docker cp phase2.py ocabra-api-1:/tmp/phase2.py
docker exec -d ocabra-api-1 python3 /tmp/phase2.py
```

Para comparar con una iteración futura, mantener constante:
- Pool CAP y FRAG (anotarlos; afectan al cold load > 10 %).
- Config de Ollama (`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`,
  `OLLAMA_NUM_PARALLEL`): un cambio en estos invalida los t/s.
- Prompt + `max_tokens`.
- Que no haya otros clientes externos (el bench de hoy coincidió con
  tráfico real pero poco; marcar como ruido).
