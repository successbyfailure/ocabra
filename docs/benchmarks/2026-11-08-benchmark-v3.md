# Referencia de modelos oCabra — Benchmark v3

**Fecha**: 2026-11-08  
**Hardware**: RTX 3060 (12 GB) + RTX 3090 (24 GB)  
**oCabra**: v0.49+

Guia para usuarios del servidor de IA. Cubre todos los modelos disponibles: LLM (generacion
de texto), STT (transcripcion de voz), embeddings (indexado y busqueda) y reranker.

---

# LLM — Modelos de lenguaje

9 modelos probados, todos disponibles. Cada uno evaluado en velocidad, calidad de extraccion,
contexto largo, latencia de primer token (TTFT) y concurrencia. Los modelos con
`reasoning_effort` (ravenx, qwen3.8:27b) se probaron a off/low/medium.

## Velocidad (tok/s)

Promedio de 4 tareas (extract_claims, summarize, verify_claim, section_analysis) por tamaño.

| Modelo | S (239w) | M (2033w) | L (5076w) | Validez M |
|--------|--------:|---------:|---------:|:---------:|
| **gemma4:26b** | **158** | **123** | **102** | **4/4** |
| nemotron-3-nano:30b | 134 | 102 | 91 | 4/4 |
| ravenx | 130 | 97 | 87 | 4/4 |
| gemma4:12b | 114 | 92 | 75 | 4/4 |
| nemotron3:33b | 9 ¹ | 76 ¹ | 85 | 2/3 |
| qwen3.8:27b | 67 | 62 | 47 | 4/4 |
| mistral:7b | 55 | 31 | 18 | 4/4 |
| qwen3:8b | 51 | 33 | 19 | 4/4 |
| qwen3:32b | 25 | 17 | 11 | 4/4 |

¹ nemotron3:33b: HTTP 500 en M/extract_claims y hit max_tokens en M/verify_claim.
Velocidad S anomala (9 tok/s) por cargas residuales.

**Top 3 velocidad (M)**: gemma4:26b (123), nemotron-3-nano:30b (102), ravenx (97).

## Calidad de extraccion

Extraer claims de una transcripcion M (2033 palabras) y comparar contra 100 claims de
referencia. Metricas por similitud coseno de embeddings (qwen3-embedding:0.6b).

- **Coverage**: que porcentaje de la referencia se captura (mayor = mas completo)
- **Precision**: que tan fieles son los claims extraidos (mayor = mas preciso)

### Modelos sin reasoning_effort

| Modelo | Claims | Coverage | Precision | tok/s |
|--------|-------:|---------:|----------:|------:|
| nemotron3:33b | **98** | 0.617 | 0.558 | 107 |
| nemotron-3-nano:30b | 80 | **0.619** | 0.613 | 102 |
| gemma4:12b | 35 | 0.559 | **0.685** | 76 |
| gemma4:26b | 33 | 0.521 | 0.574 | 131 |
| qwen3:8b | 41 | 0.538 | 0.591 | 24 |
| qwen3:32b | 26 | 0.522 | 0.617 | 11 |
| mistral:7b | — | — | — | 20 ² |

² mistral:7b no produjo JSON valido.

### Modelos con reasoning_effort

| Modelo | Nivel | Claims | Coverage | Precision | tok/s |
|--------|:-----:|-------:|---------:|----------:|------:|
| ravenx | **off** | 54 | **0.585** | **0.599** | 89 |
| ravenx | low | 65 | 0.571 | 0.577 | 103 |
| ravenx | medium | — | — | — | ³ |
| qwen3.8:27b | off | **69** | **0.574** | 0.561 | 67 |
| qwen3.8:27b | low | 34 | 0.549 | 0.610 | 64 |
| qwen3.8:27b | **medium** | 36 | 0.547 | **0.614** | 72 |

³ ravenx medium genera razonamiento infinito (hit max_tokens, JSON inutilizable).

**Hallazgos clave**:
- **gemma4:12b** logra la precision mas alta del benchmark (**0.685**) — cada claim
  que extrae es muy fiel al original.
- **ravenx**: usar siempre `off`. Mas claims y mejor calidad que con razonamiento.
- **qwen3.8:27b**: `off` para exhaustividad (69 claims), `medium` para precision (0.614).
- **nemotron-3-nano:30b**: mejor balance global (coverage 0.619, precision 0.613, 102 tok/s).

## Contexto largo

Resumen de transcripciones de 15k y 25k palabras.

| Modelo | 15k tok/s | 25k tok/s | Nota |
|--------|----------:|----------:|------|
| **gemma4:26b** | **41** | **50** | **El mas rapido en ctx largo** |
| gemma4:12b | 35 | 40 | Estable |
| nemotron3:33b | 39 | 37 | Estable |
| ravenx | 19 | 24 | OK |
| qwen3.8:27b | 21 | 24 | OK |
| nemotron-3-nano:30b | 18 | 20 | OK |
| mistral:7b | 17 | **FAIL** | Pool KV 32k insuficiente |
| qwen3:8b | 7 | **FAIL** | Preset no cubre 47k tokens |
| qwen3:32b | 5 | **FAIL** | Preset no cubre 47k tokens |

**Soportan 25k+ palabras**: gemma4:26b, gemma4:12b, ravenx, qwen3.8:27b, nemotron-3-nano:30b, nemotron3:33b.  
**Limite ~15k**: qwen3:8b, qwen3:32b, mistral:7b.

## TTFT (Time To First Token)

Latencia de streaming, promedio de 3 runs calientes.

| Modelo | Nivel | Total (s) | Nota |
|--------|:-----:|----------:|------|
| nemotron3:33b | default | **1.12** | El mas rapido |
| ravenx | off | **1.25** | Rapido sin razonamiento |
| nemotron-3-nano:30b | default | 1.42 | |
| qwen3.8:27b | off | 1.85 | |
| ravenx | low/medium | 2.15 | Razonamiento añade ~1s |
| gemma4:26b | default | 2.46 | |
| mistral:7b | default | 2.88 | |
| qwen3.8:27b | low | 3.11 | |
| gemma4:12b | default | 3.23 | |
| qwen3.8:27b | medium | 3.47 | |
| qwen3:8b | default | 8.74 | Lento |
| qwen3:32b | default | **22.73** | Muy lento |

**`reasoning_effort=off` reduce el TTFT drasticamente**: ravenx de 2.15s a 1.25s,
qwen3.8:27b de 3.47s a 1.85s.

## Concurrencia

Throughput total (tok/s) con peticiones simultaneas.

| Modelo | c=1 | c=2 | c=4 | Escala |
|--------|----:|----:|----:|:------:|
| gemma4:26b | 76 | 121 | **206** | 2.7x |
| ravenx | 38 | 81 | **188** | 5.0x |
| gemma4:12b | 56 | 88 | **170** | 3.0x |
| nemotron3:33b | 63 | 109 | 109 | 1.7x |
| nemotron-3-nano:30b | 33 | 61 | 88 | 2.7x |
| qwen3.8:27b | 27 | 56 | 80 | 3.0x |
| qwen3:8b | 17 | 27 | 40 | 2.3x |
| qwen3:32b | 9 | 20 | 34 | 3.6x |
| mistral:7b | 8 | 14 | 16 | 2.0x |

## Disponibilidad

| Modelo | Estado | Nota |
|--------|:------:|------|
| gemma4:12b | OK | |
| ravenx | OK | Mejor con `reasoning_effort=off` |
| nemotron-3-nano:30b | OK | |
| qwen3.8:27b | OK | Soporta `reasoning_effort` |
| qwen3:8b | OK | Contexto limitado ~27k tok |
| qwen3:32b | OK | Lento (11 tok/s en L) |
| nemotron3:33b | OK | Errores intermitentes |
| mistral:7b | OK | Contexto limitado ~32k tok |
| gemma4:26b | OK ¹ | El mas rapido; necesita 23,6 GB VRAM (solo cabe en la 3090). Disponibilidad depende de que la GPU este libre |

## Configuracion de reasoning_effort

Solo dos modelos responden a `reasoning_effort`. Valores: `off`, `low`, `medium`, `high`.

| Modelo | off | low | medium | Recomendacion |
|--------|-----|-----|--------|---------------|
| ravenx | Mejor calidad y velocidad | Mas claims, peor precision | Inutilizable (hit max) | **Usar siempre off** |
| qwen3.8:27b | Mas exhaustivo (69 claims) | Equilibrado | Mas preciso (0.614) | Segun prioridad |

---

# STT — Transcripcion de voz

Audio de prueba: 85,93 s, español, mono PCM 16 kHz, 2 hablantes reales.

## Velocidad y calidad

| Perfil | Motor | WER diar | DER | Veloc diar | WER sin | Veloc sin |
|--------|-------|--------:|----:|-----------:|--------:|----------:|
| `whisper-large-v3-turbo` | faster-whisper (GPU) | **4,18 %** | **10,39 %** | 17× | 3,77 % | 36× |
| `whisper-large-v3` | faster-whisper (GPU) | 5,86 % | 12,08 % | 15× | **3,77 %** | 19× |
| `faster-whisper-large-v3` | faster-whisper (GPU) | 5,44 % | 12,10 % | 10× | 3,77 % | 14× |
| `turbo-latam int8` | faster-whisper (GPU) | 19,67 % | 11,53 % | **27×** | 19,25 % | **64×** |
| `whisper.cpp` | whisper.cpp (CPU) | 11,72 % | 30,27 % | 1,1× | 5,02 % | 2,7× |

- **WER**: Word Error Rate (mas bajo = mejor transcripcion)
- **DER**: Diarization Error Rate (mas bajo = mejor identificacion de hablante)
- **Veloc**: veces mas rapido que tiempo real

## VRAM

| Perfil | VRAM | Nota |
|--------|-----:|------|
| `turbo-latam int8` | ~2.575 MiB | Cuantizado, muy ligero |
| `whisper-large-v3-turbo` | ~11.900 MiB | Incluye diarizador |
| `whisper-large-v3` | ~11.700 MiB | |
| `faster-whisper-large-v3` | ~9.260 MiB | |
| `whisper.cpp` (CPU) | 0 MiB | No toca GPU |

## Carga en frio

| Perfil | Primera peticion con diar | Primera sin diar |
|--------|-------------------------:|-----------------:|
| `whisper-large-v3-turbo` | 11,6 s | 2,5 s |
| `turbo-latam int8` | 30,4 s | 1,4 s |
| `faster-whisper-large-v3` | 32,3 s | 6,3 s |
| `whisper-large-v3` | 34,9 s | 4,7 s |
| `whisper.cpp` (CPU) | 87,2 s | 32,5 s |

## Notas

- **La diarizacion cuesta**: multiplica x2-3 el tiempo en GPU, x2 en CPU. Solo usarla
  cuando se necesite saber quien habla.
- **Perfiles `*-diarized`** son atajos: `whisper-large-v3-turbo-diarized` equivale a
  `whisper-large-v3-turbo` + `diarize=true`.
- **`recomendado-stt`** es un router que delega al modelo STT ya cargado, evitando espera.
- Los dos perfiles whisper.cpp son alias del mismo backend.

---

# Embeddings — Indexado y busqueda

Corpus de 200 evidence quotes de reuniones reales, 80 queries, relevancia graduada.

## Calidad de retrieval

| Modelo | Dims | NDCG@5 | NDCG@10 | MRR | Velocidad |
|--------|-----:|-------:|--------:|----:|----------:|
| `qwen3-embedding:8b` | 4096 | **0.366** | **0.296** | 0.595 | 28 ms/texto |
| `qwen3-embedding:4b` | 2560 | 0.361 | 0.294 | 0.595 | 18 ms/texto |
| `qwen3-embedding:0.6b` | 1024 | 0.350 | 0.286 | 0.570 | **8 ms/texto** |

Las diferencias de calidad son pequeñas (NDCG@5: 0.350 vs 0.366, solo 4,6% mas).

## Concurrencia embeddings

| Modelo | c=1 (texts/s) | c=4 (texts/s) | Ganancia |
|--------|-------------:|-------------:|---------:|
| `qwen3-embedding:0.6b` | 101 | **185** | **1,8x** |
| `qwen3-embedding:4b` | 46 | 62 | 1,4x |
| `qwen3-embedding:8b` | 30 | 44 | 1,5x |

## Reranker

| Modelo | MRR | NDCG@5 | Velocidad | Concurrencia |
|--------|----:|-------:|----------:|:------------:|
| `bge-reranker-v2-m3` | **0.601** | 0.349 | 412 ms/query | 2,9 q/s (c=4) |

El reranker iguala o mejora la calidad del retrieval puro (MRR 0.601 vs 0.595),
pero a un coste de 412 ms/query. Justificado como segunda pasada sobre top-K.

---

# Guia: que modelo usar para cada caso

## LLM

| Caso de uso | Modelo | Por que |
|-------------|--------|---------|
| **Procesamiento de reuniones** (Escriba) | `gemma4:12b` | 92 tok/s, precision 0.685, ctx largo 41k tok |
| **Maxima velocidad** | `gemma4:26b` | 123 tok/s en M, 206 tok/s a c=4, ctx largo 50 tok/s — necesita 3090 libre |
| **Mayor cobertura de extraccion** | `nemotron-3-nano:30b` | Coverage 0.619, precision 0.613, 102 tok/s |
| **Extraccion rapida de datos** | `ravenx` + `reasoning_effort=off` | 89 tok/s, coverage 0.585, TTFT 1.25s |
| **Maxima precision con razonamiento** | `qwen3.8:27b` + `reasoning_effort=medium` | Precision 0.614, razonamiento visible |
| **Chat interactivo / baja latencia** | `ravenx` + `reasoning_effort=off` | TTFT 1.25s, 130 tok/s en inputs cortos |
| **Documentos largos (>15k palabras)** | `gemma4:26b` / `gemma4:12b` | 50 / 40 tok/s a 41k tokens |
| **Volumen alto / batch** | `ravenx` + `reasoning_effort=off` | 188 tok/s a c=4, escala 5x |

## STT

| Caso de uso | Modelo | Por que |
|-------------|--------|---------|
| **Actas de reunion** (quien dice que) | `whisper-large-v3-turbo` + `diarize=true` | Mejor WER+DER (4,18% / 10,39%), 17x RT |
| **Transcripcion rapida sin hablantes** | `whisper-large-v3-turbo` | 3,77% WER, 36x RT |
| **Maxima velocidad GPU** | `turbo-latam int8` | 64x RT, pero WER ~20% |
| **GPU ocupada (fallback CPU)** | `whisper.cpp` sin diarizar | 5% WER, 2,7x RT, sin GPU |
| **Evitar espera de carga** | `recomendado-stt` (router) | Delega al STT que ya este en memoria |

## Embeddings

| Caso de uso | Modelo | Por que |
|-------------|--------|---------|
| **Pipeline RAG (Escriba)** | `qwen3-embedding:0.6b` | Mejor relacion calidad/velocidad, 8 ms/texto |
| **Maxima calidad de retrieval** | `qwen3-embedding:8b` + reranker | NDCG@5=0.366, MRR=0.601 |
| **Alta concurrencia** | `qwen3-embedding:0.6b` | 185 texts/s con c=4 |
| **Reranking (segunda pasada)** | `bge-reranker-v2-m3` | MRR 0.601, usar sobre top-K |

---

# Configuracion actual de Escriba

| Componente | Modelo | Configuracion |
|------------|--------|---------------|
| **STT** | `whisper-large-v3-diarized` | Perfil con diarizacion forzada |
| **LLM** | `gemma4:12b` (vLLM, ctx 128k) | `ESCRIBA_LLM_MAX_TOKENS=16384` |
| **Vision** | `gemma4:12b` | Override en `app_settings` |
| **Embeddings** | `qwen3-embedding:0.6b` | dim=1024 |
| **Reranker** | `bge-reranker-v2-m3` | Segunda pasada sobre top-K |
| **TTS** | No configurado | |

---

# Metodo

- **LLM**: script `eval/benchmark_v3_consolidated.py`. Transcripcion sintetica de reuniones
  en español (S=239w, M=2033w, L=5076w). 100 claims de referencia. Un modelo a la vez,
  secuencial, con warm-up previo.
- **STT**: script `eval/benchmark_ocabra_whisper.py`. Audio real 85,93s español, 2 hablantes.
  WER/DER contra referencia de consenso. Dos ejecuciones (fria + caliente).
- **Embeddings**: script `eval/benchmark_ocabra_embeddings_v2.py`. 200 docs reales, 80 queries,
  relevancia graduada (1.0/0.5/0.0). Metricas IR estandar.
- **Fecha**: STT y embeddings 2026-11-05/06, LLM 2026-11-08.
- **Datos completos**: `results.json` en este directorio y en los directorios de cada benchmark.
