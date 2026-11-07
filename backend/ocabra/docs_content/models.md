# Modelos

Cómo rinde cada modelo **en este despliegue concreto**. No son las cifras del
autor del modelo ni un benchmark sintético: son percentiles sobre el tráfico
real que ha pasado por este servidor entre el **23 de septiembre y el 7 de
noviembre de 2026**, con el número de peticiones de cada muestra a la vista
para que sepas cuánto fiarte.

Hay una copia en Markdown sin maquetar en `GET /docs/models.md`, que es la
misma fuente que genera esta página. Los cambios de comportamiento de la API
están en [`/docs/changelog`](/docs/changelog).

## Antes de leer las tablas

- **Los tokens/segundo dependen sobre todo del prompt.** Un modelo con prompts
  de 45.000 tokens de media sale peor que el mismo modelo con prompts de 5.000,
  aunque el motor sea idéntico. Por eso la tabla incluye el **prompt medio** de
  cada muestra: compara modelos con prompts parecidos, no la columna de
  tok/s a secas.
- **Hay 4 peticiones concurrentes por modelo.** Si las cuatro ranuras están
  ocupadas, tu petición espera; el tiempo de cola no está en estas cifras pero
  sí en `GET /ocabra/stats/performance`, que lo separa del tiempo de motor.
- **El contexto es un fondo común, no una cuota por petición.** Cada modelo
  reserva su caché KV entera por adelantado y las 4 ranuras la comparten. Una
  petición sola puede usar casi todo el contexto; cuatro peticiones grandes a la
  vez, no.
- Las cifras de tok/s salen solo de peticiones **terminadas con éxito**. Los
  fallos están en la sección [Qué falla y por qué](#que-falla-y-por-que).

## Cómo se pide un modelo

Pones su **nombre** en el campo `model` y ya está: `gemma4:26b`,
`qwen3-embedding:4b`, `whisper-large-v3-turbo`. Los nombres de esta página son
los que devuelve `GET /v1/models`, y son los únicos que necesitas.

Con qué motor se ejecuta cada modelo —llama.cpp, Ollama, vLLM, Whisper— lo
decide este servidor, y puede cambiarlo sin que tengas que tocar tu cliente. Lo
que sí cambia con el motor es el rendimiento, y eso es justo lo que miden las
tablas de abajo: son del despliegue que está configurado ahora.

## El hardware

| GPU | Tarjeta | VRAM |
|-----|---------|------|
| 0 | RTX 3060 | 12 GB |
| 1 | RTX 3090 | 24 GB |

Dos tarjetas desiguales, y de ahí vienen casi todas las particularidades: los
modelos grandes solo caben en la 3090, la 3060 se reserva para los pequeños y
para Whisper, y un par de modelos se reparten entre las dos. No esperes que dos
modelos de 30B estén cargados a la vez.

## Si no quieres elegir

Seis perfiles enrutan por ti al mejor modelo disponible en ese momento:
prefieren uno que ya esté cargado antes que forzar una descarga, así que
también son la opción más rápida en frío.

| Nombre | Para qué | Destinos, en orden de preferencia |
|--------|----------|-----------------------------------|
| `recomendado-sota` | Lo mejor que haya para chat general | `qwen3.8:27b`, `gemma4:26b`, `qwen3.8:27b-abliterated`, `ravenx-chaos-agent` |
| `recomendado-ha-chatbots` | Chat con disponibilidad por encima de calidad | `gemma4:26b`, `qwen3.8:27b`, `qwen3.8:27b-abliterated`, `gemma4:12b` |
| `recomendado-embedding` | Vectores | `qwen3-embedding:4b` |
| `recomendado-rerank` | Reordenar resultados | `bge-reranker-v2-m3` |
| `recomendado-stt` | Transcripción | `whisper-large-v3-turbo`, `faster-whisper-large-v3`, el turbo LATAM, y como último recurso la versión CPU |
| `recomendado-tts` | Voz | `piper-es-mx-claude-high`, `glados-es-fine-tuned`, `piper-es-es-davefx-medium` |

## Modelos de texto

| Modelo | Peticiones | tok/s (p50) | Prefill (tok/s) | Prompt medio | Contexto por defecto | Estado |
|--------|-----------:|------------:|----------------:|-------------:|---------------------|--------|
| `gemma4:26b` | 861 | **149** | 845 | 9.600 | 256k | siempre cargado |
| `nemotron-3-nano:30b` | 105 | **150** | 539 | 5.100 | 512k | bajo demanda, 2 GPUs |
| `ravenx` | 465 | 109 | 730 | 34.600 | 240k | bajo demanda |
| `gemma4:12b` | 231 | 105 | 656 | 6.700 | 256k | bajo demanda |
| `qwen3-coder:30b` | 131 | 59 | 711 | 29.400 | 116k (`extended`: 256k) | bajo demanda |
| `qwen3.8:27b` | 551 | 47 | 412 | 45.500 | 96k (`extended`: 256k) | se mantiene caliente |
| `qwen3:8b` | 120 | 45 | 584 | 3.800 | 32k | bajo demanda |
| `mistral:7b` | 75 | 44 | 622 | 4.200 | 32k | bajo demanda |
| `ravenx-chaos-agent` | 1.417 | 24 | 328 | 33.700 | 128k (`extended`: 256k) | bajo demanda |
| `qwen3:32b` | 90 | 21 | 333 | 4.500 | 40k | bajo demanda, 2 GPUs |
| `qwen3.8:27b-abliterated` | 7 | ~47 | 77 | — | 40k (`extended`: 224k) | bajo demanda |

`gemma4:12b`, `gemma4:26b`, `qwen3.8:27b` y `qwen3.8:27b-abliterated` usan
decodificación especulativa (MTP) y por eso van más rápido de lo que su tamaño
sugiere: `gemma4:26b` dobla en tok/s a modelos de 8B. El contraste con
`qwen3.8:27b` no es del motor, es del prompt — 45.000 tokens de media frente a
9.600.

La muestra de `qwen3.8:27b-abliterated` (7 peticiones) no da para un percentil;
está en la tabla para que sepas que existe, no para compararlo.

### Razonamiento

Solo **tres** modelos distinguen de verdad los niveles de `reasoning_effort`:
`qwen3.8:27b`, `qwen3.8:27b-abliterated` y `ravenx-chaos-agent`. Su plantilla
acepta `low`, `medium` y `xhigh`; oCabra traduce el `high` de OpenAI a `xhigh`.

El resto —`gemma4:26b`, `gemma4:12b`, `qwen3:32b`, `qwen3:8b`,
`nemotron-3-nano:30b`— solo tiene **interruptor**: piensa o no piensa. Con esos
modelos `low`, `medium` y `high` producen lo mismo, y `"off"` sí apaga el
razonamiento. `mistral:7b` y `qwen3-coder:30b` no razonan en absoluto.

Cuánto piensan de media, cuando piensan:

| Modelo | Tokens de razonamiento por respuesta |
|--------|-------------------------------------:|
| `ravenx` | 2.750 |
| `gemma4:26b` | 1.715 |
| `nemotron-3-nano:30b` | 970 |
| `gemma4:12b` | 560 |
| `qwen3.8:27b` | 405 |
| `qwen3:8b` | 270 |
| `ravenx-chaos-agent` | 28 |

Si pagas por token de salida o tienes un presupuesto de latencia, cuenta con
esto: `gemma4:26b` gasta de media 1.700 tokens antes de empezar a responder.
Todos los modelos tienen un tope de salida que oCabra aplica aunque tú no
mandes `max_tokens` (ver la entrada 0.41.0 del
[changelog](/docs/changelog)), así que un razonamiento desbocado se corta en vez
de consumirte el contexto entero.

### Visión

`gemma4:26b` y `gemma4:12b` aceptan imágenes (`image_url` en las content
parts). El proyector visual se carga **cuando llega la primera imagen**, no al
arrancar: la primera petición con imagen de una sesión tarda unos segundos más.
Ningún otro modelo de texto del parque acepta imágenes.

## Embeddings

| Modelo | Dimensiones | Peticiones | Latencia p50 | p90 | Prompt mediano |
|--------|------------:|-----------:|-------------:|----:|---------------:|
| `qwen3-embedding:0.6b` | 1.024 | 2.004 | 97 ms | 3,8 s | 9 tokens |
| `qwen3-embedding:4b` | 2.560 | 572 | 65 ms | 2,6 s | 12 tokens |
| `qwen3-embedding:8b` | 4.096 | 3.238 | 416 ms | 4,1 s | 48 tokens |

Los tres aceptan hasta 32k–40k tokens por entrada y los tres soportan
`dimensions` para truncar el vector (oCabra lo trunca y **renormaliza a L2**,
así que el coseno sigue siendo comparable). El salto del p50 al p90 es cola de
arranque en frío y de cola de espera, no del modelo: si tu carga es a ráfagas,
cuenta con el p90.

`qwen3-embedding:8b` se mantiene caliente; los otros dos se cargan bajo
demanda. Para indexar mucho, el 4b es el mejor compromiso y es el que sirve
`recomendado-embedding`.

## Reranking

| Modelo | Peticiones | Latencia p50 | p90 | Prompt mediano |
|--------|-----------:|-------------:|----:|---------------:|
| `bge-reranker-v2-m3` | 849 | 1,0 s | 20,4 s | 8.131 tokens |

El p90 alto es el coste de los lotes grandes: la latencia crece con el número
de documentos que mandas. Parte en tandas si necesitas una respuesta acotada.

## Transcripción (STT)

| Modelo | Peticiones | p50 | p95 |
|--------|-----------:|----:|----:|
| `faster-whisper-large-v3` | 2.062 | 2,0 s | 115 s |
| `faster-whisper-large-v3-turbo-latam-int8-ct2` | 75 | 4,1 s | 128 s |
| `whisper-large-v3-turbo` | 97 | 7,9 s | 215 s |
| `whisper-large-v3` | 34 | 23,3 s | 159 s |

Aquí la latencia es casi toda duración del audio, así que estos números dicen
más de lo que la gente sube que del modelo. Lo que sí es comparable es el
**streaming** del endpoint Realtime, donde los trozos son del mismo tamaño:

| Modelo | Trozos | p50 | p95 |
|--------|-------:|----:|----:|
| `whisper-large-v3` | 59.433 | **685 ms** | 730 ms |
| `whisper-large-v3-turbo` | 20 | 667 ms | 4,9 s |

`whisper-large-v3-turbo` está fijado en la 3060, así que transcribir no compite
con el modelo de texto por la 3090. Las variantes `-diarized` añaden separación
de hablantes y arrancan un proceso aparte la primera vez: cuenta con unos
segundos extra en la primera petición.

Hay además una variante que corre **en CPU**
(`whisper.cpp-large-v3-turbo-q8-0-cpu`), mucho más lenta, que existe como red
de seguridad cuando las dos GPUs están ocupadas. `recomendado-stt` solo cae en
ella como último recurso.

## Voz (TTS)

| Modelo | Peticiones | p50 | p95 |
|--------|-----------:|----:|----:|
| `piper-es-es-davefx-medium` | 42 | 295 ms | 4,0 s |
| `glados-es-fine-tuned` | 160 | 1,3 s | 9,5 s |

En el endpoint Realtime, con frases cortas, los dos bajan de 550 ms de media
(`glados-es-fine-tuned` 409 ms p50 sobre 90 frases). Son modelos Piper: ligeros,
sin GPU dedicada y sin arranque en frío apreciable. Hay también
`piper-es-mx-claude-high` (el que sirve `recomendado-tts` por defecto) y
`piper-en-us-lessac-high` para inglés.

## Imagen, música y escalado

| Modelo | Para qué | Peticiones | p50 |
|--------|----------|-----------:|----:|
| `qwen-image-2.1` | Generar imagen | 15 | 114 s |
| `qwen-image-edit-plus` | Editar imagen (`/v1/images/edits`) | — | — |
| `ace-step-turbo` | Música | — | — |
| `realesrgan-x4plus`, `remacri-x4`, `ultrasharp-x4`, `ultrasharp-v2-lite`, `real-esrgan-compact-x4` | Escalar imagen ×4 | — | — |
| `flashvsr-v1-1` | Escalar vídeo | — | — |

Generar una imagen son **dos minutos**, y el arranque en frío del pipeline son
otros 90–210 s. Si tu cliente tiene un timeout de 60 s, no vas a ver una sola
imagen: usa la cola de Batches o sube el timeout. Las muestras de edición,
música y escalado son demasiado pequeñas para publicar un percentil.

## Arranque en frío

Casi todos los modelos se cargan **bajo demanda**: si nadie lo ha usado
últimamente, tu petición paga la carga. Mediana medida sobre las cargas reales
del periodo:

| Qué | Cuánto tarda |
|-----|--------------|
| GGUF grande (27B–32B) en llama.cpp | 38–51 s |
| `gemma4:26b` | 41 s (pero está fijado: normalmente no lo pagas) |
| `nemotron-3-nano:30b` | 62 s |
| Reranker, Whisper | ~50 s |
| Pipeline de imagen (Diffusers) | 88–213 s |

Tres excepciones que no pagas casi nunca: `gemma4:26b` está **fijado** en la
3090, `qwen3.8:27b` y `qwen3-embedding:8b` se mantienen **calientes**, y
`whisper-large-v3-turbo` y dos voces Piper están fijados también.

El consejo práctico: si vas a hacer muchas peticiones, mándalas seguidas. Y si
te da igual el modelo, usa un perfil `recomendado-*`, que prefiere lo que ya
esté en VRAM.

## Qué falla y por qué

Sobre 3.207 peticiones de chat en la última semana del periodo:

| Código | Qué significa | Cuántas |
|--------|---------------|--------:|
| `409` `insufficient_vram` | No cabía en la GPU, ni descargando otros modelos | 36 |
| `429` | Las 4 ranuras del modelo estaban llenas | 7 |
| `504` | El motor no respondió a tiempo | 6 |

El `409` es el fallo propio de este despliegue: dos tarjetas desiguales y
modelos que no caben juntos. Durante la semana del 26 de octubre hubo **1.242**
de estos, un 12,5 % del tráfico, por tres carreras de condición en el reparto
de VRAM entre las dos GPUs que se arreglaron el 4 de noviembre (entradas 0.44.0
y siguientes del [changelog](/docs/changelog)). Ahora están en el 1,1 %.

Qué hacer con cada uno:

- **`409`** — Reintenta tras unos segundos, o pide un preset de contexto más
  pequeño (`normal` en vez de `extended`), o cambia a un modelo que quepa en la
  3060. No viene con `Retry-After` porque no es transitorio por sí solo: algo
  tiene que liberar VRAM.
- **`429`** y **`503`** — Vienen con `Retry-After`. Respétalo; la admisión
  reparte por turnos entre claves, así que insistir no te adelanta.
- **`504`** — Casi siempre un prompt enorme contra un modelo con el contexto
  casi lleno. Baja el contexto o parte la petición.

## Medirlo tú mismo

Nada de esto hay que creérselo: son consultas sobre las métricas que el propio
servidor publica.

- `GET /v1/models` — qué hay disponible, con las capacidades (`chat`, `tools`,
  `vision`, `reasoning`, `embeddings`) y el contexto que anuncia cada modelo.
- `GET /ocabra/stats/performance` — latencias reales, con la espera en cola, el
  arranque en frío y el tiempo de motor **separados**. Es la cifra que importa
  si te preocupa la latencia de punta a punta.
- `GET /ocabra/stats/recent` — las últimas peticiones con su código de estado,
  para diagnosticar un fallo concreto que acabas de provocar.

Esta página se regenera a mano; si la fecha del encabezado está lejos, pide los
números en vivo a esos tres endpoints.
