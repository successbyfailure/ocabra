# Modelos

Cómo rinde cada modelo **en este despliegue concreto**. No son las cifras del
autor del modelo: salen de un benchmark propio con tareas reales —extracción
de datos de transcripciones de reuniones, resumen de documentos largos,
transcripción de audio en español, recuperación sobre un corpus real— más los
percentiles del tráfico que pasa por este servidor.

**Medido**: LLM el 8 de noviembre de 2026; voz y embeddings el 5-6 de
noviembre. Hardware: RTX 3060 (12 GB) + RTX 3090 (24 GB).

Hay una copia en Markdown sin maquetar en `GET /docs/models.md`, que es la
misma fuente que genera esta página. Los cambios de comportamiento de la API
están en [`/docs/changelog`](/docs/changelog).

## Cómo se pide un modelo

Pones su **nombre** en el campo `model` y ya está: `gemma4:26b`,
`qwen3-embedding:4b`, `whisper-large-v3-turbo`. Los nombres de esta página son
los que devuelve `GET /v1/models`, y son los únicos que necesitas.

Con qué motor se ejecuta cada modelo —llama.cpp, Ollama, vLLM, Whisper— lo
decide este servidor, y puede cambiarlo sin que tengas que tocar tu cliente. Lo
que sí cambia con el motor es el rendimiento, y eso es justo lo que miden las
tablas de abajo: son del despliegue que está configurado ahora.

Si tienes una razón para fijar el motor —comparar dos, o reproducir un
resultado— pídelo como `<motor>/<modelo>`: `ollama/gemma4:26b`,
`llamacpp_mks/gemma4:26b`. No salen en la lista, pero resuelven.

## Antes de leer las tablas

- **La 3090 es un hueco único.** Los modelos de 27B-33B piden entre 23 y 25 GB
  cada uno, así que **solo cabe uno a la vez**. Pedir un modelo grande desaloja
  al que estuviera. No esperes tener dos disponibles en caliente.
- **Hay 4 peticiones concurrentes por modelo.** Si las cuatro ranuras están
  ocupadas, tu petición espera. Ese tiempo de cola no está en estas cifras pero
  sí en `GET /ocabra/stats/performance`.
- **El contexto es un fondo común, no una cuota por petición.** Cada modelo
  reserva su caché entera por adelantado y las 4 ranuras la comparten. Una
  petición sola puede usar casi todo el contexto; cuatro grandes a la vez, no.
- **Mide dos veces.** Verás el mismo modelo tardando 120 s y luego 1 s con el
  mismo prompt: la primera pagó el arranque en frío. Si quieres medir el motor
  y no el planificador, descarta la primera.

## Si no quieres elegir

Seis perfiles enrutan por ti al mejor modelo disponible en ese momento.
Prefieren uno que ya esté cargado antes que forzar una descarga, así que
también son la opción más rápida en frío.

| Nombre | Para qué | Destinos, en orden de preferencia |
|--------|----------|-----------------------------------|
| `recomendado-sota` | Lo mejor que haya para chat general | `qwen3.8:27b`, `gemma4:26b`, `qwen3.8:27b-abliterated`, `ravenx-chaos-agent` |
| `recomendado-ha-chatbots` | Chat con disponibilidad por encima de calidad | `gemma4:26b`, `qwen3.8:27b`, `qwen3.8:27b-abliterated`, `gemma4:12b` |
| `recomendado-embedding` | Vectores | `qwen3-embedding:4b` |
| `recomendado-rerank` | Reordenar resultados | `bge-reranker-v2-m3` |
| `recomendado-stt` | Transcripción | `whisper-large-v3-turbo`, `faster-whisper-large-v3`, el turbo LATAM, y como último recurso la versión CPU |
| `recomendado-tts` | Voz | `piper-es-mx-claude-high`, `glados-es-fine-tuned`, `piper-es-es-davefx-medium` |

---

# Modelos de texto

## Qwen3.8 Flash-Next Coder en Strata

`qwen3.8-coder:iq1-m` es el perfil grande de programación y agentes. Usa
cuantización IQ1_M, contexto de 65.536 tokens, dos peticiones paralelas y
`reasoning_effort: medium` por defecto. Ocupa las dos GPU de forma asimétrica
(aprox. 11,3 GiB en la RTX 3060 y 23,7 GiB en la RTX 3090) y cerca de 90 GiB
de RAM total del host, por lo que se carga bajo demanda y desaloja los modelos
que estuvieran en cualquiera de las tarjetas.

En la prueba concurrente sostuvo aproximadamente 36–40 tok/s por petición.
Está pensado para tareas de código difíciles que justifican el arranque y la
presión de memoria, no para respuestas cortas de baja latencia.

## Velocidad de generación

Media de cuatro tareas reales (extraer afirmaciones, resumir, verificar,
analizar una sección) sobre transcripciones de 239, 2.033 y 5.076 palabras.

| Modelo | corto | medio | largo | Concurrencia ×4 | Contexto |
|--------|------:|------:|------:|----------------:|---------:|
| **`gemma4:26b`** | **158** | **123** | **102** | **206** (2,7×) | 256k |
| `nemotron-3-nano:30b` | 134 | 102 | 91 | 88 (2,7×) | 512k |
| `ravenx` | 130 | 97 | 87 | **188** (5,0×) | 240k |
| `gemma4:12b` | 114 | 92 | 75 | 170 (3,0×) | 256k |
| `qwen3.8:27b` | 67 | 62 | 47 | 80 (3,0×) | 96k · 256k con `extended` |
| `mistral:7b` | 55 | 31 | 18 | 16 (2,0×) | 32k |
| `qwen3:8b` | 51 | 33 | 19 | 40 (2,3×) | 32k |
| `qwen3:32b` | 25 | 17 | 11 | 34 (3,6×) | 40k |
| `nemotron3:33b` | — | — | 85 | 109 (1,7×) | 65k |

Cifras en tokens por segundo. "Concurrencia ×4" es el *throughput total* con
cuatro peticiones simultáneas, y el multiplicador respecto a una sola.

`gemma4:26b` es el más rápido en todos los tamaños y dobla en velocidad a
modelos de 8B porque usa decodificación especulativa (igual que `gemma4:12b`,
`qwen3.8:27b` y `qwen3.8:27b-abliterated`). Si vas a mandar volumen, `ravenx`
es el que mejor escala: ×5 al pasar de una petición a cuatro.

**`nemotron3:33b` no es recomendable.** Es el único modelo que queda sobre
Ollama y, si las GPUs están ocupadas cuando le toca cargar, Ollama lo reparte
en CPU y se queda ahí todo su keep-alive: medido a **0,3 tok/s**, con
peticiones de dos minutos para 32 tokens. Sus cifras buenas de esta tabla son
de cuando le toca entero en GPU, y no puedes saber en qué caso estás. Usa
`nemotron-3-nano:30b`, que es el mismo tipo de modelo sobre llama.cpp.

## Calidad de extracción

Extraer afirmaciones de una transcripción de 2.033 palabras y comparar contra
100 de referencia, por similitud coseno de embeddings.

- **Cobertura**: cuánto de la referencia se captura. Más alto = más completo.
- **Precisión**: cuán fieles son las afirmaciones extraídas. Más alto = menos
  invención.

| Modelo | Afirmaciones | Cobertura | Precisión |
|--------|-------------:|----------:|----------:|
| `nemotron-3-nano:30b` | 80 | **0,619** | 0,613 |
| `gemma4:12b` | 35 | 0,559 | **0,685** |
| `qwen3:8b` | 41 | 0,538 | 0,591 |
| `qwen3:32b` | 26 | 0,522 | 0,617 |
| `gemma4:26b` | 33 | 0,521 | 0,574 |
| `ravenx` (`off`) | 54 | 0,585 | 0,599 |
| `qwen3.8:27b` (`off`) | 69 | 0,574 | 0,561 |
| `qwen3.8:27b` (`medium`) | 36 | 0,547 | 0,614 |
| `mistral:7b` | — | — | — |

`mistral:7b` no produjo JSON válido en ninguna tirada: **no lo uses para
salida estructurada**.

Lo importante aquí no es el ranking, es que **velocidad y calidad no van
juntas**: `gemma4:26b` es el más rápido y el que menos extrae; `gemma4:12b` es
el más fiel de todo el parque; `nemotron-3-nano:30b` es el mejor equilibrio.

## Documentos largos

Resumen de transcripciones de 15.000 y 25.000 palabras.

| Modelo | 15k palabras | 25k palabras |
|--------|-------------:|-------------:|
| **`gemma4:26b`** | **41** | **50** |
| `gemma4:12b` | 35 | 40 |
| `nemotron3:33b` | 39 | 37 |
| `ravenx` | 19 | 24 |
| `qwen3.8:27b` | 21 | 24 |
| `nemotron-3-nano:30b` | 18 | 20 |
| `mistral:7b` | 17 | no cabe |
| `qwen3:8b` | 7 | no cabe |
| `qwen3:32b` | 5 | no cabe |

**Un documento de 25.000 palabras son unos 43.000 tokens**, así que no entra en
`qwen3:8b` (32k), `mistral:7b` (32k) ni `qwen3:32b` (40k): recibirás un `400`
`context_length_exceeded`. Sí entra en los demás. A 15.000 palabras (~29.000
tokens) entran todos.

Y ojo con el tokenizador: el **mismo** texto de 25.000 palabras son 46.860
tokens para la familia Qwen y **52.133 para `mistral:7b`**, un 11 % más. Si
vas al límite de la ventana, cuenta con el modelo concreto, no con una regla
de palabras.

## Latencia al primer token

Streaming, media de tres tiradas en caliente.

| Modelo | Nivel | TTFT |
|--------|:-----:|-----:|
| `ravenx` | `off` | **1,25 s** |
| `nemotron-3-nano:30b` | — | 1,42 s |
| `qwen3.8:27b` | `off` | 1,85 s |
| `ravenx` | `low`/`medium` | 2,15 s |
| `gemma4:26b` | — | 2,46 s |
| `mistral:7b` | — | 2,88 s |
| `qwen3.8:27b` | `low` | 3,11 s |
| `gemma4:12b` | — | 3,23 s |
| `qwen3.8:27b` | `medium` | 3,47 s |
| `qwen3:8b` | — | 8,74 s |
| `qwen3:32b` | — | **22,73 s** |

Para chat interactivo, `reasoning_effort: "off"` es lo que más baja el TTFT:
`ravenx` pasa de 2,15 s a 1,25 s y `qwen3.8:27b` de 3,47 s a 1,85 s.

`qwen3:32b` tarda 22 s en empezar incluso cargado, porque se reparte entre las
dos tarjetas. No lo uses para nada interactivo.

## Razonamiento

Solo **tres** modelos distinguen de verdad los niveles de `reasoning_effort`,
porque su plantilla los lee: `qwen3.8:27b`, `qwen3.8:27b-abliterated` y
`ravenx-chaos-agent`. Aceptan `low`, `medium` y `xhigh`; oCabra traduce el
`high` de OpenAI a `xhigh` por ti.

El resto —`gemma4:26b`, `gemma4:12b`, `qwen3:32b`, `qwen3:8b`,
`nemotron-3-nano:30b`, `ravenx`— solo tiene **interruptor**: piensa o no
piensa. Con ellos `off` apaga el razonamiento, y `low`, `medium` y `high` se
diferencian únicamente en cuánto pensamiento les deja oCabra antes de cortar.

Qué conviene pedir, medido:

| Modelo | Recomendación |
|--------|---------------|
| `ravenx` | **`off` siempre.** Con razonamiento extrae menos y peor, y el TTFT casi se dobla. |
| `qwen3.8:27b` | `off` si quieres exhaustividad (69 afirmaciones), `medium` si quieres precisión (0,614). |

Cuánto piensan de media, cuando piensan, sobre el tráfico real del servidor:

| Modelo | Tokens de razonamiento por respuesta |
|--------|-------------------------------------:|
| `ravenx` | 2.750 |
| `gemma4:26b` | 1.715 |
| `nemotron-3-nano:30b` | 970 |
| `gemma4:12b` | 560 |
| `qwen3.8:27b` | 405 |
| `ravenx-chaos-agent` | 28 |

**Acota la salida y el razonamiento se acota con ella.** oCabra reserva siempre
un cuarto de tu `max_tokens` para la respuesta, así que un `max_tokens` pequeño
recorta el pensamiento en vez de dejarte sin respuesta (ver la entrada 0.51.2
del [changelog](/docs/changelog)). Aun así, si pides razonamiento, da holgura:
con `max_tokens: 4096` y `medium` el modelo piensa como mucho 3.072 tokens.

## Visión

`gemma4:26b` y `gemma4:12b` aceptan imágenes (`image_url` en las content
parts). El proyector visual se carga **cuando llega la primera imagen**, no al
arrancar: esa primera petición tarda unos segundos más. Ningún otro modelo de
texto del parque acepta imágenes.

## Disponibilidad

| Modelo | Nota |
|--------|------|
| `gemma4:26b` | Fijado en la 3090, **pero su pin es evictable**: reserva 23.600 de los 24.576 MB, así que cuando otro modelo grande se queda la tarjeta sale, y no vuelve hasta que quede libre. Entre tanto responde `409`. Si tu carga alterna modelos grandes, usa `recomendado-ha-chatbots`. |
| `qwen3.8:27b` | Se mantiene caliente. |
| `qwen3-embedding:8b` | Se mantiene caliente. |
| `whisper-large-v3-turbo` | Fijado en la 3060, así que transcribir no compite por la 3090. |
| `nemotron3:33b` | Ver el aviso de la tabla de velocidad. |
| El resto | Bajo demanda: la primera petición paga la carga. |

---

# Transcripción de voz

Audio real de 85,9 s en español, mono 16 kHz, dos hablantes.

| Perfil | WER | Velocidad | Con diarización: WER / DER / velocidad |
|--------|----:|----------:|---------------------------------------:|
| **`whisper-large-v3-turbo`** | 3,77 % | 36× | **4,18 % / 10,39 % / 17×** |
| `whisper-large-v3` | **3,77 %** | 19× | 5,86 % / 12,08 % / 15× |
| `faster-whisper-large-v3` | 3,77 % | 14× | 5,44 % / 12,10 % / 10× |
| `turbo-latam int8` | 19,25 % | **64×** | 19,67 % / 11,53 % / 27× |
| `whisper.cpp` (CPU) | 5,02 % | 2,7× | 11,72 % / 30,27 % / 1,1× |

**WER** = errores de palabra (más bajo, mejor). **DER** = errores de
atribución de hablante. **Velocidad** = veces más rápido que el tiempo real.

`whisper-large-v3-turbo` gana en todo: mismo WER que los grandes, el doble de
rápido, y el mejor DER cuando diarizas. Es lo que sirve `recomendado-stt`.

**`turbo-latam int8` tiene un WER de casi el 20 %**, cinco veces peor que el
resto, y no es la cuantización: es un problema del perfil. Es muy rápido (64×)
pero no lo uses si te importa lo que dice el texto.

La versión **CPU** (`whisper.cpp`) transcribe razonablemente bien (5 % WER) y
no toca la GPU, pero su diarización no sirve (DER 30 %) y a 1,1× el tiempo
real tarda casi lo que dura el audio. Está como red de seguridad cuando las
dos tarjetas están ocupadas.

## Diarización y arranque en frío

| Perfil | VRAM | Primera petición | Con diarización |
|--------|-----:|-----------------:|----------------:|
| `turbo-latam int8` | 2,6 GB | 1,4 s | 30,4 s |
| `faster-whisper-large-v3` | 9,3 GB | 6,3 s | 32,3 s |
| `whisper-large-v3` | 11,7 GB | 4,7 s | 34,9 s |
| `whisper-large-v3-turbo` | 11,9 GB | 2,5 s | 11,6 s |
| `whisper.cpp` (CPU) | 0 | 32,5 s | 87,2 s |

**La diarización cuesta de 2 a 3 veces más tiempo** y arranca un proceso
aparte la primera vez. Pídela solo cuando necesites saber quién habla:
`diarize: true`, o los perfiles `*-diarized`, que son el mismo modelo con la
opción puesta.

Nota de capacidad: `whisper-large-v3-turbo` con diarizador ocupa 11,9 GB de los
12,2 de la 3060, es decir el 97 %. Deja la tarjeta prácticamente llena.

---

# Embeddings y reranking

Corpus de 200 fragmentos reales de reuniones, 80 consultas, relevancia
graduada.

| Modelo | Dimensiones | NDCG@5 | MRR | Latencia | Concurrencia ×4 |
|--------|------------:|-------:|----:|---------:|----------------:|
| `qwen3-embedding:8b` | 4.096 | **0,366** | 0,595 | 28 ms | 44 textos/s |
| `qwen3-embedding:4b` | 2.560 | 0,361 | 0,595 | 18 ms | 62 textos/s |
| `qwen3-embedding:0.6b` | 1.024 | 0,350 | 0,570 | **8 ms** | **185 textos/s** |

**La diferencia de calidad es mínima y la de velocidad enorme.** Entre el de
0,6B y el de 8B hay un 4,6 % de NDCG@5, pero el pequeño es 3,5 veces más
rápido y escala a 185 textos por segundo. Para indexar volumen, el de 0,6B; el
de 4B es el término medio y es el que sirve `recomendado-embedding`.

Los tres aceptan hasta 32k-40k tokens por entrada y los tres soportan
`dimensions` para truncar el vector: oCabra lo trunca y **renormaliza a L2**,
así que el coseno sigue siendo comparable.

| Reranker | MRR | NDCG@5 | Latencia |
|----------|----:|-------:|---------:|
| `bge-reranker-v2-m3` | **0,601** | 0,349 | 412 ms/consulta |

El reranker mejora el MRR por encima de cualquier embedding solo (0,601 frente
a 0,595), pero cuesta 412 ms por consulta. Tiene sentido como **segunda pasada
sobre los primeros K resultados**, no como buscador.

---

# Qué usar para cada caso

| Caso | Modelo | Por qué |
|------|--------|---------|
| Chat interactivo | `ravenx` + `reasoning_effort: "off"` | TTFT 1,25 s, 130 tok/s en entradas cortas |
| Máxima velocidad | `gemma4:26b` | 123 tok/s, 206 con concurrencia — si la 3090 está libre |
| Documentos largos | `gemma4:26b` o `gemma4:12b` | 50 y 40 tok/s con 25.000 palabras |
| Extracción exhaustiva | `nemotron-3-nano:30b` | Mejor cobertura (0,619) con precisión alta |
| Extracción fiel | `gemma4:12b` | La precisión más alta del parque (0,685) |
| Volumen / lotes | `ravenx` + `off` | 188 tok/s con 4 en paralelo, escala ×5 |
| Razonamiento visible | `qwen3.8:27b` + `medium` | Es uno de los tres que distinguen niveles de verdad |
| Salida estructurada | cualquiera **menos** `mistral:7b` | `mistral:7b` no produjo JSON válido |
| Actas con hablantes | `whisper-large-v3-turbo` + `diarize: true` | WER 4,18 %, DER 10,39 %, 17× tiempo real |
| Transcripción rápida | `whisper-large-v3-turbo` | WER 3,77 %, 36× tiempo real |
| Transcribir sin GPU | `whisper.cpp` sin diarizar | WER 5 %, no toca VRAM |
| Indexar volumen | `qwen3-embedding:0.6b` | 8 ms por texto, 185/s en paralelo |
| Máxima calidad de búsqueda | `qwen3-embedding:8b` + `bge-reranker-v2-m3` | NDCG@5 0,366 y MRR 0,601 |

## Qué falla y por qué

| Código | Qué significa | Qué hacer |
|--------|---------------|-----------|
| `400` `context_length_exceeded` | Tu prompt no entra en la ventana del modelo. El número que da el error es el real, contado con el tokenizador del modelo | Parte la petición o usa un modelo con más contexto |
| `409` `insufficient_vram` | No cabía en la GPU ni desalojando. Es el fallo propio de este despliegue: dos tarjetas desiguales y modelos que no caben juntos | Reintenta en unos segundos, pide un preset de contexto más pequeño, o usa un `recomendado-*` |
| `429` | Las 4 ranuras del modelo estaban llenas | Respeta el `Retry-After`; la admisión reparte por turnos entre claves, insistir no te adelanta |
| `504` | El motor no respondió a tiempo | Casi siempre un prompt enorme con el contexto casi lleno |

El `409` llegó a ser el 12,5 % del tráfico la semana del 26 de octubre, por
tres carreras de condición en el reparto de VRAM que se arreglaron el 4 de
noviembre. Ahora está en el 1,1 %.

## Medirlo tú mismo

- `GET /v1/models` — qué hay disponible, con capacidades y contexto.
- `GET /ocabra/stats/performance` — latencias reales, con la espera en cola, el
  arranque en frío y el tiempo de motor **separados**.
- `GET /ocabra/stats/recent` — las últimas peticiones con su código de estado,
  para diagnosticar un fallo que acabas de provocar.
