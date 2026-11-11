# Modelos

Cómo rinde cada modelo **en este despliegue concreto**. No son las cifras del
autor del modelo: salen de un benchmark propio con tareas reales —extracción
de datos de transcripciones de reuniones, resumen de documentos largos,
transcripción de audio en español, recuperación sobre un corpus real— más los
percentiles del tráfico que pasa por este servidor.

**Medido**: LLM el 11 de noviembre de 2026, con el llama.cpp actualizado y
oCabra 0.68; voz y embeddings el 5-6 de noviembre. Hardware: RTX 3060 (12 GB)
+ RTX 3090 (24 GB).

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

- **La 3090 es un hueco único.** Los modelos de 27B-40B piden entre 22 y 25 GB
  cada uno, así que **solo cabe uno grande a la vez**. Pedir otro grande
  desaloja al que estuviera. No esperes tener dos disponibles en caliente.
  Lo que sí ocurre es que un modelo grande ocioso **reduce su contexto** para
  dejar sitio a uno pequeño, como un embedding, en vez de salir. Si después le
  llega una petición que no cabe en el contexto reducido, lo recupera en menos
  de un segundo.
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
| **`nemotron-3.5-lightning:30b`** | 158 | **148** | **134** | 221 (1,9×) | 128k, en las dos tarjetas |
| **`gemma4:26b`** | **175** | 144 | 119 | **234** (2,5×) | 256k |
| `nemotron-3-nano:30b` | 142 | 134 | 124 | 132 (1,6×) | 1M, en las dos tarjetas |
| `ornith-1.5:35b` | 150 | 131 | 110 | 180 (2,3×) | 128k · 256k con `extended` |
| `ravenx` | 135 | 117 | 98 | 147 (1,9×) | 240k |
| `flint:40b` | 110 | 97 | 82 | 192 (2,6×) | 64k · 256k con `extended` |
| `gemma4:12b` | 115 | 96 | 84 | 201 (2,7×) | 256k |
| `qwen3-coder:30b` | 138 | 86 | 44 | 60 (2,5×) | 116k · 256k con `extended` |
| `qwen3:8b` | 104 | 69 | 41 | 130 (**3,4×**) | 40k |
| `qwen3.8:27b` | 52 | 63 | 50 | 87 (2,6×) | 96k · 256k con `extended` |
| `qwen3.8:27b-abliterated` | 60 | 58 | 42 | 78 (2,5×) | 16k · 224k con `extended` |
| `bonsai-2:27b` | 59 | 52 | 45 | 87 (2,0×) | 256k |
| `ravenx-chaos-agent` | 35 | 28 | 22 | 51 (2,5×) | 128k · 256k con `extended` |
| `qwen3:32b` | 30 | 21 | 16 | 36 (2,5×) | 30k · 40k con `dual-gpu` |

Cifras en tokens por segundo. "Concurrencia ×4" es el *throughput total* con
cuatro peticiones simultáneas, y el multiplicador respecto a una sola. Una
tirada por modelo, con el modelo ya cargado.

Respecto a la medición del 8 de noviembre, la actualización de llama.cpp sube
sobre todo a los modelos MoE (`nemotron-3-nano:30b` de 102 a 134 en el tamaño
medio, `ravenx` de 97 a 117) y a `qwen3:8b`, que pasa de 33 a 69. La mejora
grande, sin embargo, está en **leer el prompt**, que en esta tabla apenas se
nota: en los MoE se lee entre 2,7 y 4,7 veces más rápido. Se ve en la tabla de
documentos largos.

Los `-coder` y los qwen3 densos pierden mucha velocidad en las entradas
largas (`qwen3-coder:30b` va de 138 a 44). Para documentos grandes, mira la
tabla de abajo antes de elegir.

## Calidad de extracción

Extraer afirmaciones de una transcripción de 2.033 palabras y comparar contra
100 de referencia, por similitud coseno de embeddings.

- **Cobertura**: cuánto de la referencia se captura. Más alto = más completo.
- **Precisión**: cuán fieles son las afirmaciones extraídas. Más alto = menos
  invención.

| Modelo | Afirmaciones | Cobertura | Precisión |
|--------|-------------:|----------:|----------:|
| `ravenx` (razonando) | 72 | **0,617** | 0,602 |
| `qwen3-coder:30b` | 72 | 0,613 | 0,617 |
| `ravenx` (`off`) | 50 | 0,601 | **0,620** |
| `ravenx-chaos-agent` | 75 | 0,601 | 0,602 |
| `qwen3.8:27b` (`off`) | 74 | 0,581 | 0,562 |
| `ornith-1.5:35b` | 45 | 0,561 | 0,588 |
| `nemotron-3.5-lightning:30b` | 34 | 0,554 | 0,600 |
| `qwen3.8:27b` (`low`) | 34 | 0,550 | 0,603 |
| `nemotron-3-nano:30b` | 41 | 0,550 | 0,578 |
| `qwen3.8:27b` (`medium`) | 34 | 0,542 | 0,608 |
| `gemma4:26b` | 30 | 0,535 | 0,614 |
| `qwen3.8:27b-abliterated` | 36 | 0,535 | 0,595 |
| `qwen3:32b` | 31 | 0,533 | 0,587 |
| `bonsai-2:27b` | 41 | 0,527 | 0,564 |
| `flint:40b` | 29 | 0,523 | 0,589 |
| `gemma4:12b` | 30 | 0,507 | 0,566 |
| `qwen3:8b` | 23 | 0,488 | 0,605 |

Todos dieron JSON válido en esta prueba. En las de velocidad,
`nemotron-3-nano:30b`, `qwen3-coder:30b`, `ravenx` y `qwen3:8b` fallaron 1 de 4
tareas en algún tamaño: si dependes de JSON estricto, valida y reintenta.

Es **una tirada por modelo**: diferencias de ±0,03 entre dos modelos son ruido.
Lo que se repite entre mediciones es el patrón: los que sacan muchas
afirmaciones (`ravenx`, `qwen3-coder:30b`, `ravenx-chaos-agent`, `qwen3.8:27b`
con `off`) cubren más, y los que sacan pocas (`gemma4`, `flint:40b`) son más
conservadores. **Velocidad y calidad no van juntas**: `nemotron-3.5-lightning:30b`
es el más rápido y queda en mitad de tabla.

## Documentos largos

Resumen de transcripciones de 15.000 y 25.000 palabras, en tokens por segundo
de la petición completa (incluye leer el documento).

| Modelo | 15k palabras | 25k palabras |
|--------|-------------:|-------------:|
| **`nemotron-3.5-lightning:30b`** | **89** | **95** |
| `gemma4:26b` | 68 | 69 |
| `ornith-1.5:35b` | 56 | 60 |
| `flint:40b` | 47 | 53 |
| `ravenx` | 47 | 50 |
| `gemma4:12b` | 46 | 49 |
| `nemotron-3-nano:30b` | 30 | 54 |
| `bonsai-2:27b` | 24 | 26 |
| `qwen3.8:27b-abliterated` | 24 | 24 |
| `qwen3.8:27b` | 23 | 28 |
| `qwen3:8b` | 16 | no cabe |
| `ravenx-chaos-agent` | 9 | 12 |
| `qwen3-coder:30b` | 12 | 9 |
| `qwen3:32b` | 7 | no cabe |

**Un documento de 25.000 palabras son entre 41.000 y 45.000 tokens** según el
modelo, así que no entra en `qwen3:8b` (40k) ni en `qwen3:32b` (40k): recibirás
un `400` `context_length_exceeded`. Sí entra en los demás. A 15.000 palabras
(entre 24.700 y 26.800 tokens) entran todos.

Ojo con el tokenizador: el **mismo** texto de 25.000 palabras son 41.214 tokens
para `gemma4`, 41.916 para la familia Qwen 3.5 (`qwen3.8`, `ravenx`, `ornith`,
`bonsai`), 42.841 para `nemotron` y **44.793 para `qwen3-coder:30b`**, un 9 % más
que `gemma4`. Si vas al límite de la ventana, cuenta con el modelo concreto, no
con una regla de palabras.

## Latencia al primer token

Streaming, media de tres tiradas en caliente.

| Modelo | Nivel | TTFT |
|--------|:-----:|-----:|
| `gemma4:26b` | — | **0,80 s** |
| `nemotron-3.5-lightning:30b` | — | 0,82 s |
| `ornith-1.5:35b` | — | 0,83 s |
| `ravenx` | `low`/`medium` | 0,93 s |
| `gemma4:12b` | — | 1,02 s |
| `ravenx` | `off` | 1,05 s |
| `flint:40b` | — | 1,47 s |
| `qwen3:8b` | — | 1,51 s |
| `nemotron-3-nano:30b` | — | 1,58 s |
| `qwen3-coder:30b` | — | 1,68 s |
| `bonsai-2:27b` | — | 1,72 s |
| `qwen3.8:27b` | `low` | 1,76 s |
| `qwen3.8:27b` | `off` | 1,83 s |
| `qwen3.8:27b` | `medium` | 1,93 s |
| `qwen3.8:27b-abliterated` | — | 2,07 s |
| `ravenx-chaos-agent` | — | 2,17 s |
| `qwen3:32b` | — | **3,89 s** |

Con el llama.cpp actual **el nivel de razonamiento apenas cambia el TTFT**: la
diferencia entre niveles es de décimas. El 8 de noviembre, `off` lo bajaba casi
a la mitad.

`qwen3:32b` sigue siendo el más lento en arrancar, aunque ha bajado de 22,7 a
3,9 s. No lo uses para nada interactivo.

## Razonamiento

Cómo responde cada plantilla al nivel:

| Plantilla | Modelos | Qué hace `reasoning_effort` |
|-----------|---------|-----------------------------|
| Lee el nivel | `qwen3.8:27b`, `qwen3.8:27b-abliterated`, `bonsai-2:27b`, `ravenx-chaos-agent` | Gradúa el razonamiento. Aceptan `low`, `medium` y `xhigh`; oCabra traduce el `high` de OpenAI a `xhigh` por ti |
| Solo interruptor | `gemma4:26b`, `gemma4:12b`, `ornith-1.5:35b`, `nemotron-3.5-lightning:30b`, `nemotron-3-nano:30b`, `qwen3:8b`, `qwen3:32b`, `ravenx` | `off` apaga el pensamiento; `low`, `medium` y `high` la plantilla los trata igual |
| Ni nivel ni interruptor | `flint:40b` | Piensa siempre, también con `off` |
| No razona | `qwen3-coder:30b` | Nada |

**Para que el nivel signifique algo en todos, oCabra lo convierte en un tope
duro de tokens de pensamiento.** Cuando se agota, llama.cpp corta el
pensamiento y obliga a responder:

| Nivel | Tope de pensamiento |
|-------|--------------------:|
| `low` | 1.000 tokens |
| `medium` | 4.096 tokens |
| `high` | Sin tope propio |
| Sin `reasoning_effort` | 16.384 tokens |

En los modelos de solo interruptor y en `flint:40b`, **ese tope es la única
diferencia entre `low`, `medium` y `high`**. En los que leen el nivel se suma a
lo que ya hace la plantilla. Si no mandas nivel se aplica un tope holgado, para
no recortar el uso normal: `ravenx` piensa unos 13.000 tokens en el 10 % de sus
respuestas más largas. Si mandas tu propio `reasoning_budget_tokens`, manda el
tuyo.

Qué conviene pedir, medido:

| Modelo | Recomendación |
|--------|---------------|
| `ravenx` | Razonando si quieres exhaustividad (72 afirmaciones, cobertura 0,617); `off` si quieres precisión (0,620). Medido antes de los topes de la 0.69.0, cuando `low` y `medium` daban lo mismo. |
| `qwen3.8:27b` | `off` si quieres exhaustividad (74 afirmaciones), `medium` si quieres precisión (0,608). |

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
con `max_tokens: 4096` el modelo piensa como mucho 3.072 tokens, aunque el
nivel permitiera más.

## Visión

Aceptan imágenes (`image_url` en las content parts) `gemma4:26b`,
`gemma4:12b`, `qwen3.8:27b`, `qwen3.8:27b-abliterated` y `ornith-1.5:35b`. El
proyector visual se carga **cuando llega la primera imagen**, no al arrancar:
esa primera petición tarda unos segundos más.

## Disponibilidad

| Modelo | Nota |
|--------|------|
| `gemma4:26b` | Fijado en la 3090, **pero su pin es evictable**: ocupa casi toda la tarjeta, así que cuando otro modelo grande se la queda, sale. Vuelve solo en cuanto queda sitio (puede tardar un par de minutos), o antes si le llega una petición, que paga la carga. Si tu carga alterna modelos grandes, usa `recomendado-ha-chatbots`. |
| `qwen3.8:27b` | Se mantiene caliente. |
| `qwen3-embedding:8b` | Se mantiene caliente. |
| `whisper-large-v3-turbo` | Fijado en la 3060, así que transcribir no compite por la 3090. Si hace falta sitio en la 3060 y está ocioso, puede salir antes que un modelo grande, porque vuelve en unos 10 s. |
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
| Chat interactivo | `gemma4:26b` | TTFT 0,80 s y suele estar cargado |
| Máxima velocidad | `gemma4:26b` o `nemotron-3.5-lightning:30b` | 144 y 148 tok/s en tamaño medio; `gemma4:26b` cabe en una sola tarjeta y es el más rápido en entradas cortas (175) |
| Documentos largos | `nemotron-3.5-lightning:30b` o `gemma4:26b` | 95 y 69 tok/s con 25.000 palabras |
| Extracción exhaustiva | `ravenx` razonando o `qwen3-coder:30b` | 72 afirmaciones, cobertura ~0,615 |
| Extracción fiel | `ravenx` + `off` o `gemma4:26b` | Precisión 0,620 y 0,614 |
| Volumen / lotes | `gemma4:26b`, `nemotron-3.5-lightning:30b` o `gemma4:12b` | 200-234 tok/s con 4 en paralelo |
| Razonamiento visible | `qwen3.8:27b` + `medium` | Es uno de los tres que distinguen niveles de verdad |
| Salida estructurada | cualquiera, validando | Todos dieron JSON válido en extracción; algunos fallan 1 de 4 tareas |
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
