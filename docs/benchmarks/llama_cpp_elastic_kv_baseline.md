# Baseline llama.cpp: KV elástica por modelo y GPU

**Apertura:** 2026-10-30
**Estado:** smoke 4×32k y visión bajo demanda completados en los tres modelos; estabilidad larga pendiente
**Plan:** `docs/tasks/llama-cpp-elastic-kv-plan.md`

Este registro conserva todos los resultados, incluidos fallos y configuraciones
que solo arrancan pero no soportan prefill o generación sostenida. El objetivo es
encontrar el mayor contexto estable de Gemma 4 12B, Gemma 4 26B-A4B y Qwen3.8
27B en RTX 3060, RTX 3090 y ambas tarjetas.

El barrido completo no bloquea el desarrollo de KV elástica. Las primeras
pruebas usarán capacidades conservadoras conocidas; cada fila se completará
cuando una ventana libre o una decisión técnica justifique la prueba.

## Hardware y build inicial

| Elemento | Valor |
|---|---|
| GPU 0 | NVIDIA GeForce RTX 3060, 12.288 MiB |
| GPU 1 | NVIDIA GeForce RTX 3090, 24.576 MiB |
| llama.cpp | `0.7.0-dev`, build 1, commit `74bca39` |
| KV objetivo | `q8_0` para K y V |
| Flash Attention | activado |
| Slots objetivo inicial | 4 |
| Contexto nativo de los tres modelos | 262.144 tokens |

## Artefactos disponibles

No hace falta descargar los modelos de la primera campaña. Se usarán los blobs
que ya consume oCabra para evitar diferencias de cuantización entre el baseline
y la futura implementación.

| Modelo | Blob de pesos | Bytes | Projector | Draft/MTP |
|---|---|---:|---|---|
| Gemma 4 12B | `sha256-bb722270...62262726215391` | 7.381.383.680 | `sha256-0cb0cd0a...e6d28` | `sha256-7008a656...6594f` |
| Gemma 4 26B-A4B | `sha256-a8bdd07a...5eede2` | 17.074.419.072 | `sha256-7e78b02d...b8556d` | `sha256-491f38af...f913e32` |
| Qwen3.8 27B | `sha256-f5f1dd89...2ad57d` | 16.810.714.464 | `sha256-ac3714bf...6448e` | No incluido en el manifiesto actual |

Los hashes abreviados son solo etiquetas legibles. Los comandos reproducibles
de cada ejecución deben guardar el digest completo.

### Coste de visión

| Modelo | Projector en disco | VRAM adicional aislada | Evidencia |
|---|---:|---:|---|
| Gemma 4 12B | 175.115.616 B | 408 MiB observados | Activación en caliente: 10.211 -> 10.619 MiB con pool unificado 128k/4 y KV q8 |
| Gemma 4 26B-A4B | 1.194.828.128 B | 1.282 MiB observados | Activación en caliente: 19.733 -> 21.015 MiB con el mismo worker 128k/2 y KV q8 |
| Qwen3.8 27B | 931.146.016 B | 1.210-1.216 MiB observados | Activación en caliente: 22.171 -> 23.381/23.387 MiB con pool unificado 96k/4 y KV q8 |

El tamaño del fichero orienta, pero no sustituye la medición A/B: tensores,
buffers de cómputo y backend pueden hacer que el coste residente difiera.
En Qwen, los 887,99 MiB vistos durante los OOM eran una reserva concreta de
CUDA, no el coste residente completo de habilitar visión.

### Visión bajo demanda en Gemma 4 26B-A4B

Prueba end-to-end del 2026-10-31 con el runtime productivo de 131.072 tokens,
dos slots, KV `q8_0`, MTP y RTX 3090:

| Caso | Resultado | Tiempo/VRAM |
|---|---|---|
| `enabled` | Texto e imagen correctos; `/props` anunció visión | Projector cargado al arrancar |
| `disabled` | Texto correcto; imagen rechazada limpiamente con HTTP 400 `model_not_capable` | 13 ms hasta el rechazo |
| `on_demand`, primera imagen | El worker arrancó con `--props`, sin `--mmproj`, activó el projector y reconoció las máscaras de comedia y tragedia | 2,343 s con worker ya cargado; 19.733 -> 21.015 MiB |
| Imagen mientras un slot generaba texto | Esperó al drenado, activó visión después de terminar el texto y no corrompió ninguna petición | Texto 13,545 s; imagen 7,605 s |
| Dos primeras imágenes simultáneas desde frío | Ambas devolvieron HTTP 200 y el contenido correcto; se registró una sola activación | 11,634 s y 11,788 s, incluyendo carga fría del worker |

El incremento de 1.282 MiB es la diferencia residente observada alrededor de
la activación, no solo el tamaño de los tensores del projector. Incluye los
buffers que el backend multimodal reserva al habilitarlo. La latencia de 2,343
s incluye prompt y generación; los timings internos fueron 390 ms de prompt y
1,15 s de generación, por lo que la transición fue inferior a un segundo en
esa ejecución.

Durante la prueba apareció una regresión de integración: un runtime derivado
descargado conservaba el `vision_mode` anterior aunque cambiase la configuración
del modelo base. Se corrigió refrescando la configuración mutable antes de
recargar un derivado que no esté `loaded` ni `loading`, se añadió un test y se
repitió la transición real sin reiniciar la API. Los runtimes ya cargados no se
mutan bajo peticiones activas.

### Visión bajo demanda en Gemma 4 12B y Qwen3.8 27B

Se repitió la transición real con cuatro slots y KV unificada en la RTX 3090:

| Modelo | Pool | Activación concurrente desde frío | VRAM | Imágenes concurrentes |
|---|---:|---:|---:|---|
| Gemma 4 12B | 128k | Dos `POST /props` en 0,343 s; una sola carga del projector | 10.211 -> 10.619 MiB | Dos respuestas correctas en 2,760/2,977 s, slots 0 y 1 |
| Qwen3.8 27B | 96k | Dos `POST /props` en 0,716/0,717 s; una sola carga del projector | 22.171 -> 23.387 MiB | Dos respuestas correctas en 5,417/5,415 s, slots 0 y 1 |

Antes de activar el projector, `llama-server` rechazó directamente la imagen
con HTTP 500 por modalidad no soportada. Es un detalle de la API interna del
fork: oCabra intercepta ese caso y en política `disabled` devuelve el contrato
público HTTP 400 `model_not_capable`.

Ambos pares reconocieron correctamente las máscaras de comedia y tragedia. En
Qwen aparecieron avisos de posiciones no consecutivas con visión y KV unificada,
además del aviso genérico del projector que recomienda un mínimo de 1.024 tokens
visuales para grounding. No hubo corrupción en este smoke, pero faltaba probar
una imagen grande y grounding antes de considerar esta combinación estable.

El primer intento de Qwen a 96k coincidió con Whisper ocupando 2.336 MiB de la
3090 y falló al pedir 384 MiB para MTP. No se contabiliza como límite de Qwen:
al aislar la GPU, la misma configuración cargó y superó texto, activación de
visión e imágenes concurrentes.

#### Grounding Qwen3.8 y tamaño de ubatch

Se añadió la fixture reproducible
`docs/benchmarks/qwen_vision_grounding_fixture.svg`: una lámina 1280×960 con
seis objetos, etiquetas, ruta, código y una estrella D4 cuyo centro real
normalizado es aproximadamente `[160, 560]`.

| Configuración | Resultado | Tiempo | VRAM tras petición |
|---|---|---:|---:|
| 96k/4, `ubatch=128`, sin thinking | Todos los textos correctos; D4 `[160,555]` | 2,739 s | 23.445 MiB, 733 MiB libres |
| 96k/4, `ubatch=1024`, sin thinking | Todos los textos correctos; D4 `[158,555]` | 3,442 s | 24.059 MiB, 119 MiB libres |
| 96k/4, `ubatch=128`, dos imágenes simultáneas | Slots 0 y 1; ambas completas y D4 `[156,556]` | 5,748/5,695 s | 23.451 MiB, 727 MiB libres |

La imagen grande produjo 1.281 tokens de prompt incluso con `ubatch=128`, por
lo que el aviso de 1.024 tokens no implica por sí solo que esta entrada haya
sido recortada. No apareció el log `cap image_max_tokens`. Subir `ubatch` a
1.024 aumentó el residente base en unos 632 MiB, mantuvo los avisos de
posiciones no consecutivas, fue algo más lento y dejó un margen inaceptable.
Se conserva `ubatch=128` para el candidato 96k.

Con thinking activado, el modelo leyó todos los campos pero entró en una
deliberación repetitiva sobre dimensiones inexistentes y agotó tanto 350 como
900 tokens sin emitir contenido final. Con
`chat_template_kwargs.enable_thinking=false` respondió correctamente. Esto es
una decisión de perfil/petición para tareas visuales estructuradas, no un fallo
del pool KV, pero debe cubrirse en la integración de oCabra.

### Soak corto mixto y cancelación

Prueba del 2026-10-31 con prompt cache en RAM desactivada y sin guardado de
slots. No sustituye al soak de 12 horas, pero fuerza reutilización repetida del
pool, los cuatro slots, visión y desconexiones abruptas.

| Modelo/configuración | Carga mixta | Cancelación y recuperación | Resultado interno |
|---|---|---|---|
| Qwen3.8, 96k/4, KV q8, self-MTP, visión | 10 tandas × 4: dos textos identificables y dos imágenes 1280×960; 40/40 correctas | Cuatro streams largos desconectados a la vez; slots libres en el segundo sondeo (<1 s); recuperación 4/4 en 0,69-0,76 s | 0 OOM, 0 reallocs; ~725 MiB libres al final |
| Gemma 4 12B, 128k/4, KV q8, draft MTP, visión | 5 tandas × 4: 20/20 correctas | Mismo caso; slots libres en el segundo sondeo (<1 s); recuperación 4/4 en 0,38 s | 0 OOM, 0 reallocs; ~13,5 GiB libres al final |

En Qwen, la primera tanda fría tardó 4,04-4,58 s; las siguientes quedaron en
0,70-0,83 s para texto y 1,03-1,15 s para visión repetida. Gemma 12B quedó en
0,38-0,90 s para texto y 1,79-2,38 s para visión. Cada respuesta textual llevaba
un marcador distinto y cada respuesta visual debía contener batch, prioridad y
código; no se detectó contaminación cruzada.

#### Soak acelerado de 1.000 peticiones Qwen3.8

Sobre el mismo candidato 96k/4 se ejecutaron 250 tandas llenando siempre los
cuatro slots: 500 textos con marcador único y 500 imágenes 1280×960 validadas
por batch, prioridad y código. Resultado:

- 1.000/1.000 respuestas correctas en 237 s, sin tráfico externo concurrente;
- texto: media 0,799/0,816 s, p95 0,850/0,859 s por los dos slots textuales;
- visión: media 0,920 s, p95 0,930/0,933 s;
- máximos 4,23-4,50 s correspondientes a la primera tanda fría;
- VRAM 23.459 MiB al final frente a 23.453 MiB en el soak corto: crecimiento
  de unos 6 MiB, sin tendencia apreciable;
- cero OOM, cero reallocs de grafo y cero respuestas cruzadas;
- 44 avisos de posiciones no consecutivas durante procesamiento multimodal,
  sin error observable asociado.

Después de las 1.000 respuestas se desconectaron cuatro streams largos a la
vez. Los slots drenaron en 375 ms y cuatro peticiones inmediatas de recuperación
respondieron correctamente en 0,75-0,81 s. Este hito cubre reutilización intensa
de una imagen y marcadores textuales variables; aún no sustituye el soak de 12
horas con imágenes, longitudes y patrones de llegada diversos.

También se corrigió la carrera operativa de descarga->carga manual inmediata.
La descarga captura la VRAM libre anterior y espera una muestra nueva del
monitor, acotada a las GPU del runtime y a un incremento demostrable de hasta
1 GiB. En la repetición real, la descarga devolvió en 1,201 s y la carga
inmediata entró a la primera con HTTP 200, sin el 409 falso anterior.

## Estado observado antes de probar

Instantánea del 2026-10-30 alrededor de las 19:40 UTC:

| GPU | VRAM usada | Proceso principal | Configuración observada |
|---|---:|---|---|
| 3060 | 8.880 MiB | Ollama y Whisper | embedding con contexto 40.960 y KV q8 |
| 3090 | 21.007 MiB | llama.cpp propio | Gemma 4 26B, pool 131.072, 2 slots, KV q8, MTP y mmproj |

Aunque la utilización de cómputo era 0 %, hubo tráfico periódico de
`recomendado-ha-chatbots` y Whisper durante los 30 minutos anteriores. No se
desalojaron runtimes ni se inició un barrido destructivo.

## Matriz de capacidad

Estados permitidos: `pendiente`, `no viable full-offload`, `arranca`,
`prefill OK`, `estable` y `fallo`. Una configuración solo es `estable` tras la
carga sostenida definida en el plan.

| Modelo | Colocación | 1 slot máximo | 4 slots pool máximo | Estado | Evidencia |
|---|---|---:|---:|---|---|
| Gemma 4 12B | 3060 | - | - | pendiente | - |
| Gemma 4 12B | 3090 | - | >= 131.072 (no acotado) | arranca | 4 peticiones concurrentes aisladas; 13.613 MiB libres tras el smoke |
| Gemma 4 12B | 3060+3090 | - | - | pendiente | - |
| Gemma 4 26B-A4B | 3060 | - | - | pendiente | - |
| Gemma 4 26B-A4B | 3090 | - | >= 131.072 (no acotado) | arranca | 4 peticiones concurrentes aisladas; 3.013 MiB libres tras el smoke |
| Gemma 4 26B-A4B | 3060+3090 | - | - | pendiente | - |
| Qwen3.8 27B | 3060 | - | - | pendiente | - |
| Qwen3.8 27B | 3090 | - | 143.360 sin visión; 122.880 con visión degradado | arranca | Sin visión 140k sirve 4 prompts y 144k falla; con visión 120k sirve corto con fallo de reserva, 124k aborta |
| Qwen3.8 27B | 3060+3090 | - | - | pendiente | - |

## Protocolo por fila

1. Confirmar ventana sin peticiones activas ni tráfico reciente.
2. Guardar procesos, argumentos, VRAM y runtimes que habrá que restaurar.
3. Cargar el modelo con un slot y comenzar en 32k.
4. Subir por 64k, 96k, 128k, 160k, 192k, 224k y 256k hasta el primer fallo.
5. Acotar el máximo entre el último éxito y el primer fallo cuando aporte
   capacidad útil.
6. En cada candidato ejecutar arranque, prompt corto, prefill largo y generación
   sostenida, guardando pico y margen mínimo de VRAM.
7. Repetir el máximo candidato con cuatro slots y KV unificada.
8. En 3060+3090 guardar `tensor_split`, capas y VRAM de cada GPU.
9. Restaurar los servicios originales y verificar una petición de salud.

No se desalojará tráfico productivo solo para completar una fila pendiente.

## Ejecuciones

| Fecha UTC | Modelo | GPU | Contexto/pool | Slots | Flags diferenciales | Resultado | VRAM pico | Evidencia |
|---|---|---|---:|---:|---|---|---:|---|
| 2026-10-30 | Gemma 4 26B-A4B | 3090 | 131.072 | 2 | KV q8, MTP, mmproj, sin split | Runtime productivo observado; no cuenta aún como prueba controlada | 20.984 MiB del proceso | argumentos del proceso |
| 2026-10-30 | Gemma 4 12B | 3090 | 131.072 | 4 | KV unificada, 32k/slot, KV q8, draft MTP, mmproj, sin split | Carga en 19,3 s; `/props` y `/slots` correctos; 4 prompts identificables usaron slots 0–3 sin contaminación | 10.565 MiB | 0,52–0,60 s por petición corta; aceptación draft 0,75–0,78 |
| 2026-10-30 | Gemma 4 26B-A4B | 3090 | 131.072 | 4 | KV unificada, 32k/slot, KV q8, draft MTP, mmproj, sin split | Carga caliente en 6,77 s; `/props` y `/slots` correctos; 4 prompts identificables usaron slots 0–3 sin contaminación | 21.165 MiB | 0,68–0,72 s por petición corta; aceptación draft 0,70–1,00; cero reallocs de grafo |
| 2026-10-30 | Qwen3.8 27B | 3090 | 131.072 | 4 | KV unificada, 32k/slot, KV q8, self-MTP, mmproj, sin split | Fallo de carga tras 39,5 s: el projector pidió 887,99 MiB adicionales y CUDA devolvió OOM | no capturado | El proceso abortó antes de servir; próximo escalón: pool 96k con 4 slots |
| 2026-10-31 | Gemma 4 12B | 3090 | 131.072 | 4 | KV unificada sin cap por slot, KV q8, draft MTP, mmproj, sin split | Los cuatro slots anunciaron 131.072 tokens cada uno; 4 prompts aislados usaron slots 0–3 | 10.565 MiB | Misma VRAM que con cap 32k: confirma un solo pool físico, no cuatro reservas de 128k |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | KV unificada, KV q8, self-MTP, mmproj | Carga limpia en 37,25 s con page cache inicialmente fría | 23.309 MiB | 869 MiB libres; candidato conservador con visión |
| 2026-10-31 | Qwen3.8 27B | 3090 | 114.688 | 4 | Igual, con mmproj | Carga limpia en 5,58 s caliente | 23.989 MiB | 189 MiB libres; margen insuficiente para producción |
| 2026-10-31 | Qwen3.8 27B | 3090 | 118.784 | 4 | Igual, con mmproj | Carga limpia en 5,58 s caliente | 24.161 MiB | 17 MiB libres; máximo de carga limpia observado, no seguro |
| 2026-10-31 | Qwen3.8 27B | 3090 | 122.880 | 4 | Igual, con mmproj | Health y petición corta correctos, pero falló una reserva CUDA de 248,10 MiB durante la carga | 24.117 MiB tras petición | Arranque degradado; no aceptable |
| 2026-10-31 | Qwen3.8 27B | 3090 | 126.976 | 4 | Igual, con mmproj | Abortó al reservar los 887,99 MiB del projector | no aplica | Primer fallo duro del barrido fino con visión; 131.072 falla igual |
| 2026-10-31 | Qwen3.8 27B | 3090 | 131.072 | 4 | KV unificada, KV q8, self-MTP, sin mmproj | Carga limpia en 5,11 s caliente | 23.531 MiB | 647 MiB libres; candidato conservador sin visión |
| 2026-10-31 | Qwen3.8 27B | 3090 | 143.360 | 4 | Igual, sin mmproj | Carga limpia; cuatro peticiones concurrentes usaron slots aislados y respondieron correctamente | 24.135 MiB tras peticiones | 43 MiB libres; máximo técnico corto, no seguro |
| 2026-10-31 | Qwen3.8 27B | 3090 | 147.456 | 4 | Igual, sin mmproj | Fallo al crear el contexto MTP: faltaron 53 MiB para compute PP | no aplica | Primer fallo duro sin visión |
| 2026-10-31 | Gemma 4 12B | 3090 | 131.072 | 4 | KV unificada, KV q8, draft MTP, visión bajo demanda | Dos activaciones concurrentes se serializaron en una; dos imágenes simultáneas fueron correctas en slots distintos | 10.619 MiB | +408 MiB; activación 0,343 s; imágenes 2,760/2,977 s |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | KV unificada, KV q8, self-MTP, visión bajo demanda | Dos activaciones concurrentes se serializaron en una; dos imágenes simultáneas fueron correctas en slots distintos | 23.387 MiB | +1.216 MiB; activación 0,716/0,717 s; imágenes 5,417/5,415 s |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | Carga mixta y cancelación, KV unificada, visión | 40/40 mixtas y 4/4 recuperación tras cancelar cuatro streams | 23.453 MiB | Sin OOM ni reallocs; ~725 MiB libres |
| 2026-10-31 | Gemma 4 12B | 3090 | 131.072 | 4 | Carga mixta y cancelación, KV unificada, visión | 20/20 mixtas y 4/4 recuperación tras cancelar cuatro streams | 10.611 MiB | Sin OOM ni reallocs; ~13,5 GiB libres |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | Soak acelerado: 500 textos únicos + 500 imágenes, cancelación final | 1.000/1.000 correctas; drenado final en 375 ms y recuperación 4/4 | 23.459 MiB | p95 texto <=0,859 s, visión <=0,933 s; sin OOM/reallocs |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 1-6 | Barrido de slots con KV q8, self-MTP y visión bajo demanda | 1-5 slots correctos; con 6 slots la activación del projector abortó el proceso tras OOM | 21.516-23.912 MiB con 1-5 slots | Coste aproximado de 598 MiB por slot; 4 es el compromiso y 5 carece de margen operativo |
| 2026-10-31 | Gemma 4 12B | 3090 | 131.072 | 1-6 | Barrido de slots con KV q8, draft MTP y visión bajo demanda | Todos los niveles respondieron correctamente | 10.038-10.894 MiB | Coste aproximado de 171 MiB por slot; 6 slots aún dejan margen amplio |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | Prefills desiguales 40k+20k+10k+5k | 4/4 correctos en paralelo; pared 99,15 s | 23,3 GiB aprox. | 75.048 tokens agregados; confirma reparto desigual dentro del pool |
| 2026-10-31 | Qwen3.8 27B | 3090 | 98.304 | 4 | Sobresuscripción 4x28k | 4/4 correctos, pero serializados con purgas KV; pared 224,11 s | 23,3 GiB aprox. | Reintentos, batch menor y `purging slot`; requiere admission control ponderado |
| 2026-10-31 | Gemma 4 12B | 3090 | 262.144 | 4 | KV q8, draft MTP, visión, pool unificado | Prefill real de 200.032 tokens y respuesta exacta en 195,99 s | 12.214 MiB | Máximo nativo validado, no solo arranque |
| 2026-10-31 | Gemma 4 26B-A4B | 3090 | 262.144 | 4 | KV q8, draft MTP, visión, pool unificado | 4/4 cortas en ~0,50 s; prefill de 200.030 tokens correcto en 131,29 s | 23.614 MiB | Máximo nativo viable, pero solo 564 MiB libres |
| 2026-10-31 | Gemma 4 12B | 3060 | 131.072 | 4 | Full offload, KV q8, draft MTP y visión | 4/4 cortas en ~0,55 s; prefill de 100.029 tokens correcto en 168,53 s | 10.749 MiB | 1.209 MiB libres; perfil operativo viable en 12 GiB |
| 2026-10-31 | Qwen3.8 27B | 3060+3090 | 131.072 | 4 | `split-mode layer`, `tensor-split 1,2`, `main-gpu 1`, KV q8, self-MTP y visión | 4/4 cortas en ~1,04 s; imagen correcta en 3,74 s; prefill de 60.022 tokens correcto en 76,81 s | 9.431 + 16.678 MiB | El projector se alojó en la 3060; quedaron 2.527 + 7.500 MiB libres |
| 2026-10-31 | Gemma 4 26B-A4B | 3090 | 131.072 | 4 | Canary oCabra WeightedSlotGate, KV q8, draft MTP y visión | A=89.530; B=50.026 en cola solo por KV; C=20.026 de otro usuario usó el remanente; cancelación y drenado exactos; oversized 140.022 rechazado | 21.186 MiB aprox. | Evento `kv_queued`, snapshots lógicos 109.556/131.072, HTTP 400 y recuperación HTTP 200; producción restaurada a 2 slots sin KV unificada |
| 2026-10-31 | Gemma 4 12B | 3090 | 131.072 | 4 | Producción custom, q8, MTP externo, visión `on_demand` | Texto y primera imagen HTTP 200; hot-mmproj correcto | 10.600 MiB con visión | Perfil principal `gemma4:12b` |
| 2026-10-31 | Qwen3 Coder 30B | 3090 | 65.536 | 4 | Producción custom, q8, pool unificado | Respuesta `323`, HTTP 200 | 21.288 MiB | Perfil principal `qwen3-coder:30b` |
| 2026-10-31 | Qwen3.8 27B abliterated | 3060+3090 | 65.536 | 4 | Q5, self-MTP, visión `on_demand`, spill | Texto e imagen HTTP 200 | 9.431 + 17.578 MiB con visión | Perfil principal abliterated; requiere liberar auxiliares para cargar |
| 2026-10-31 | Qwen3.8 27B abliterated | 3060+3090 | 131.072 | 4 | Q5, self-MTP, KV q8, `main_gpu=1`, `tensor_split=1,2`, visión caliente | Texto exacto `ABLITERATED_128K_OK`; projector activado y grounding real `D4` correcto | 10.409 + 19.704 MiB con visión | El reparto sin `main_gpu=1` agotó la 3060 al reservar el buffer de cómputo; la configuración corregida dejó 1.549 + 4.474 MiB libres |
| 2026-10-31 | RavenX v6.2 | 3090 | 131.072 | 4 | Q4, q8 KV, pool unificado | `RAVENX_OK`, HTTP 200 | 22.160 MiB | Perfil principal `ravenx`; alias `ravenx-256k` |
| 2026-11-01 | Gemma 4 26B-A4B | adaptativo 3090 → 3060+3090 → 3090 | 131.072 → 262.144 → 131.072 | 4 → 1 → 4 | Selección por presupuesto, q8, split 1:2, histéresis | `NORMAL_OK`, `MAX_DUAL_OK` y `DOWNGRADE_TEST_OK`; downgrade autónomo reconstruyó default | 8.257 + 16.148 MiB en dual; 19.954 MiB normal | Petición de 140.030 tokens eligió `max`; proceso dual `--tensor-split 1,2 --parallel 1`; retorno `--split-mode none --parallel 4` |
| 2026-11-01 | Gemma 4 26B-A4B | 3090 | 262.144 | 4 | Runtime canónico, KV q8 unificada sin cap por slot, MTP, visión bajo demanda | `/props` anunció pool 262.144; los cuatro `/slots` anunciaron 262.144; chat público HTTP 200 con fingerprint `b1-74bca39` | 22.402 MiB tras carga | Sustituye el experimento adaptativo: una sola reserva física, cuatro consumidores lógicos y gate ponderado en oCabra |
| 2026-11-01 | Gemma 4 26B-A4B | 3090 | 262.144 | 4 | Restore productivo final, KV q8 unificada, MTP externo, visión bajo demanda | Proceso real con `--ctx-size 262144 --parallel 4 --kv-unified`; `/props` y los cuatro `/slots` devolvieron 262.144; chat directo respondió `OK` | 22.800 MiB de GPU, reserva oCabra 23.632 MiB | GPU0 permaneció disponible para Whisper y `qwen3-embedding:4b`; no se interrumpió esa carga real |
| 2026-11-01 | RavenX v6.2 | 3090 | 245.760 | 4 | Q4, KV q8 unificada, sin MTP/visión | `/props` 240k/4 y respuesta exacta `RAVEN_240K_OK` | 23.602 MiB; 576 MiB libres | 256k fue rechazado por 74 MiB respecto al margen del scheduler; 240k es el máximo operativo de una tarjeta |
| 2026-11-01 | Qwen3 Coder 30B | 3060+3090 | 262.144 | 4 | KV q8 unificada, split 1:3, sin MTP/visión | Contexto nativo completo; respuestas `CODER_256K_OK` y `CODER_SPLIT_OK` | 9.605 + 23.398 MiB; 2.353 + 780 MiB libres | Split 1:2 funcionó pero dejaba solo 63 MiB en GPU0; 1:3 es el perfil operativo |
| 2026-11-01 | Qwen3.8 27B | 3060+3090 | 262.144 | 4 | KV q8 unificada, self-MTP, split 1:3, visión caliente | Texto `QWEN_256_VISION_OK`; projector activo y grounding `A1` correcto | 9.859 + 23.228 MiB; 2.099 + 950 MiB libres tras imagen | Contexto nativo completo; split 1:2 dejaba solo 305 MiB en GPU0 con visión |
| 2026-11-01 | Qwen3.8 27B abliterated | 3060+3090 | 229.376 | 4 | Q5, KV q8 unificada, self-MTP, split 1:2,3, visión caliente | Texto `ABLIT_224_SPLIT_OK`; grounding `A1` correcto | 11.283 + 23.616 MiB; 675 + 562 MiB libres tras imagen | 240k/1:2 sirvió texto y activó visión, pero dejó 47 MiB en GPU0; 256k no pasó admisión |
| 2026-11-01 | Qwen3.8 27B abliterated | adaptativo 3090 → 3060+3090 → 3090 | 65.536 → 229.376 → 65.536 | 1 → 4 → 1 | Presets de colocación, KV q8 unificada, visión bajo demanda, split dual 1:2,3 | Petición extendida e imagen HTTP 200; downgrade autónomo correcto tras la gracia de prueba | Dual validado con 675 + 562 MiB libres | La transición drenó y desalojó vecinos; la vuelta liberó íntegramente GPU0. La gracia productiva se restauró a 300 s y la residencia mínima a 120 s |
| 2026-11-01 | Qwen3 Coder 30B | 3090 | 118.784 | 4 | KV q8 unificada, sin MTP/visión, full offload | Respuesta exacta `323`, HTTP 200 | 24.058 MiB; 120 MiB libres | 114.688 dejó 334 MiB libres; 122.880 falló al reservar 285,08 MiB para el grafo CUDA. 116k es el máximo operativo validado de una tarjeta |
| 2026-11-01 | Qwen3.8 27B | 3090 | 98.304 | 4 | KV q8 unificada, self-MTP, visión bajo demanda, VRAM calibrada 24.000 MiB | Carga nativa y chat HTTP 200 tras desalojar el fallback Ollama | 22.810 MiB usados; 1.368 MiB libres | La estimación genérica de 24.884 MiB superaba en 725 MiB la capacidad observable y producía un falso 409 antes de arrancar; `validated_vram_mb` se aplica solo junto al contexto exacto validado |
| 2026-11-01 | RavenX Chaos Agent Qwen3.8 27B | 3090 | 98.304 | 4 | Q4_K_M, KV q8 unificada, texto, sin MTP ni visión | Respuesta exacta `CHAOS_OK`; cuatro peticiones concurrentes devolvieron sus cuatro marcadores correctos | 19.708 MiB del proceso; calibración operativa 20.000 MiB | El GGUF no contiene capas MTP: activarlo aborta la carga. Rendimiento corto observado: 40,9 tok/s |
| 2026-11-01 | RavenX Chaos Agent Qwen3.8 27B | 3060+3090 | 262.144 | 4 | Q4_K_M, KV q8 unificada, split 1:3, texto | Benchmark de preset HTTP 200, carga fría 27,1 s | 7.634 + 19.846 MiB; calibración conservadora 30.536 MiB agregados | Contexto nativo completo; 23,3 tok/s en el smoke. El perfil vuelve a 96k/4 en una 3090 para peticiones normales |
| 2026-11-01 | RavenX OpenFable Coder Gemma 4 12B | 3060 | 114.688 | 4 | Q4_K_M, KV q8 unificada, texto, sin MTP ni visión | Chat HTTP 200 y cuatro peticiones concurrentes HTTP 200 | 9.114 MiB del proceso; reserva oCabra calibrada 11.066 MiB | 131k excedió en 224 MiB el margen de admisión de la 3060; 112k es el perfil normal estable |
| 2026-11-01 | RavenX OpenFable Coder Gemma 4 12B | 3090 | 131.072 | 4 | Q4_K_M, KV q8 unificada, texto | Benchmark de preset HTTP 200, carga fría 7,2 s | Reserva oCabra 11.403 MiB | Contexto nativo completo; 61,0 tok/s en el smoke. El modelo antepone de forma persistente su identidad OpenFable y falla instrucciones de salida exacta, por lo que queda experimental |

### Límite empírico Qwen3.8 en la 3090

Con cuatro slots, KV q8, self-MTP, full offload y un solo pool unificado:

| Configuración | Máximo técnico observado | Primer fallo duro | Candidato con margen |
|---|---:|---:|---:|
| Con visión cargada | 120k sirve una petición corta, pero con reserva CUDA fallida; 116k es la mayor carga limpia | 124k | 96k |
| Sin visión | 140k sirve cuatro peticiones cortas | 144k | 128k |

El perfil Qwen recomendado en una sola 3090 sigue siendo 96k/4 con visión o
128k/4 sin ella. El primero superó 1.000 peticiones mixtas, cancelaciones y una
carga desigual de 75k tokens agregados. La sobresuscripción deliberada de 112k
sobre el pool de 96k no falló, pero llama.cpp serializó trabajo y purgó KV con
latencia impredecible; el `WeightedSlotGate` sigue siendo necesario.

### Carreras corregidas durante el spill adaptativo

La validación contra servicios residentes reprodujo cuatro condiciones que
podían causar el fallo nocturno: un runner Ollama físicamente presente pero
sin estado lógico actualizado, un PIN adoptado y recargado de nuevo al
arrancar, una muestra intermedia de VRAM después del desalojo y la misma
estimación de Whisper contabilizada dos veces. La admisión reconcilia ahora el
inventario físico antes y después del drain, adopta un PIN residente sin
recargarlo, espera la liberación física completa y, si persiste déficit, hace
una última pasada con la VRAM real. El smoke final escaló el abliterated pese a
Gemma, Whisper y embeddings residentes y devolvió HTTP 200 en 88,4 s.

Ambos Gemma completaron prefills reales de 200k sobre el pool nativo de 256k
con cuatro slots y visión. Gemma 12B conserva margen amplio; Gemma 26B queda a
564 MiB libres y por ello 256k/4 se considera máximo técnico. Qwen 128k/4 con
visión se validó mediante spill 1:2 sobre 3060+3090, no mediante full offload
en una sola 3090.

Los smokes iniciales fijaron `--kv-unified-per-slot 32768`; la ejecución del
2026-10-31 omitió el cap y confirmó que los cuatro slots ven el pool completo
sin multiplicar la reserva física. Las combinaciones desiguales funcionan; al
exceder el pool no hay rechazo limpio, sino reintentos, serialización y purgas.

### Fallos de activación visual observados

- Qwen 96k con seis slots dejó menos memoria libre que los 887,99 MiB pedidos
  por CUDA durante la activación. `cudaMalloc` devolvió OOM y un `GGML_ASSERT`
  abortó el servidor, en vez de devolver un error recuperable.
- Activar en Gemma 12B un projector de Gemma 26B devolvió HTTP 500 por
  incompatibilidad de `n_embd`, pero retuvo unos 1,2 GiB hasta reiniciar el
  proceso. El projector correcto cargó después, confirmando que era una fuga en
  la ruta de error y no el consumo residente normal.

Ambos casos son candidatos acotados para reproducción y propuesta upstream.

### Incidente de reserva de servicio durante los smokes

Al reiniciar la API se reprodujo una reserva fantasma: un servicio podía
reservar VRAM y detener su contenedor antes de llegar a `runtime_loaded=true`.
El refresco registraba el fallo de salud, pero conservaba indefinidamente la
reserva aunque el contenedor ya no existiera. Esto bloqueó una admisión dual
por 136 MiB y se podía confundir con consumo real.

`ServiceManager.refresh()` libera ahora la reserva no materializada si el
contenedor está detenido. Si el contenedor continúa ejecutándose, la conserva
porque un health check puede fallar transitoriamente durante una carga pesada.
Dos regresiones cubren ambos casos. Tras desplegar la corrección, la reserva
de GPU0 que reapareció correspondía a cargas reales de Whisper y
`qwen3-embedding:4b`, verificadas por proceso y telemetría, y no se retiró.

El mismo escenario expuso una segunda carrera: la admisión de Qwen abliterated
desalojaba primero Gemma 26B y solo después descubría que, aun recuperando sus
23.632 MiB, no alcanzaba los 29.656 MiB requeridos mientras GPU0 atendía carga
real. La prevalidación suma ahora la VRAM libre efectiva y toda la VRAM
recuperable como límite superior optimista. Si ese máximo es insuficiente,
responde sin desalojar nada. Prueba en vivo: 26.437 MiB máximos, Qwen rechazado
con HTTP 409 y Gemma permaneció cargado con 256k/4.

El barrido de 256k descubrió además que una descarga Ollama podía liberar el
runner físico pero conservar su lock lógico cuando el backend informaba
`gpu=[]` durante la parada. `GPUManager` permite ahora consultar en qué GPU
permanece reservado un `model_id`, y `ModelManager.unload()` libera la unión
de esa información con `state.current_gpu`. La regresión reproduce la pérdida
de atribución y comprueba que el lock de 5.903 MiB se retira.

Todos los procesos temporales usaron el build `b1-74bca39`, escucharon solo en
`127.0.0.1:19099` y se ejecutaron con `--cache-ram 0
--no-cache-idle-slots`. Al terminar se comprobó el cierre del puerto y la
liberación de la GPU. El runtime productivo Gemma 4 26B se restauró después en
su perfil habitual de 2×64k de aquel momento. Esa restauración es histórica;
el perfil productivo actual es el pool unificado 256k/4 documentado arriba.
También se restauraron Whisper y Qwen embeddings
en la 3060; chat y embeddings respondieron HTTP 200 antes de retirar
mantenimiento.

## Smoke de infraestructura

| Fecha UTC | Modelo | Dispositivo | Pool | Slots | Configuración | Resultado |
|---|---|---|---:|---:|---|---|
| 2026-10-30 | Qwen2.5 0.5B Q4_K_M | CPU | 8.192 | 4 | `--kv-unified --kv-unified-per-slot 2048 --cache-ram 0 --no-cache-idle-slots` | Arranque correcto; `/props` anunció 4 slots y 2.048 por slot; cuatro peticiones concurrentes ocuparon los slots 0, 1, 2 y 3 |

El proceso temporal escuchó solo en `127.0.0.1:19099`, se terminó tras la
prueba y se verificó que el puerto quedó cerrado. Este smoke valida el wiring y
la semántica básica del build, no el rendimiento ni la capacidad de los tres
modelos objetivo.

## Descargas adicionales

| Fecha | Repositorio/fichero | Quant | Bytes | SHA-256 | Motivo |
|---|---|---|---:|---|---|
| 2026-11-01 | `deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-Frontier-Intelligence-Injected-OBLITERATED-GGUF` / `RavenX-Chaos-Agent-Q4_K_M.gguf` | Q4_K_M | 16.547.399.968 | `4f5df36aca2c2547445987630dadfb102f1948232babe1f53c4365759590b7bf` | Candidato experimental de agentes y seguridad; perfil público `ravenx-chaos-agent` |
| 2026-11-01 | `deadbydawn101/RavenX-OpenFable-Coder-Gemma-4-12B-GGUF` / `RavenX-OpenFable-Coder-Gemma-4-12B-Q4_K_M.gguf` | Q4_K_M | 7.381.381.760 | `4001f2bcee18ad52da149f361f6c089fa7b0f2cc34144f29e24436c904548f67` | Candidato experimental de código; perfil público `openfable-coder` |

## Evaluación funcional RavenX del 2026-11-01

| Perfil | Obediencia exacta | Código | Herramientas | Generación observada | Veredicto |
|---|---|---|---|---:|---|
| `ravenx-chaos-agent` | Contenido correcto, con bloque `<think></think>` vacío | Algoritmo de merge correcto; incumple «sin Markdown» | Llamada válida `get_weather({"city":"Madrid"})` y roundtrip correcto | 37-39 tok/s | Útil como agente experimental |
| `openfable-coder` | Falla incluso con `temperature=0`; genera branding y contenido no pedido | Función compleja incorrecta; función trivial correcta pero contaminada | Nombre correcto, argumentos corruptos hasta agotar tokens | 35-38 tok/s | No apto como coder o agente principal |

Chaos se ejecutó con `--jinja --reasoning off --reasoning-format none`, por lo
que no consumió el presupuesto en razonamiento: la prueba exacta terminó en
cinco tokens y no publicó `reasoning_content`. El formato `none` de llama.cpp
conserva las etiquetas de pensamiento vacías dentro de `content`.

OpenFable repitió el fallo con muestreo por defecto y determinista. Ante una
petición de un único token devolvió su cabecera de identidad y un pseudo-tool;
su implementación de `merge_intervals` accedía a `merged[0]` cuando la lista
estaba vacía y construía resultados incorrectos. Se mantiene únicamente como
perfil experimental hasta disponer de otro quant o template que cambie este
comportamiento.
