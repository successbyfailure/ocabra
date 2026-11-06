# Plan de KV elástica y paralelismo dinámico en llama.cpp

**Fecha de apertura:** 2026-10-30
**Estado:** activo; reparto y escalado adaptativo implementados, soak pendiente
**Backlog maestro:** Bloque 22 de `docs/ROADMAP.md`
**Backend inicial:** `llama_cpp` nativo
**Modelos iniciales:** Gemma 4 12B, Gemma 4 26B-A4B y Qwen3.8 27B
**Hardware objetivo:** RTX 3060 12 GB, RTX 3090 24 GB y ambas combinadas
**Fork:** `makespacemadrid/llama.cpp-mks`, rama `ocabra/elastic-kv`
(clon de trabajo en `/docker/llama.cpp-ocabra`). Publicado 2026-11-06 con el
commit `743e3ee6e` de visión bajo demanda (Fase 1b); hasta entonces la rama no
tenía ningún commit y el trabajo vivía sin commitear en el árbol. Ver
[`backend-distribution-plan.md`](backend-distribution-plan.md).

Este trabajo busca mantener los pesos de un modelo residentes y reservar el
mayor pool KV seguro en cada colocación de hardware para repartirlo
dinámicamente entre las peticiones. El primer objetivo es ofrecer cuatro slots
físicos sin dividir la KV en cuatro parcelas rígidas. oCabra admitirá cada
petición según su presupuesto de tokens, la memoria libre del pool y las
prioridades de la cola.

El resultado esperado es eliminar las recargas al alternar cargas como
`4×32k`, `3×64k` o combinaciones desiguales que quepan en el mismo pool y en
la misma colocación de pesos. El salto de una GPU a dos GPU seguirá siendo una
transición drenada mientras implique redistribuir pesos.

## Estado de ejecución

| Fase | Resultado | Estado | Evidencia |
|---|---|---|---|
| 0 | Baseline mínimo e inventario reproducible | En curso | Inventario y smokes 3090 hechos; barrido oportunista pendiente |
| 1 | Cuatro slots y pool KV unificado configurables | En curso | Wiring, tests y smokes reales en ambos Gemma; validación larga pendiente |
| 1b | Visión enabled/disabled/on-demand | Validación funcional completada | Backend y UI desplegados; transición, grounding y concurrencia validados en Gemma 12B/26B y Qwen3.8; faltan carga diversa y soak |
| 2 | Admission control ponderado por tokens | Implementado y validado en canary | Gemma 26B 128k/4 real, cancelación, remanente y oversized; tokenizer exacto pendiente |
| 3 | Validación de rendimiento y corrección | En curso | Smokes, grounding, cancelación y soak acelerado de 1.000 peticiones completados; matriz diversa y 12 h pendientes |
| 4 | Observabilidad KV candidata a upstream | Pendiente | Patch aislado y tests llama.cpp |
| 5 | Recreación de contexto conservando pesos | Pendiente | Benchmark frente a recarga completa |
| 6 | Escalado drenado de una a dos GPU | Implementado y validado | Abliterated 64k/1 → 224k/4 con visión → 64k/1; reconciliación, drenado e histéresis probados |
| 7 | Despliegue gradual y propuestas upstream | Pendiente | Feature flag, rollback y PRs |

No se avanzará una fase por haber terminado el código. Deben cumplirse sus
pruebas y registrar la evidencia en este documento o en un fichero enlazado de
`docs/benchmarks/`.

### Evidencia de visión bajo demanda (2026-10-31)

- oCabra persiste `vision_mode` y `mmproj_model_id`, muestra las tres políticas
  en la UI y bloquea configuraciones activas sin projector.
- El backend arranca `on_demand` con `--props` y sin `--mmproj`, serializa la
  primera imagen con las peticiones de texto y reintenta mientras haya slots
  activos.
- El fork local implementa `POST /props` mediante la cola principal del
  servidor, espera a que no haya slots activos y actualiza de forma atómica el
  contexto multimodal de todos los slots. Las rutas HTTP que leen ese estado
  quedan protegidas durante la transición.
- `llama-server` compiló completo en CPU y CUDA (`sm_86`). Un smoke real devolvió
  `400` para un projector ausente y `501` para descarga no soportada.
- El build CUDA vivía en `/data/backends/llama_cpp/ocabra-src/build-cuda/`,
  construido a mano y seleccionado con `LLAMA_CPP_CUSTOM_SERVER_BIN`.
  **Obsoleto desde 2026-11-06**: el fork es el backend `llamacpp_mks`, que
  instala su propio binario desde `makespacemadrid/llama.cpp-mks`; el árbol a
  mano y el env var se eliminaron. Ver
  [`backend-distribution-plan.md`](backend-distribution-plan.md).
- Gemma 4 26B se validó end-to-end en la RTX 3090: `enabled` sirvió imagen,
  `disabled` la rechazó con HTTP 400 en 13 ms y `on_demand` activó el projector
  sin recargar pesos. Con el worker cargado, la primera imagen tardó 2,343 s y
  la VRAM pasó de 19.733 a 21.015 MiB.
- Una imagen concurrente con texto esperó al drenado y dos primeras imágenes
  simultáneas se serializaron en una sola activación; ambas devolvieron 200 y
  reconocieron las máscaras de comedia y tragedia.
- Gemma 4 12B repitió el caso con pool unificado 128k/4: dos activaciones
  simultáneas tardaron 0,343 s, cargaron una sola vez y el coste residente fue
  408 MiB. Dos imágenes simultáneas ocuparon slots distintos y fueron correctas.
- Qwen3.8 repitió el caso con pool unificado 96k/4: la activación tardó
  0,716-0,717 s, el coste residente fue 1.210-1.216 MiB y dos imágenes
  simultáneas respondieron correctamente.
- Una fixture 1280×960 validó OCR y grounding: con `ubatch=128` situó D4 en
  `[160,555]` frente a `[160,560]` real, y dos peticiones concurrentes fueron
  correctas en slots 0 y 1. `ubatch=1024` no mejoró precisión, mantuvo los
  avisos de posiciones no consecutivas y redujo el margen a 119 MiB, por lo que
  se descarta para el perfil 96k.
- Para la tarea estructurada Qwen agotó 350 y 900 tokens deliberando cuando se
  dejó thinking activo; con `chat_template_kwargs.enable_thinking=false`
  completó el JSON correcto. Debe convertirse en política explícita del perfil
  visual o de la petición, no en un supuesto oculto del benchmark.
- El soak corto mixto completó 40/40 peticiones Qwen y 20/20 Gemma 12B llenando
  los cuatro slots con texto e imágenes. Cuatro desconexiones simultáneas por
  modelo liberaron todos los slots en menos de un segundo y cada runtime
  respondió correctamente a cuatro peticiones inmediatas de recuperación. No
  hubo OOM, reallocs ni contaminación; el soak de 12 horas sigue pendiente.
- Qwen3.8 completó después un soak acelerado de 1.000/1.000 peticiones en 237 s
  (500 textos únicos y 500 imágenes), sin OOM, reallocs ni contaminación. La
  VRAM creció solo unos 6 MiB. Tras el soak, cuatro streams se cancelaron y
  drenaron en 375 ms; la recuperación inmediata fue 4/4. El p95 quedó por
  debajo de 0,86 s en texto y 0,94 s en visión repetida.
- Se corrigió y cubrió con regresión la configuración obsoleta de runtimes
  derivados descargados.
- Se corrigió la carrera de descarga->carga manual: la descarga espera una
  muestra fresca y acotada de VRAM. En hardware devolvió en 1,201 s y la carga
  inmediata entró con HTTP 200 sin reintento.

## Decisiones vigentes

- El objetivo inicial son **cuatro slots físicos**. Después se medirán 1, 2, 3,
  5 y 6 slots; no se asumirá que cuatro sea el óptimo final.
- La primera campaña se limita a **Gemma 4 12B**, **Gemma 4 26B-A4B** y
  **Qwen3.8 27B**, usando los artefactos productivos ya instalados.
- Encontrar el contexto máximo de cada modelo y colocación no bloquea el
  desarrollo inicial. Se usarán pools conservadores conocidos y el barrido se
  completará de forma oportunista cuando haya una ventana operativa adecuada.
- No se fija el pool de la 3090 en 128k. Se medirán al menos 128k, 160k, 192k,
  224k y 256k, y se adoptará el mayor valor que supere carga, concurrencia,
  prefill, generación sostenida y margen de seguridad.
- Los slots no reciben fracciones fijas del pool. La unidad de admisión es una
  reserva de tokens KV por petición.
- La primera implementación reserva el presupuesto completo antes de admitir
  la petición. El overcommit o crecimiento incremental quedan fuera hasta
  demostrar que la versión conservadora es correcta.
- Se prefiere mantener el modelo en la RTX 3090. Solo se usarán ambas GPU cuando
  una petición o la demanda agregada no quepa en la configuración validada de
  una GPU.
- Cada modelo se probará en 3060, 3090 y 3060+3090. Si los pesos no caben
  completamente en una GPU, se registrará `no viable full-offload`; una prueba
  con offload parcial a CPU será diagnóstica y no se presentará como capacidad
  productiva de esa GPU.
- Cambiar el número lógico de peticiones no debe cambiar `n_parallel`: se
  arrancará con capacidad física para cuatro secuencias y oCabra controlará la
  admisión activa.
- El fork se mantendrá orientado a upstream. Los cambios de llama.cpp no
  contendrán perfiles, políticas de GPU ni nombres propios de oCabra.
- Toda funcionalidad nueva para llama.cpp comenzará con una discusión o issue
  reproducible antes de abrir un PR. La comunicación pública y la descripción
  del PR las escribirá y revisará una persona, conforme a su política de IA.
- No se desplegará KV unificada en producción hasta superar las pruebas de
  aislamiento de conversaciones, MTP, visión, cancelación y carga sostenida.
- La visión admite tres políticas: cargada al arranque, desactivada o cargada
  al llegar la primera imagen. En modo bajo demanda el projector permanece
  residente hasta descargar el worker; su desalojo automático se diseñará
  después de medir coste, latencia y seguridad de la transición.
- La caché de prompts en RAM y el guardado automático de slots se desactivarán
  durante el spike inicial para aislar los problemas conocidos de reutilización
  cruzada. Se reactivarán solo con pruebas específicas.

## Alcance

### Incluido

- Gemma 4 12B, Gemma 4 26B-A4B y Qwen3.8 27B servidos por llama.cpp nativo.
- KV q8, Flash Attention, continuous batching, MTP/speculative y mmproj cuando
  correspondan al modelo.
- Cuatro slots físicos con pool KV compartido.
- Admission control por tokens integrado con la cola justa actual.
- Observabilidad de capacidad, reserva, uso y saturación del pool.
- Recreación futura de `llama_context` sin recargar `llama_model` cuando no
  cambie la colocación de pesos.
- Carga en caliente del mmproj sin recargar los pesos para visión bajo demanda.
- Transición drenada a dos GPU para presupuestos que excedan una GPU.
- Parches pequeños y genéricos que puedan proponerse a llama.cpp upstream.

### Fuera de alcance inicial

- Implementar PagedAttention completo.
- Redimensionar in-place los buffers KV mientras hay inferencias activas.
- Mover solo una parte de la KV a la segunda GPU manteniendo todas las capas en
  la primera; esto introduciría transferencias entre dispositivos por capa.
- Overcommit de KV sin garantía de completar el `max_tokens` solicitado.
- Cambiar de modelo, `gpu_layers`, `tensor_split` o draft model sin recargar
  los pesos afectados. El mmproj queda incluido de forma explícita para visión
  bajo demanda.
- Sustituir las políticas generales de fairness, batches o preemption; el nuevo
  gate debe integrarse con ellas.

## Situación de partida

oCabra incluye hoy contexto, paralelismo, tipo KV y colocación en la identidad
del runtime. Los presets generan workers derivados y `llama-server` recibe
`--ctx-size` y `--parallel` solo al arrancar. El build instalado el 2026-10-30
corresponde al commit `74bca395` de llama.cpp; el master revisado durante la
investigación era `792fcf4e` y tampoco exponía una operación para cambiar
`n_ctx` o `n_parallel` en caliente.

Los logs del nodo principal del 2026-10-30 registraron aproximadamente 42 s
para una carga inicial de Gemma4 26B y 10,5 s para una segunda carga con la
page cache caliente. Estos tiempos son el baseline preliminar; la Fase 0 debe
repetirlos de forma controlada.

Los tres artefactos iniciales ya están presentes en el almacén Ollama y pueden
ser consumidos directamente por el `llama-server` propio:

| Modelo | Arquitectura GGUF | Pesos | Projector | Contexto nativo |
|---|---|---:|---:|---:|
| Gemma 4 12B | `gemma4` | 7.381.383.680 B | 175.115.616 B | 262.144 |
| Gemma 4 26B-A4B | `gemma4` | 17.074.419.072 B | 1.194.828.128 B | 262.144 |
| Qwen3.8 27B | `qwen35` | 16.810.714.464 B | 931.146.016 B | 262.144 |

No se descargará una copia duplicada. Si una prueba exige otra cuantización,
se registrará antes el repositorio, fichero, checksum, tamaño y motivo.

### Ventana operativa para benchmarks

Un uso instantáneo de GPU cercano a cero no basta para considerar el nodo
libre. Antes de una prueba que descargue o reemplace un runtime se comprobará:

- ninguna petición de inferencia activa;
- ausencia de tráfico de usuario durante el periodo de guarda acordado;
- ninguna descarga, batch o sesión Realtime activa;
- capacidad de restaurar los runtimes y perfiles anteriores;
- captura previa de procesos, VRAM, argumentos y modelos residentes.

El tráfico periódico de Home Assistant cuenta como actividad. Las tareas de
inventario y metadata pueden continuar, pero el barrido destructivo esperará
una ventana que no interrumpa ese servicio.

La configuración objetivo conceptual es:

```text
RTX 3090
├── pesos, MTP/mmproj y buffers fijos
└── pool KV compartido máximo seguro
    └── hasta 4 secuencias físicas
        └── reservas variables decididas por oCabra
```

Un pool validado de 192k, por ejemplo, admitiría cualquiera de estas formas:

```text
4 × 32k             = 128k
3 × 64k             = 192k
2 × 96k             = 192k
128k + 32k + 32k    = 192k
```

Los valores ilustran el reparto; el límite real se decidirá con medidas.

## Modelo de capacidad y admisión

### Capacidad segura

El pool no ocupará toda la VRAM que quede tras cargar los pesos. Se reservará
margen para CUDA graphs, Flash Attention, buffers de prefill, speculative/MTP,
mmproj y variaciones no modeladas. El planner seguirá aplicando como punto de
partida el factor de seguridad existente y el benchmark podrá exigir un margen
mayor.

```text
kv_pool_safe = floor(
    (vram_total - weights - fixed_overhead - operational_headroom)
    / kv_bytes_per_token
)
```

La capacidad aceptada debe ser estable durante la carga máxima, no solo durante
el arranque o una generación corta.

### Reserva por petición

```text
request_reservation =
    exact_or_estimated_prompt_tokens
    + requested_or_default_output_tokens
    + modality_overhead
    + estimation_margin
```

- Se usará tokenización exacta cuando pueda obtenerse sin cargar otro runtime.
- El estimador conservador actual será el fallback.
- `max_completion_tokens`, `max_tokens` y `num_predict` deben converger en una
  única reserva de salida.
- Visión, audio y herramientas deben aportar su coste al presupuesto.
- Una petición sin límite explícito usará la reserva por defecto del perfil.
- La reserva se libera exactamente una vez al completar, cancelar o fallar.

### Reglas del gate

Una petición puede entrar cuando se cumplen simultáneamente:

```text
active_requests < physical_slots
reserved_kv_tokens + request_reservation <= kv_pool_tokens
```

El gate debe además:

- conservar la justicia por usuario de `FairSlotGate`;
- mantener prioridad interactiva sobre batches;
- evitar starvation de una petición grande frente a un flujo de peticiones
  pequeñas;
- cancelar esperas al desconectarse el cliente;
- despertar a los candidatos cuando se libera KV o un slot;
- exponer saturación por slots y por KV como causas distintas;
- impedir que una actualización de capacidad invalide reservas activas.

En la primera versión no se usará la ocupación instantánea para admitir por
encima de las reservas. Esa optimización requeriría una política explícita de
crecimiento, pausa o terminación cuando el pool se agote.

## Arquitectura de runtimes

### Identidad residente

Se introducirá conceptualmente un `resident_key` para los elementos que obligan
a recargar o redistribuir pesos:

- modelo y cuantización de pesos;
- GPU visibles y colocación;
- `gpu_layers`, `split_mode`, `tensor_split` y `main_gpu`;
- draft model o MTP que altere los pesos/contextos auxiliares;
- mmproj y adaptadores cuyo ciclo de vida no pueda separarse;
- modo de carga y overrides de tensores.

### Identidad del contexto

El `runtime_key` continuará representando el contexto activo:

- tamaño total del pool KV;
- slots físicos;
- tipo de K y V;
- KV unificada y límite por secuencia;
- batch y ubatch si requieren recrear el contexto;
- parámetros de RoPE que formen parte del contexto.

Si coincide `resident_key`, oCabra podrá reutilizar el proceso o pedir una
recreación de contexto. Si cambia, usará la recarga completa actual.

### Transiciones

| Origen | Destino | Tratamiento esperado |
|---|---|---|
| `4×32k` en 3090 | `3×64k` en el mismo pool | Sin cambio de runtime; solo admisión |
| Una petición | Varias peticiones dentro del pool | Sin recarga |
| Pool pequeño en 3090 | Pool mayor que aún cabe en 3090 | Recrear contexto conservando pesos |
| 3090 | 3060+3090 | Drenar y recargar pesos con nuevo split |
| 3060+3090 | 3090 | Downgrade con histéresis y recarga drenada |
| Cambio de KV q8 a q4/f16 | Misma colocación | Recrear contexto; nunca durante una petición |

## Estrategia para llama.cpp upstream

El desarrollo debe empezar sobre un commit upstream fijado y mantener una serie
de parches pequeños. Cada cambio candidato a upstream necesita reproducción,
tests del propio proyecto y una motivación independiente de oCabra.

El fork local parte de `74bca3950` (tag `b11870`), el mismo commit del build
instalado. El remoto oficial se llama `upstream` y tiene el push deshabilitado;
se añadirá `origin` solo cuando exista un fork remoto propio.

### Responsabilidad de oCabra

- Perfiles y selección de runtime.
- Presupuesto por petición y reservas conservadoras.
- Fairness por usuario, batches y preemption.
- Selección de una o dos GPU e histéresis.
- Capacidad administrativa, UI y estadísticas históricas.

### Cambios potencialmente upstreamables

1. Exponer en `/props`, `/slots` o métricas la capacidad KV total, ocupada y
   libre, junto con el uso por secuencia.
2. Separar capacidad física de secuencias de la concurrencia activa que acepta
   el scheduler.
3. Permitir un límite de contexto por tarea dentro de un pool unificado.
4. Cambiar en runtime el límite de concurrencia activa sin recrear el contexto.
5. Separar claramente la vida de `llama_model` y `llama_context` para recrear
   este último sin volver a cargar pesos.
6. Corregir cualquier fallo reproducible de KV unificada, MTP o prompt cache
   encontrado por la matriz de pruebas.

La recreación de contexto debe plantearse primero como discusión/RFC upstream si
obliga a cambiar `common_init_result` o APIs públicas. No se mezclará con la
instrumentación KV en un mismo PR.

### Discusiones upstream relacionadas

- `llama-server` ya ofrece `--kv-unified` y `--kv-unified-per-slot`. En modo
  unificado, `-c` reserva el pool total y cada secuencia puede direccionar todo
  el pool; `--kv-unified-per-slot` solo aplica un cap lógico. Por tanto, el
  perfil flexible debe dejar ese cap sin fijar o igualarlo al pool, no usar 32k.
- La discussion `#22658` documenta esta separación entre capacidad total y cap
  por slot. La discussion `#27916` confirma la semántica contra el código.
- La separación entre slots almacenados y paralelismo activo se ha solicitado
  en `ggml-org/llama.cpp` discussion `#22401`.
- La discussion `#25674` y el PR cerrado `#25963` prototiparon exactamente la
  recreación de contexto conservando pesos, además de recargar mmproj y
  speculative. Reportaron unos 200 ms para redimensionar el contexto de un
  Qwen 27B, pero el PR quedó cerrado y su diff ya diverge ampliamente del
  servidor actual; se usará como referencia de diseño, no como cherry-pick.
- Paged KV continúa como trabajo experimental fuera de trunk; no es una
  dependencia de este plan.
- Los problemas de prompt cache/KV unificada deben vigilarse mediante los
  issues upstream `#27148`, `#30252` y `#30961` o sus sucesores. Durante el
  spike se mantienen `--cache-ram 0` y `--no-cache-idle-slots` por aislamiento
  y para evitar I/O bloqueante en el hilo del servidor.
- El PR abierto `#31377` modifica `n_ctx_slot()`; hoy conserva el cap de
  `--kv-unified-per-slot`, pero debe revisarse antes de cada rebase.

### Post mortem de los prototipos anteriores (2026-10-31)

No hubo un rechazo técnico único de upstream. Hubo dos experimentos con
alcance y destino distintos:

1. La serie `--ctx-dynamic` de junio de 2026 seleccionaba tiers y llamaba a la
   ruta de recarga existente. Releía pesos, recalculaba `fit`, exigía
   `--parallel 1` y reprocesaba tokens al crecer durante una generación. Además
   reiniciaba el historial del sampler, podía perder un stop partido por la
   transición y no soportaba crecimiento multimodal durante el stream. No era
   un pool KV elástico ni resolvía concurrencia variable. La serie quedó dentro
   de la PR `#20872`, cuya rama se reutilizó y se forzó repetidas veces para un
   cambio de SSE; terminó cerrada sin merge como una PR incoherente (`Bad
   merge`). Los commits históricos siguen accesibles, pero no hubo una revisión
   mantenible de esta feature.
2. El RFC `#25674` y el draft `#25963` sí separaban la vida de pesos y contexto.
   `reload_context` reconstruía el contexto sin tocar pesos y midió unos 200 ms
   en Qwen 27B, aproximadamente una décima parte de la carga completa. Sin
   embargo, cada transición destruía slots y prompt cache, no ejecutaba `fit`,
   dejaba al caller la seguridad de memoria y todavía no tenía API HTTP. El
   autor lo dejó draft sobre `#25465`, un cambio previo para persistir `n_ctx`;
   no recibió review, quedó desfasado por rebases/refactors y se cerró el 11 de
   agosto. La dependencia se resolvió después por otra PR, no reabriendo el
   prototipo.

Consecuencia para oCabra: reutilizaremos la idea pequeña de conservar
`llama_model` al recrear `llama_context`, pero no portaremos ninguna de las dos
ramas. Primero explotaremos el pool KV unificado ya disponible con cuatro slots
residentes y admission control externo. La recreación sólo se añadirá después
como transición drenada, con estimación/`fit`, rollback por OOM y restauración
explícita de estado; slots activos y prompt cache no se sacrificarán de forma
silenciosa. Esta separación también produce unidades upstream más revisables:
observabilidad, límite de concurrencia activa y context reinit serán cambios
independientes.

### Normas de contribución y diseño upstream

Estas reglas se comprobaron contra `CONTRIBUTING.md`, `AGENTS.md`,
`tools/server/README-dev.md`, `.editorconfig`, `.clang-format` y `.clang-tidy`
del master revisado el 2026-10-30. Forman parte del criterio de aceptación de
cualquier patch del fork, no son una limpieza posterior.

#### Antes de escribir código

- Buscar issues, discusiones y PRs existentes para evitar propuestas
  duplicadas.
- Las features empiezan con issue o discusión; un bugfix empieza con un caso
  reproducible y debe incorporar un test de regresión.
- Acordar el alcance mínimo y mantener una sola unidad lógica por PR. Para un
  primer contribuidor, se priorizará una contribución abierta cada vez.
- Evitar una API nueva cuando baste con internos de `llama-server`, sus
  tareas/colas o métricas existentes. Una API experimental, si se acuerda,
  pertenece a `llama-ext.h` antes que a la API estable `llama.h`.
- Toda API nueva necesita un ejemplo o herramienta que la ejercite en el mismo
  cambio. Los tipos públicos deben usar enteros dimensionados o `size_t` según
  corresponda.

#### Encaje en `llama-server`

- La gestión de memoria, contextos y modelos está dentro del alcance declarado
  del servidor, siempre que siga siendo independiente del modelo.
- `server_context` trabaja en un único hilo dedicado. No se introducirá en él
  postprocesado pesado, espera de I/O ni trabajo administrativo bloqueante.
- Los cambios deben reutilizar `server_task`, `server_queue`,
  `server_response` y el ciclo de los slots. La capa HTTP convierte JSON a
  tipos C++ nativos; no se pasará JSON crudo hasta los slots.
- Capacidad física, ocupación observada y límite administrativo deben ser
  conceptos separados. Las métricas no pueden cambiar la semántica actual de
  batching ni exponer estado interno inestable como contrato público.
- La reconfiguración debe partir del ciclo ya usado por sleep/wake, que destruye
  y vuelve a crear contextos, pero demostrar por separado que puede conservar
  `llama_model` de forma segura.

#### Estilo de código

- C/C++ con cuatro espacios, llaves en la misma línea, `void * ptr`,
  `int & value`, LF, nueva línea final y sin espacios finales.
- Nombres `snake_case`; funciones con patrón `<clase>_<acción>_<sustantivo>`;
  enums en mayúsculas y prefijados por su tipo. Los nombres relacionados
  comparten el prefijo más largo, por ejemplo `kv_pool_total` y `kv_pool_used`.
- Ficheros C/C++ en minúsculas y `kebab-case`; Python en minúsculas y
  `snake_case`.
- Solo ASCII en código y comentarios. Los comentarios explican brevemente el
  porqué no obvio, sin repetir el código ni hacer referencia al PR o a oCabra.
- Mantener el cambio simple: infraestructura existente, STL básica, sin
  dependencias, headers, ficheros, plantillas o abstracciones nuevas salvo que
  sean imprescindibles y defendibles.
- Aplicar `clang-format` compatible con la versión requerida por upstream y
  atender `.clang-tidy`; no reformatear código ajeno al cambio.

#### Pruebas y evidencia

- Los helpers pequeños, colas y tipos internos del servidor se prueban primero
  en `tests/test-server-component.cpp`. Los pytest end-to-end de
  `tools/server/tests` se reservan para comportamiento que necesite un modelo
  real.
- Reutilizar ficheros y fixtures existentes. No introducir sleeps, umbrales de
  tiempo frágiles, descargas grandes ni otro framework de tests.
- Ejecutar los checks de estilo y EditorConfig del proyecto, la suite del
  servidor y CI local relevante. Para CUDA, compilar y probar con
  `GGML_CUDA=ON`.
- Un cambio que afecte a rendimiento necesita resultados de `llama-bench` y,
  cuando proceda, `llama-perplexity`. Un cambio de ggml/backend exigiría además
  `test-backend-ops` en al menos dos backends; este plan intentará no tocar
  ggml.
- Probar KV normal y unificada, KV cuantizada, uno y varios slots, atención
  convencional e híbrida, MTP y cancelación según el patch afectado.

#### Política de IA y ownership humano

llama.cpp admite código asistido por IA solo cuando el contribuidor entiende,
puede explicar y asume cada línea, y exige declarar el uso significativo. No
permite delegar en IA la redacción de issues, solicitudes de feature,
descripciones de PR ni respuestas a reviewers, ni automatizar la publicación.
Por tanto:

- este documento puede guiar la investigación y revisión local, pero una
  persona redactará desde cero toda comunicación pública upstream;
- no se abrirá ni enviará automáticamente ningún issue o PR;
- antes de publicar, el responsable humano reproducirá los resultados y hará
  una revisión línea a línea;
- `Co-authored-by` queda reservado a personas; si upstream pide atribución de
  asistencia se usará su convención `Assisted-by`.

### Secuencia prevista de contribuciones

| Orden | Unidad independiente | Superficie preferida | Condición previa |
|---:|---|---|---|
| 1 | Reproducción o corrección de aislamiento/rendimiento encontrada | Bugfix interno y test de regresión | Fallo reproducible fuera de oCabra |
| 2 | Observabilidad KV inequívoca | Métricas/props del servidor, sin C API nueva | Semántica validada en KV normal y unificada |
| 3 | Separar slots físicos de concurrencia activa | Scheduler interno de servidor | Discusión aceptada y benchmark |
| 4 | Límite por tarea dentro del pool | `server_task`/slot, modelo agnóstico | Pruebas de aislamiento y compatibilidad |
| 5 | Recrear contexto conservando pesos | RFC y luego cambio interno mínimo | Ganancia medida frente a recarga caliente |

Cada unidad tendrá su propia reproducción, tests y datos. No se encadenarán
en un macro-PR y no se mezclará el admission control específico de oCabra.

## Fases de trabajo

### Fase 0 Baseline mínimo e inventario

- [x] Fijar commit exacto de llama.cpp, flags, modelo y artefactos GGUF/mmproj.
- [x] Identificar los artefactos de Gemma 4 12B, Gemma 4 26B-A4B y Qwen3.8
  27B ya instalados, incluidos projector y contexto nativo.
- [x] Guardar la configuración productiva actual de los tres modelos.
- [ ] Medir carga fría, carga con page cache caliente y descarga.
- [x] Medir VRAM residente por pool, slots, MTP y activación de mmproj; queda
  pendiente separar buffers temporales de prefill en telemetría upstream.
- [x] Medir cada modelo con y sin `mmproj` manteniendo iguales pool y slots;
  costes residentes observados: Gemma 12B 408 MiB, Gemma 26B 1.282 MiB y
  Qwen3.8 1.210-1.216 MiB. Son medidas de activación completa, no solo fichero.
- [x] Guardar smokes baseline conservadores para Gemma 4 12B y 26B, y el fallo
  reproducible de Qwen3.8 a 4×32k; continuar el barrido sin bloquear Fase 1.

**Salida bloqueante:** artefactos, comandos y al menos un baseline conservador
reproducible para comenzar el spike de cuatro slots.

#### Barrido de capacidad oportunista

Estas tareas permanecen en la Fase 0 como track continuo, pero no bloquean las
Fases 1 y 2:

- [x] Ejecutar Gemma 12B en 3060 y Qwen en 3060+3090, además de los tres en
  3090; Qwen y Gemma 26B superan por pesos los 12 GiB de full-offload.
- [ ] Encontrar primero el pool máximo con un slot mediante barrido ascendente y
  búsqueda acotada hasta el límite nativo de 262.144 tokens.
- [x] Repetir con cuatro slots y KV unificada, y después con 1, 2, 3, 5 y 6
  slots para Qwen y Gemma 12B alrededor de la mejor configuración.
- [ ] En dos GPU, comparar el reparto manual 1:2 ya validado con
  `tensor_split` automático; VRAM manual guardada: 9.431+16.678 MiB.
- [x] Registrar por separado el mayor pool que arranca, el que completa prefill
  y el que supera carga sostenida.
- [x] Medir concurrencia corta de 1-6 slots, prefills desiguales concurrentes y
  prefills largos individuales; ampliar a concurrentes largos durante soak.
- [x] Guardar resultados en `docs/benchmarks/llama_cpp_elastic_kv_baseline.md`.

**Salida no bloqueante:** mapa completo de capacidad por modelo y colocación.

### Fase 1 Pool unificado y cuatro slots

- [x] Exponer en el schema llama.cpp `kv_unified`, `kv_unified_per_slot`,
  `cache_ram_mib` y `cache_idle_slots` si el build fijado los soporta.
- [x] Propagar flags por backend y worker sin cambiar defaults existentes.
- [x] Arrancar un smoke con cuatro slots, KV unificada y prompt cache aislada
  usando Qwen2.5 0.5B en CPU, sin afectar las GPU productivas.
- [x] Repetir el smoke en GPU con Gemma 4 12B y Gemma 4 26B-A4B; ambos
  atendieron cuatro peticiones simultáneas en slots distintos sin contaminación.
- [x] Empezar con un pool conservador; ampliar solo cuando una medida sea útil
  para la siguiente decisión, no como prerrequisito del spike.
- [x] Verificar en `/props` y `/slots` el número de slots y el límite por slot.
- [ ] Verificar el pool KV total sin depender solo de argumentos y logs; el
  build actual no lo expone y enlaza esta tarea con la Fase 4.
- [x] Añadir tests de traducción de config snake_case/camelCase y comando final.
- [x] Documentar rollback a presets rígidos.

**Salida:** un runtime de cuatro slots configurable, todavía protegido por el
admission control conservador existente.

### Fase 2 WeightedSlotGate

Estado de continuidad (2026-10-31):

- Decisión: ampliar ``FairSlotGate`` en lugar de crear una segunda cola.
- Fuente de peso: ``RuntimePlan.required_tokens``, ya calculado por
  ``estimate_request_token_budget`` y guardado en ``request.state``.
- Capacidad: ``_runtime.effective_context`` solo para llama.cpp con
  ``kv_unified=true``; los demás runtimes mantienen semántica por slots.
- Implementado el gate bidimensional, el cableado en ``WorkerPool`` y las
  rutas chat/completions OpenAI y Ollama. La reserva se libera por contexto
  async en éxito, error, cancelación y streaming.
- El scheduler permite saltar una cabecera que no cabe para aprovechar el
  remanente con otra petición justa que sí cabe. Una petición mayor que el pool
  falla inmediatamente con ``context_length_exceeded``/HTTP 400.
- La prioridad interactiva detecta tanto slots llenos por batch como KV
  bloqueada por reservas batch aun con slots físicos libres.
- ``GET /ocabra/models/{id}/status`` publica slots, activas, encoladas y pool
  lógico total/reservado/libre. Los IDs canónicos resuelven el único runtime
  derivado cargado; no se presentan estas reservas como ocupación física KV.
- OpenAI expone headers de saturación y eventos SSE separados ``slot_queued``
  y ``kv_queued``. Ollama rechaza oversized antes de abrir un stream.
- El estimador reserva 128 tokens por imagen conforme al límite observado del
  runtime visual. Sigue siendo conservador por caracteres/palabras: oCabra no
  dispone aún de tokenización exacta pre-load que aplique la plantilla de chat
  específica del GGUF.
- Protección adicional: un gate activo no puede redimensionarse bajo el mismo
  ``worker_key``; una configuración distinta debe producir otro runtime key y
  así no puede dejar leases huérfanos.
- El canary encontró y corrigió dos fallbacks de perfiles sin presets: ahora
  siempre calculan ``required_tokens`` y leen ``ctx_size``/``parallel_slots``
  del runtime fijo. Sin ambas correcciones el proceso real era 128k/4 pero el
  admission quedaba inactivo o anunciaba un solo slot.
- No se cambiarán todavía defaults productivos ni perfiles activos.

Implementación y cobertura para continuar sin reconstruir contexto:

- Código: ``core/fair_slot_gate.py``, ``core/worker_pool.py``,
  ``core/runtime_coordinator.py``, ``api/openai/{_deps,chat,completions}.py``,
  ``api/ollama/{chat,generate}.py`` y ``api/internal/models.py``.
- Tests: ``test_fair_slot_gate.py``, ``test_worker_pool.py``,
  ``test_runtime_coordinator.py``, ``test_openai_api.py``,
  ``test_ollama_api.py`` y dos casos focales de ``test_models_api.py``.
- Regresión focal al cerrar esta iteración: 144 tests verdes. Único warning:
  ``.pytest_cache`` no escribible; no es un fallo funcional.
- Regresión ampliada llama.cpp/model manager/API: 266 tests verdes y uno de
  persistencia excluido porque el entorno local no resuelve el host
  ``postgres``. Ruff lint, Ruff format y ``git diff --check`` pasan. El test
  focal de la UI suma 4/4 casos verdes.
- Versión de la iteración: ``0.22.3`` sincronizada en backend, entorno,
  frontend y lockfile.
- Canary real Gemma 26B, 128k/4, KV q8 y ``kv_unified=true``:
  A reservó 89.530 tokens; B pidió 50.026 y quedó en ``kv_queued`` con tres
  slots físicos libres; C de otro usuario reservó 20.026 en el remanente
  (109.556 agregados). Cancelar C devolvió exactamente 20.026; cancelar A
  admitió B; cancelar B dejó 0 activas, 0 encoladas y 131.072 libres.
- Una reserva de 140.022 fue rechazada inmediatamente con HTTP 400 y
  ``context_length_exceeded``; la petición corta posterior respondió HTTP 200.
- Restauración verificada: perfil temporal eliminado, preset 128k de nuevo en
  paralelismo 1, Gemma productivo ``131072/parallel 2`` sin ``kv_unified``,
  Whisper y embeddings en la 3060, API healthy y chat público HTTP 200.
- Aclaración verificada contra los argumentos del proceso: un preset expresa
  contexto **por petición**, por lo que 32k/p3 materializa un pool llama.cpp de
  98.304 tokens. El máximo 44.219 para p4 era correcto como máximo por slot
  (176.876 agregados), no un límite del pool fijo. La canary 128k/4 sigue
  siendo válida porque usó una configuración fija sin presets, donde
  ``ctx_size=131072`` representa directamente el pool agregado.

- [x] Diseñar contrato antes de modificar `WorkerPool` y rutas OpenAI/Ollama.
- [x] Integrar reservas KV con `FairSlotGate`, sin crear una segunda cola
  independiente.
- [ ] Usar tokenización exacta cuando esté disponible y fallback conservador.
- [x] Hacer atómicas reserva, admisión y liberación.
- [x] Cubrir streaming, non-streaming, desconexión, cancelación y error backend.
- [x] Integrar prioridad interactiva y preemption de batches.
- [x] Exponer headers y eventos para `slot_saturated` y `kv_saturated`.
- [x] Publicar en el estado activo pool total, reservado, libre, slots físicos
  y peticiones activas.

**Salida:** reparto flexible por petición sin cambiar el proceso llama.cpp.

### Fase 3 Corrección y rendimiento

- [ ] Ejecutar toda la matriz funcional y de concurrencia.
- [x] Comparar KV unificada frente a los presets: se adopta el pool unificado
  fijo y se retiran los presets de Gemma/Qwen.
- [x] Validar Qwen3.8 híbrido/recurrente con MTP y mmproj.
- [x] Validar ambos Gemma 4 con draft/MTP, mmproj y peticiones multimodales.
- [x] Ejecutar un primer hito de contaminación con prompts y respuestas
  identificables: 1.000/1.000 correctas. El criterio final de 10.000 y entradas
  visuales diversas sigue pendiente.
- [ ] Medir fragmentación y degradación tras cientos y miles de peticiones.
- [ ] Ejecutar soak test mínimo de 12 horas con carga mixta.
- [x] Probar una primera tanda de cancelaciones y desconexiones abruptas en
  Qwen3.8 y Gemma 12B; repetir durante el soak largo y añadir clientes lentos.
- [x] Probar OOM deliberado y recuperación sin procesos huérfanos; el OOM de
  activación visual aborta el proceso, pero el siguiente arranque fue limpio.
- [x] Repetir con 1, 2, 3, 5 y 6 slots: Qwen favorece 4 y Gemma 12B mantiene
  margen hasta 6; falta evaluar throughput sostenido antes de cambiar defaults.

**Salida:** decisión documentada de continuar, ajustar el diseño o abandonar
KV unificada para alguno de los modelos.

### Fase 4 Observabilidad upstreamable

- [x] Buscar y registrar issues, discusiones y PRs equivalentes antes de
  diseñar la interfaz.
- [ ] Abrir primero una discusión o issue redactado por el responsable humano.
- [ ] Definir semántica exacta de capacidad, ocupación y reserva; no confundir
  celdas asignadas con tokens lógicos.
- [ ] Implementar métricas en un commit independiente del scheduler de oCabra.
- [ ] Añadir tests unitarios y de servidor en llama.cpp.
- [ ] Verificar KV unificada y no unificada, modelos de atención e híbridos.
- [ ] Ejecutar estilo, EditorConfig, CI del servidor y benchmarks aplicables.
- [ ] Someter el PR solo tras acuerdo de alcance, revisión humana línea a línea
  y declaración de asistencia conforme a la plantilla upstream.
- [ ] Mantener adaptador temporal en oCabra hasta conocer la respuesta upstream.

**Salida:** patch pequeño, genérico y revisable por llama.cpp.

### Fase 5 Recreación de contexto conservando pesos

- [ ] Separar el ciclo de vida de modelo, contexto y componentes auxiliares.
- [ ] Drenar todas las peticiones antes de comenzar.
- [ ] Conservar `llama_model` y recrear `llama_context` con nuevo pool/tipo KV.
- [ ] Recrear samplers, threadpools, slots y contexto speculative/MTP.
- [ ] Revisar el ciclo de vida de mmproj, adaptadores y TTS/multimodal.
- [ ] Vaciar prompt cache, checkpoints y estados incompatibles.
- [ ] Definir fallo y rollback cuando el contexto nuevo no pueda reservarse.
- [ ] Exponer una operación administrativa interna, no accesible al cliente.
- [ ] Añadir estado `reconfiguring` o documentar la transición equivalente.
- [ ] Comparar tiempo con la carga caliente completa y con la carga fría.
- [ ] Presentar primero el diseño upstream; si cambia APIs comunes, esperar
  consenso antes de implementarlo como superficie pública.

**Salida:** cambio de pool en la misma colocación sin releer ni transferir los
pesos del modelo.

### Fase 6 Escalado entre una y dos GPU

- [x] Definir umbral: petición individual, demanda agregada o espera estimada.
- [x] Drenar el runtime de una GPU antes de cambiar colocación.
- [x] Reutilizar el admission y eviction gate global durante el spill.
- [x] Evitar que dos peticiones concurrentes disparen dos transiciones.
- [x] Mantener un tiempo mínimo de residencia en dos GPU.
- [x] Volver a una GPU solo tras `downgrade_grace_seconds` y sin trabajo activo.
- [ ] Medir throughput prolongado y efecto sobre modelos vecinos con tráfico real.
- [x] Confirmar que el scheduler nunca anuncia una capacidad que aún no está
  activa.

**Salida:** spill predecible y poco frecuente, con la 3060 libre en el caso
normal.

### Fase 7 Despliegue y upstream

- [ ] Introducir feature flag global y opt-in por modelo/perfil.
- [x] Retirar los presets rígidos tras el canary; el rollback es volver a una
  configuración fija menor, no reintroducir escalones por petición.
- [ ] Canary con un solo modelo y un subconjunto de tráfico.
- [ ] Comparar al menos 24 horas contra baseline antes de ampliar.
- [ ] Actualizar contratos, API, UI, variables y documentación operativa.
- [ ] Separar los commits upstream de la integración oCabra.
- [ ] El responsable humano abrirá discusiones/PRs en el orden acordado y se
  registrarán enlaces y decisiones en este documento.
- [ ] Rebasear el fork solo después de repetir la matriz mínima.

**Salida:** función reversible en producción y serie de contribuciones upstream
mantenible.

## Matriz de pruebas

### Modelos y capacidades

| Modelo | Pesos GGUF | MTP/draft | mmproj | Contextos de sondeo | Slots iniciales |
|---|---:|---|---|---|---:|
| Gemma 4 12B | 7,38 GB | Sí | Sí | 32k, 64k, 96k, 128k, 160k, 192k, 224k, 256k | 4 |
| Gemma 4 26B-A4B | 17,07 GB | Sí | Sí | 32k, 64k, 96k, 128k, 160k, 192k, 224k, 256k | 4 |
| Qwen3.8 27B | 16,81 GB | Sí | Sí | 32k, 64k, 96k, 128k, 160k, 192k, 224k, 256k | 4 |

| Colocación | Objetivo | Regla |
|---|---|---|
| 3060 | Capacidad individual y límite de full-offload | No confundir offload CPU con capacidad de 12 GB |
| 3090 | Camino productivo preferido y mayor pool en una GPU | Conservar margen operacional medido |
| 3060+3090 | Máximo combinado y coste del spill | Medir split, transferencias y efecto sobre otros servicios |

Los puntos intermedios se buscarán de forma acotada entre el mayor éxito y el
menor fallo. No se forzarán mediante overcommit o memoria unificada CUDA. El
contexto extendido por encima de 262.144 queda fuera de la primera campaña para
no mezclar capacidad de memoria con calidad fuera del contexto nativo.

### Formas de carga

| Caso | Reservas | Objetivo |
|---|---|---|
| Usuario único corto | `1×8k` | Latencia y throughput base |
| Usuario único largo | `1×64k`, `1×128k`, máximo seguro | Uso completo del pool |
| Cuatro usuarios | `4×32k` | Objetivo inicial de paralelismo |
| Tres usuarios | `3×64k` | Validar pool mayor de 128k |
| Mezcla | `128k+32k+32k` | Reparto desigual |
| Cola por KV | solicitud que exceda el remanente | Espera sin OOM |
| Cola por slots | cinco solicitudes pequeñas | Límite físico y fairness |
| Batch e interactiva | slots y KV saturados por batch | Preemption y prioridad |
| Visión concurrente | imagen y texto simultáneos | mmproj y presupuesto multimodal |
| Cancelación | durante prefill y decode | Liberación exacta de reserva |

### Corrección

- Cada petición incluirá identificadores únicos y contenido incompatible con
  las demás para detectar contaminación cruzada.
- Se comparará salida concurrente y secuencial con seed y sampling controlados
  cuando el modelo lo permita.
- Se validará que ningún slot restaure KV, checkpoints o estado recurrente de
  otra conversación.
- Se comprobará tool calling, JSON schema, reasoning, streaming y non-streaming.
- Las pruebas de visión alternarán imágenes diferentes entre slots.

### Fallos y ciclo de vida

- OOM durante creación del contexto.
- Proceso llama.cpp muerto durante una reserva activa.
- Timeout durante drain.
- Cliente desconectado mientras espera y mientras genera.
- Cancelación de batch preemptado.
- Reconfiguración fallida y restauración del runtime anterior.
- Reinicio de oCabra con worker vivo o estado persistido obsoleto.
- Cambio de capacidad mientras hay peticiones en cola.

## Métricas obligatorias

| Área | Métricas |
|---|---|
| Capacidad | pool total, reservado, usado y libre; slots físicos/activos |
| Memoria | VRAM por GPU, pico, margen mínimo y RAM del proceso |
| Carga | carga fría/caliente, recreación de contexto y spill |
| Latencia | queue time, TTFT p50/p95, TPOT p50/p95, duración total |
| Throughput | tokens/s individual y agregado, prefill y decode separados |
| Scheduler | esperas por slots, esperas por KV, cancelaciones y starvation |
| Estabilidad | OOM, crash, restart, respuestas corruptas y reservas filtradas |
| Calidad operativa | transiciones evitadas y tiempo no disponible ahorrado |

Las métricas estimadas por oCabra y las observadas en llama.cpp deben guardarse
por separado hasta validar su equivalencia.

## Criterios de aceptación

### Pool flexible en una GPU

- Cuatro slots físicos arrancan y procesan cuatro peticiones solapadas.
- `4×32k` y al menos una combinación desigual se sirven sin recargar el modelo.
- El mayor pool de la 3090 queda fijado por prueba sostenida y conserva margen
  operativo medido.
- Existe una fila concluyente para cada combinación de los tres modelos y las
  tres colocaciones, incluso cuando el resultado sea `no viable full-offload`.
- Ninguna petición admitida supera la capacidad reservada del pool.
- No hay OOM, crash ni reserva filtrada durante 12 horas de carga mixta.
- No aparece contaminación entre conversaciones en al menos 10.000 peticiones
  automatizadas con identificadores cruzados.
- El throughput de usuario único no cae más de un 10 % frente al baseline sin
  KV unificada. Si cae más, se necesita una decisión explícita basada en el
  throughput agregado y la latencia.
- El throughput agregado con cuatro usuarios no es inferior al baseline del
  perfil rígido comparable.

### Recreación de contexto

- No se releen ni transfieren de nuevo los pesos principales cuando coincide
  `resident_key`.
- No se aceptan peticiones durante la ventana de reconfiguración.
- El nuevo contexto publica capacidad y slots reales antes de reabrir tráfico.
- Un fallo deja un runtime anterior recuperado o un estado de error explícito;
  nunca un proceso aparentemente cargado pero inutilizable.
- La mediana de reconfiguración es al menos tres veces más rápida que la carga
  completa con page cache caliente.

### Spill entre GPU

- Solo ocurre cuando el presupuesto no cabe en la configuración validada de
  una GPU o cuando una política explícita justifica la espera evitada.
- No desaloja ni duplica el runtime equivocado ante solicitudes concurrentes.
- El downgrade respeta residencia mínima, histéresis y trabajo activo.
- La capacidad pública nunca refleja el runtime futuro antes de estar listo.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Contaminación de prompt/KV entre slots | Desactivar prompt cache en el spike, identificadores cruzados y soak tests |
| Degradación CUDA con KV unificada | Comparar por forma de carga y limitar slots por modelo |
| Fragmentación tras cargas desiguales | Métricas de huecos/ocupación y secuencias largas repetidas |
| Estado recurrente o MTP multiplicado por slots | Medición específica de Qwen3.8 y draft contexts |
| Estimación corta de prompt/salida | Tokenización exacta cuando sea posible y margen conservador |
| Reservas demasiado conservadoras | Medir utilización antes de diseñar overcommit |
| Pool que arranca pero falla en prefill | Capacidad aceptada solo tras carga máxima sostenida |
| Doble asignación al liberar/cancelar | Lease con identidad y liberación idempotente |
| Fork difícil de mantener | Parches pequeños, tests upstream y rebase controlado |
| Cambio upstream incompatible | Pin de commit y matriz mínima tras cada actualización |
| Reconfiguración sin memoria para rollback | Admission previo, error explícito y fallback a recarga completa |

## Rollback

- Los canaries iniciales fueron opt-in por modelo/perfil.
- Los presets rígidos ya no forman parte del rollback productivo.
- Para volver atrás, drenar y cargar una configuración fija con un pool menor;
  no crear workers derivados por tamaño de petición.
- Desactivar KV unificada requiere volver explícitamente a la semántica de
  contexto por slot y recalcular ``ctx_size``; no es un cambio transparente.
- Un fallo de observabilidad upstream no debe impedir usar reservas estimadas.
- Un fallo de recreación de contexto debe caer en recarga completa, no en
  reintentos ilimitados.

## Ficheros previstos

Esta lista orienta ownership; se concretará antes de cada fase y cualquier
cambio de contrato se documentará primero en `docs/CONTRACTS.md`.

| Área | Ficheros probables |
|---|---|
| Config llama.cpp | `backend/ocabra/schemas/backend_load.py`, `backend/ocabra/backends/llama_cpp_backend.py`, `backend/ocabra/workers/llama_cpp_worker.py` |
| Capacidad y presets | `backend/ocabra/core/vram_planner.py`, `runtime_preset_capacity.py`, `runtime_coordinator.py` |
| Admission control | `backend/ocabra/core/fair_slot_gate.py`, `worker_pool.py`, rutas OpenAI/Ollama compartidas |
| Ciclo de vida | `backend/ocabra/core/model_manager.py`, backend llama.cpp y process manager |
| Observabilidad | estado de modelos, stats, WebSocket y Dashboard |
| Tests oCabra | `backend/tests/test_llama_cpp_*`, `test_worker_pool.py`, `test_runtime_*`, nuevos tests del gate ponderado |
| Fork llama.cpp | `common/common.*`, `tools/server/server-context.*`, rutas, schemas y tests de servidor |

No se modificarán simultáneamente ficheros de distintos streams sin coordinar
ownership conforme a `AGENTS.md`.

## Registro de benchmarks

Cada ejecución debe añadir una fila y enlazar resultados detallados cuando los
haya. No se reemplazarán resultados negativos.

| Fecha | Commit oCabra | Commit llama.cpp | Modelo | GPU | Pool | Slots | Carga | Resultado | Evidencia |
|---|---|---|---|---|---:|---:|---|---|---|
| 2026-10-30 | `03e9a08` | `74bca395` | Gemma4 26B | 3090 | 128k total observado | 2 | Carga ordinaria | 42 s fría preliminar; 10,5 s caliente preliminar | Logs, repetir en Fase 0 |
| 2026-10-30 | árbol de trabajo | `74bca395` | Gemma4 12B | 3090 | 128k | 4 | 4 prompts cortos concurrentes | Arranca y aísla slots; 10.565 MiB observados | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-30 | árbol de trabajo | `74bca395` | Gemma4 26B | 3090 | 128k | 4 | 4 prompts cortos concurrentes | Arranca y aísla slots; 21.165 MiB observados | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-30 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 128k | 4 | Carga con self-MTP y mmproj | OOM al cargar projector; probar 96k | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Gemma4 12B | 3090 | 128k | 4 | Visión bajo demanda y dos imágenes concurrentes | Una activación, +408 MiB; respuestas correctas en slots 0 y 1 | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 96k | 4 | Visión bajo demanda y dos imágenes concurrentes | Una activación, +1.216 MiB; respuestas correctas en slots 0 y 1 | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 96k | 4 | Grounding 1280×960, `ubatch=128` | OCR y coordenadas correctos; dos peticiones concurrentes, ~727 MiB libres | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 96k | 4 | Grounding 1280×960, `ubatch=1024` | Correcto pero sin mejora y solo 119 MiB libres; descartado | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 96k | 4 | 40 peticiones mixtas y cuatro cancelaciones | 44/44 correctas incluyendo recuperación; sin OOM/reallocs | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Gemma4 12B | 3090 | 128k | 4 | 20 peticiones mixtas y cuatro cancelaciones | 24/24 correctas incluyendo recuperación; sin OOM/reallocs | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | árbol de trabajo | `74bca395` | Qwen3.8 27B | 3090 | 96k | 4 | Soak acelerado de 1.000 peticiones y cancelación final | 1.000/1.000; drenado 375 ms; recuperación 4/4; sin OOM/reallocs | `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |

## Registro de decisiones

| Fecha | Decisión | Motivo | Revisión |
|---|---|---|---|
| 2026-10-30 | Comenzar con cuatro slots físicos | Objetivo de concurrencia y base para comparar otros valores | Tras Fase 3 |
| 2026-10-30 | No limitar a priori la 3090 a 128k | La capacidad real puede ser mayor y depende del modelo/runtime | Tras Fase 0 |
| 2026-10-30 | Pool compartido y reservas por petición | Evita que slots inactivos desperdicien una partición fija | Tras Fase 3 |
| 2026-10-30 | Upstream-first | La funcionalidad es genérica y se quiere minimizar deuda del fork | En cada patch llama.cpp |
| 2026-10-30 | Spill solo cuando una GPU no alcance | Mantener libre la 3060 para otros modelos en el caso normal | Tras Fase 6 |
| 2026-10-30 | Fork local separado en `/docker/llama.cpp-ocabra` | Mantener limpios los cambios upstreamables y separarlos de la integración Python | En cada rebase |
| 2026-10-30 | Bajar Qwen3.8 a 96k/4 en la próxima ventana | 128k/4 deja menos de los 887,99 MiB que necesita cargar el projector tras pesos, KV y self-MTP | Tras el siguiente smoke |
| 2026-10-30 | No portar directamente el PR upstream `#25963` | Ya cubrió el concepto y midió la ganancia, pero está cerrado, cambia 964 líneas frente al pin actual y precede a refactors importantes | Tras medir recarga de contexto en una rama experimental mínima |
| 2026-10-30 | Pool completo visible para cada slot | En KV unificada el pool físico no se multiplica; el cap por slot es lógico y el reparto agregado pertenece al admission control de oCabra | Tras smoke sin cap y carga larga |

## Registro de implementación

| Fecha | Unidad | Resultado | Verificación |
|---|---|---|---|
| 2026-10-30 | Flags de KV unificada y prompt cache | Schema, backend y worker soportan tri-state, snake/camel y rangos sin cambiar defaults | 49 tests focalizados y `ruff check` |
| 2026-10-30 | Smoke llama-server con KV unificada | Qwen2.5 0.5B CPU arrancó con pool 8192, 4 slots de 2048; cuatro peticiones concurrentes usaron slots 0-3 | `/props`, `/slots`, logs y cierre del puerto 19099 |
| 2026-10-30 | Smokes GPU Gemma 4 | 12B y 26B arrancaron con pool 128k y 4 slots; cuatro peticiones simultáneas quedaron aisladas en slots 0-3 | `/props`, `/slots`, logs, VRAM y restauración saludable del runtime productivo |
| 2026-10-30 | Límite inicial Qwen3.8 | 128k/4 con self-MTP y mmproj no cabe en 24 GiB; fallo reproducible antes de servir | OOM CUDA de 887,99 MiB al cargar projector; GPU liberada sin proceso huérfano |
| 2026-10-31 | Smoke de pool flexible sin cap | Gemma 4 12B expuso 128k en cada uno de cuatro slots con una sola reserva física de 10.565 MiB; cuatro peticiones quedaron aisladas | `/slots`, logs 0-3, VRAM antes/después y restauración saludable de producción |
| 2026-10-31 | Barrido Qwen3.8 con/sin visión | Con mmproj: 116k carga limpio, 120k sirve degradado y 124k aborta; sin mmproj: 140k sirve cuatro prompts y 144k aborta | VRAM y logs por escalón en `docs/benchmarks/llama_cpp_elastic_kv_baseline.md`; producción restaurada |
| 2026-10-31 | Visión configurable end-to-end | Gemma 4 26B validó `enabled`, `disabled` y activación `on_demand`; drenado con texto y dos primeras imágenes concurrentes fueron correctos | 150 tests backend del área y 4 del modal UI; HTTP, logs, `/props`, VRAM y `docs/benchmarks/llama_cpp_elastic_kv_baseline.md` |
| 2026-10-31 | Campaña disruptiva de capacidad | Gemma 12B y 26B completaron 200k sobre 256k/4; Gemma 12B validó 128k/4 en 3060; Qwen validó 128k/4 con visión en spill 1:2 | Prefills reales, concurrencia 1-6, imagen real, VRAM por GPU y restauración productiva |
| 2026-10-31 | Sobresuscripción y errores visuales | El exceso de KV serializa y purga en vez de rechazarse; OOM de hot-mmproj aborta; projector incompatible fuga VRAM hasta reinicio | Logs de llama-server y recuperación limpia; candidatos de fix/RFC upstream |
| 2026-10-31 | Canary WeightedSlotGate Gemma 26B | Pool 128k/4: 89.530 activos + 50.026 en cola por KV; otro usuario usó 20.026 del remanente; cancelaciones liberaron exactamente y oversized devolvió 400 | Snapshots de `/status`, evento `kv_queued`, recuperación HTTP 200 y restauración del runtime productivo |
| 2026-10-31 | Semántica planner/presets verificada | Presets expresan contexto por petición y construyen `ctx_size=contexto*slots`; configuración fija expresa directamente el pool total | Qwen 3.8 arrancó realmente como 32k/p3 → `--ctx-size 98304 --parallel 3`; regresión restaurada |
| 2026-10-31 | Activación productiva acotada | Se promovieron los perfiles llama.cpp custom de Gemma, Qwen Coder, Qwen 3.8 y RavenX; Ollama queda accesible con sufijo `-ollama` | Smokes de texto de seis modelos, visión hot en tres, procesos/VRAM, listado público y chat final HTTP 200 |
| 2026-11-01 | Escalado adaptativo por presets de colocación | El contexto es un pool agregado unificado; los presets ya no lo multiplican por slots y solo recrean el worker cuando cambia la colocación | Abliterated completó 64k/1/3090 → 224k/4/dual → 64k/1/3090; visión y restauración automática correctas |
| 2026-11-01 | Capacidad validada por preset | Una combinación exacta de contexto y colocación medida puede usar la VRAM física completa sin que el margen genérico del planner la rechace | Coder 116k/4 estable con 120 MiB libres; 120k/4 descartado por OOM de 285,08 MiB en el grafo CUDA |
| 2026-11-01 | Pico VRAM calibrado por preset | `validated_vram_mb` sustituye la estimación genérica solo si también coincide el contexto validado exacto | Qwen normal estimaba 24.884 MiB y era rechazado antes de arrancar; con 24.000 MiB calibrados cargó 96k/4 en GPU1 y respondió HTTP 200 |

## Configuración productiva unificada y adaptativa activada el 2026-11-01

``ctx_size`` es el pool KV físico agregado, los slots ven el pool completo y
oCabra concede reservas por la suma estimada de prompt más salida. No existe
un reparto fijo de KV por slot. Todos usan KV Q8 unificada,
``cache_idle_slots=false`` y continuous batching. Gemma y RavenX caben ya con
su contexto máximo en una 3090; los Qwen usan una colocación normal que llena
la 3090 y otra extendida que drena vecinos y reparte pesos/KV entre ambas GPU:

| Perfil | Normal | Colocación normal | Extendido | Colocación extendida |
|---|---:|---|---:|---|
| `gemma4:12b` | 256k/4 | RTX 3090 | — | — |
| `gemma4:26b` | 256k/4 | RTX 3090 | — | — |
| `qwen3.8:27b` | 96k/4 | RTX 3090 | 256k/4 | RTX 3060+3090, 1:3 |
| `qwen3.8:27b-abliterated` | 64k/1 | RTX 3090 | 224k/4 | RTX 3060+3090, 1:2,3 |
| `qwen3-coder:30b` | 116k/4 | RTX 3090 | 256k/4 | RTX 3060+3090, 1:3 |
| `ravenx` | 240k/4 | RTX 3090 | — | — |

Compatibilidad y rollback:

- Las alternativas públicas son `gemma4:12b-ollama`, `gemma4:26b-ollama`,
  `qwen3-coder:30b-ollama`, `qwen3.8:27b-ollama` y
  `qwen3.8:27b-abliterated-ollama`.
- `ravenx-256k` permanece como alias oculto hacia `ravenx`.
- Gemma usa un worker canónico. Los Qwen con preset extendido vuelven al
  normal tras 300 segundos de gracia y 120 segundos de residencia mínima;
  una petición que cabe en el normal no provoca escalado.
- Gemma 26B usa `vram_estimate_mb=23632`, calibrado contra los 23.614 MiB
  observados a 256k/4 con visión. El override evita que la estimación genérica
  de 26.054 MiB fuerce spill innecesario y mantiene la 3060 disponible para
  Whisper y embeddings.
- La reserva dual valida cada shard de `tensor_split`, no solo la suma de VRAM.
  La colocación forzada de una GPU tampoco puede ensancharse silenciosamente a
  spill por una estimación conservadora.

Smokes reales completados antes de promover etiquetas y capacidades:

- Gemma 12B y 26B: texto y activación visual caliente HTTP 200.
- Qwen Coder 30B: 116k/4 en una 3090, respuesta exacta `323`; 120k/4 falla al
  reservar 285,08 MiB para el grafo CUDA, por lo que no se anuncia.
- Qwen 3.8 estándar: 128k/4 en spill 1:2, self-MTP, texto y visión HTTP 200.
- Qwen 3.8 abliterated: 128k/4 en spill 1:2, texto HTTP 200 y visión
  validada con grounding real (`D4`). Requiere `main_gpu=1` para alojar el
  buffer de cómputo en la 3090; el reparto por defecto agotaba la 3060.
- RavenX: 128k/4 en una 3090, texto HTTP 200.
- Regresión ampliada del área: 323 tests verdes y uno excluido por persistencia;
  API desplegada como `0.22.5`.

Estado operativo al cierre (2026-11-01):

- Gemma 26B quedó restaurado como modelo `pin` en GPU1 con el runtime canónico
  256k/4. El proceso real usa `--ctx-size 262144 --parallel 4 --kv-unified`;
  `/props` y los cuatro `/slots` exponen 262.144, y el smoke final respondió
  `OK`.
- Qwen abliterated escaló en vivo desde 64k/1 en GPU1 hasta 224k/4 dual,
  desalojando de forma drenada Gemma, Whisper y embeddings cuando fue
  necesario. Texto e imagen devolvieron HTTP 200. Tras la gracia volvió a
  64k/1 en GPU1 y dejó GPU0 libre.
- Se corrigió en `ServiceManager.refresh()` la fuga de una reserva no
  materializada cuando el contenedor del servicio se detiene durante el
  arranque. Una reserva se conserva si el contenedor sigue vivo, para tolerar
  health checks transitorios. Ambos caminos tienen tests de regresión.
- La admisión por presión hace además un preflight no destructivo: si VRAM
  libre + toda la VRAM recuperable sigue por debajo del requisito, no descarga
  ningún candidato. Se validó en vivo con Qwen abliterated (29.656 MiB
  requeridos, 26.437 MiB máximos): HTTP 409 y Gemma 26B continuó cargado.
- Barrido ampliado posterior: los cuatro GGUF declaran 262.144 tokens nativos.
  Qwen Coder completó 256k/4 y texto con split 1:3; Qwen 3.8 estándar
  completó 256k/4, visión y grounding con 1:3; RavenX completó 240k/4 en
  una 3090. Abliterated carga 240k y visión, pero deja solo 47 MiB en GPU0;
  su perfil productivo queda en 224k/4 con split 1:2,3 y 675/562 MiB libres
  tras inferencia visual.
- La descarga limpia ahora locks por `model_id` aunque el backend pierda
  temporalmente la atribución de GPU (`gpu=[]`). Esto eliminó una reserva
  fantasma de 5.903 MiB observada al descargar Qwen Embedding antes del
  barrido dual.
- La admisión reconcilia el inventario físico Ollama antes y después del
  desalojo, adopta pins ya residentes sin recargarlos, espera la liberación
  física completa y hace una última pasada basada en el déficit real. Esto
  cubre las carreras observadas de madrugada: cambios de estado durante el
  drain, muestras parciales de VRAM y residentes duplicados en la estimación.

### Perfiles RavenX experimentales añadidos el 2026-11-01

Se descargaron, verificaron y registraron como ``on_demand`` dos GGUF Q4_K_M:

| Perfil público | Normal | Colocación normal | Extendido | Colocación extendida |
|---|---:|---|---:|---|
| ``ravenx-chaos-agent`` | 96k/4 | RTX 3090 | 256k/4 | RTX 3060+3090, 1:3 |

- Los dos usan KV Q8 unificada, continuous batching y el worker del fork
  custom. Se validaron carga, inferencia y cuatro peticiones concurrentes.
- ``ravenx-chaos-agent`` no contiene capas MTP y ninguno de los dos repos
  incluye ``mmproj``; por tanto MTP y visión quedan desactivados, sin anunciar
  capacidades que los artefactos descargados no pueden servir.
- Chaos requiere pensamiento desactivado según su model card. Su perfil usa
  ``--jinja --reasoning off --reasoning-format none``, ``temperature=0`` y
  ``repeat_penalty=1.15``. En llama.cpp actual ``reasoning-format=none`` solo
  controla el parser; ``reasoning=off`` es lo que evita consumir la salida en
  bucles de razonamiento. El smoke produjo la respuesta pedida en 6 tokens,
  sin ``reasoning_content``.
- El recorrido normal → extendido → descarga → normal de Chaos quedó probado.
  El preset extendido cargó con ``--ctx-size 262144 --parallel 4`` y reparto
  ``--tensor-split 1,3``.
- ``openfable-coder`` se retiró del catálogo y del almacenamiento el 2026-11-01
  tras mostrar contaminación fuerte de identidad, baja obediencia a salida
  exacta, código incorrecto y argumentos de herramientas corruptos.
- Chaos mantiene thinking desactivado por defecto, pero oCabra 0.22.9 permite
  activarlo por petición con ``reasoning_effort=low|medium|high`` sin recargar:
  la traducción también fuerza ``chat_template_kwargs.enable_thinking=true``.
  ``default`` conserva el arranque en off y ``off``/``none`` lo desactiva para
  la petición. El smoke de ``default`` produjo 3 tokens y ningún
  ``reasoning_content``; el smoke de ``low`` se canceló al entrar tráfico real
  y queda pendiente repetirlo en una ventana tranquila.
- Esa cancelación descubrió una carrera: ``CancelledError`` durante el
  health-check de carga dejaba un llama-server huérfano reteniendo 19,7 GiB.
  Corregido en 0.22.9: backend y model manager limpian proceso, puerto y estado;
  hay test de regresión específico.
  Queda disponible como experimental, no como sustituto del Gemma principal.
- Tras las pruebas se descargaron ambos workers experimentales y se comprobó
  la restauración automática de ``gemma4:26b`` como ``pin``.

### Decisión sobre liberación dinámica de KV en colocación dual

El fork actual no puede redimensionar un `llama_context` vivo: hacerlo bien
requiere drenar, recrear contexto, slots, samplers, MTP y estado multimodal,
con rollback si falla la nueva reserva. Además, reducir solo la KV mientras
los pesos siguen repartidos no libera GPU0 por completo y normalmente no deja
espacio suficiente para los servicios vecinos. Por ello la operación
productiva es dual → single, que libera íntegramente la 3060. La recreación de
contexto conservando pesos permanece en Fase 5 como optimización futura para
cambios de pool que mantengan la misma colocación.

### Observabilidad y controles operativos (2026-11-02)

- La proyección del modelo canónico incluye ahora workers derivados en
  ``loading``, ``loaded``, ``unloading`` y ``error``. Esto corrige el caso en
  que Chaos atendía o se estaba cargando pero no aparecía en la UI.
- La resolución conserva el separador propio del artefacto GGUF
  ``repositorio::fichero`` y solo elimina el sufijo final del worker. El corte
  antiguo por el primer ``::`` era la causa concreta de que Chaos no se
  asociase a su entrada canónica.
- El contrato expone ``runtime_phase`` (incluidos ``upsizing`` y
  ``downsizing``), ``runtime_worker_id``, GPUs objetivo, slots físicos y las
  acciones que son seguras en ese instante.
- La UI muestra siempre los slots, refresca la proyección canónica al recibir
  eventos de un worker ``::<hash>`` y ofrece cancelar una carga o restaurar el
  preset de una GPU. El downsize rechaza runtimes ocupados y también cualquier
  perfil que no tenga un destino single-GPU verificable antes de descargarlo.
- La cancelación usa la tarea propietaria de ``_load_model`` para conservar la
  limpieza de proceso, puerto y estado ya probada; no modifica un estado
  ``LOADING`` por fuera de la corrutina.
- El normalizador de salida elimina un bloque inicial ``<think></think>`` solo
  cuando está vacío, incluso si llega partido entre chunks SSE. El razonamiento
  no vacío se conserva.

### Tráfico real de Chaos y admisión adaptativa (2026-11-02)

- En 1.406 peticiones del 2026-11-01, dos concurrentes en el runtime extendido
  elevaron el throughput agregado estimado de 19,1 a 26,4 tok/s con una
  penalización moderada por stream. Con tres, el p95 subió de 15,4 a 60,6 s y
  cinco generaciones cayeron por debajo de 5 tok/s.
- El runtime conserva ``--parallel 4`` y el pool KV completo. El gate limita
  sólo la concurrencia efectiva: 4 para prompts pequeños, 3 desde 32k y 2
  desde 64k. Un prefill grande ya activo limita también admisiones pequeñas
  posteriores.
- ``request_stats.runtime_telemetry`` registra prompt estimado, reserva KV,
  espera de gate, carga, TTFT, prefill/slot cuando el engine los informa,
  slots físicos, límite efectivo y concurrencia al conceder el lease. La
  tabla de peticiones recientes muestra un resumen y conserva el detalle en
  tooltip.
- La campaña normal de Chaos probó 104k, 112k, 120k y 128k con KV Q8 y
  cuatro slots físicos. Los cuatro cargaron y generaron; 128k consumió un
  pico observado de 20.990 MiB en la RTX 3090 y dejó unos 3.188 MiB libres.
  El perfil normal queda promovido a 131.072 tokens, con el runtime extendido
  de 262.144 sin cambios.
- En 128k, cuatro peticiones con reserva estimada de 39.898 tokens concedieron
  exactamente tres leases (concurrencias persistidas 1, 2 y 3); la cuarta
  esperó 62.986 ms antes de entrar. Tres peticiones en torno a 64k confirmaron
  la interacción con el gate KV. En un pool de 128k, dos reservas de al menos
  64k más salida no caben simultáneamente, por lo que el cap computacional 2
  sólo es independiente del gate KV en perfiles mayores como el extendido.
- El smoke de streaming verificó persistencia real de carga, TTFT, cola,
  prefill y admisión. La primera ejecución detectó que un stream completo en
  menos de 500 ms podía eludir TTFT; se corrigió y el segundo smoke registró
  ``load_ms=93379``, ``ttft_ms=93772`` y ``prefill_ms=188``.

### RavenX real: leases y contención de prefill (2026-11-03)

- Una ráfaga de 21 peticiones a RavenX completó 20 y rechazó preventivamente
  una reserva de 267.693 tokens sobre un pool de 245.760, sin OOM ni caída del
  worker. Con dos admisiones, la mediana observada fue 71,5 tok/s por petición
  frente a unos 118 tok/s aislada.
- Un prompt estimado en 96.812 tokens coincidió con otro de 7.578 y elevó el
  TTFT del pequeño a 47,8 s. El gate conserva cuatro slots físicos, pero desde
  esta revisión serializa la fase de prefill cuando interviene un prompt de al
  menos 32k; al primer chunk abre de nuevo los slots restantes para decode.
- Se localizó un stream iniciado a las 01:54:52 que adquirió capacidad pero no
  llegó a ``request_stats`` y dejó ``in_flight=1`` en admisiones posteriores.
  La causa era el cierre incompleto del iterador interior al cancelar el tee
  del middleware. Ahora el tee ejecuta ``aclose()``, los leases tienen identidad
  y liberación idempotente, y el gate recupera una tarea propietaria terminada
  antes de admitir trabajo nuevo.
- La cobertura automatizada reproduce desconexión después de cabeceras,
  cancelación directa, doble liberación tardía, recuperación de propietario y
  apertura de concurrencia decode al primer chunk.

## Cuestiones abiertas

- [ ] ¿Qué margen productivo aplicar a los máximos ya medidos (Gemma 12B y
  26B 256k/4; Qwen 96k/4 local o 128k/4 en spill)?
- [ ] ¿Cuánto penaliza KV unificada al usuario único con uno, cuatro y seis
  slots físicos?
- [x] El límite lógico por slot es el pool completo; el límite agregado lo
  aplica ``WeightedSlotGate``.
- [ ] ¿La ocupación real expuesta por llama.cpp es suficientemente estable para
  decisiones de admission o solo para observabilidad?
- [ ] ¿Qué parte del contexto speculative/MTP puede recrearse sin tocar pesos?
- [ ] ¿Puede mmproj permanecer cargado durante la recreación del contexto?
- [ ] ¿Qué umbral de espera justifica escalar a dos GPU por demanda agregada?
- [ ] ¿Qué cambios aceptaría upstream como PR y cuáles requieren primero RFC?

## Referencias

- Contrato de runtimes adaptativos: `docs/CONTRACTS.md`, sección 8.9.
- Backlog de presets actuales: `docs/ROADMAP.md`, Bloque 21.
- Loader llama.cpp: `docs/tasks/llama-cpp-parity-plan.md`.
- Documentación upstream de `llama-server`:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Guía de desarrollo de `llama-server`:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md>
- Normas de contribución y política de IA:
  <https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md>
- Convenciones para agentes, estilo y tests:
  <https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md>
- Tests de `llama-server`:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/tests/README.md>
- Slots separados de concurrencia, discussion `#22401`:
  <https://github.com/ggml-org/llama.cpp/discussions/22401>
- Pool unificado y límite por slot, discussions `#22658` y `#27916`:
  <https://github.com/ggml-org/llama.cpp/discussions/22658>,
  <https://github.com/ggml-org/llama.cpp/discussions/27916>
- Runtimes reconfigurables y prototipo cerrado de context reinit:
  <https://github.com/ggml-org/llama.cpp/discussions/25674>,
  <https://github.com/ggml-org/llama.cpp/pull/25963>
- Prototipo anterior de tiers con recarga (`--ctx-dynamic`) y sus commits
  históricos dentro de la PR posteriormente reutilizada `#20872`:
  <https://github.com/ggml-org/llama.cpp/commit/934ec480ddc8f7887986bb6ae0dc162a89e36814>,
  <https://github.com/ggml-org/llama.cpp/commit/9292b466a16101095bb799a0487f733ff814fdf0>,
  <https://github.com/ggml-org/llama.cpp/pull/20872>
- Dependencia original del context reinit (`n_ctx`/fit), PR `#25465`:
  <https://github.com/ggml-org/llama.cpp/pull/25465>
- Paged KV experimental, discussion `#21961`:
  <https://github.com/ggml-org/llama.cpp/discussions/21961>
- Prompt cache y aislamiento, issue `#27148`:
  <https://github.com/ggml-org/llama.cpp/issues/27148>
- Rendimiento/restauración de KV unificada, issues `#30252` y `#30961`:
  <https://github.com/ggml-org/llama.cpp/issues/30252>,
  <https://github.com/ggml-org/llama.cpp/issues/30961>
- Cambio abierto en el cap por slot, PR `#31377`:
  <https://github.com/ggml-org/llama.cpp/pull/31377>
