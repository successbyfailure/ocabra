# Plan: rediseño del dashboard

**Fecha de apertura:** 2026-11-09
**Estado:** v0.61.1 desplegada y verificada en producción (2026-11-09) · v0.61.2 con commit, pendiente de desplegar
**Objetivo:** que el dashboard enseñe lo que está pasando *ahora* (peticiones en
curso, cola, contexto, GPUs) en vez de sobre todo lo que ya pasó. El histórico
se queda en `/stats`.

---

## Decisiones tomadas

1. **Card por modelo activo**, con:
   - Nombre, **backend con su icono** (reutilizar `components/models/BackendIcons.tsx`,
     ya usado en Explore y Models; no hacer un segundo set de iconos), estado,
     preset o runtime activo y colocación (qué GPU o GPUs).
   - Totales de lo que está procesando: peticiones en curso y en cola, t/s
     agregados de prefill y de generación.
   - **Lista de peticiones en curso**, cada una como una barra de progreso:
     fase (cola → prefill → generación), t/s de prefill y de generación, y
     avance real cuando lo haya o estimado cuando no (ver "Fuentes de progreso").
   - **Barra de contexto** (ver sección propia).
2. **Sección de cola**: el orden real en que se van a atender (el del
   `FairSlotGate`, que va por "menos servido primero", no por orden de
   llegada), el tiempo estimado de espera y el tiempo estimado de proceso.
3. **Widget de GPU y CPU**: iconos y barras de potencia, uso y memoria; los
   modelos cargados en cada dispositivo con su estado, sus peticiones en curso
   y en cola y los t/s totales de prefill y generación.
4. **Se retiran** las secciones "Active agents" (ya tienen su página) y
   "Descargas activas". Propuesta pendiente de confirmar: un indicador pequeño
   en la cabecera mientras haya alguna descarga en curso, para no perder de
   vista una descarga atascada.
5. **Últimas peticiones mejoradas**, aprovechando lo que ya publica
   `/ocabra/stats/recent` desde la v0.60.0: espera, t/s de prefill y de
   generación, tokens cacheados, contexto efectivo, preset y GPU.
6. **Estadísticas del dashboard** = "ahora + última hora". Candidatas: TTFT y
   espera en cola (p50 y p95) por modelo, porcentaje de 429 y 503, ratio de
   cache hit (`cachedTokens`), tokens por vatio, y peticiones colgadas o
   canceladas.

## Barra de contexto

### Qué representa

No es cómo se reparte la memoria entre tarjetas (en tensor-split la KV se
reparte por capas, así que cada token ocupa memoria en las dos GPUs). Es
**cuánto contexto cabe en cada colocación**. Ejemplo real, qwen3.8:27b:

```
qwen3.8:27b · llamacpp_mks · 3090 (1 GPU) · pool 96k / 4 slots
[████▓▓▓▓░░░░░░░░░░░|············································]
 0   req A  req B  libre   96k ↑ tope 1 GPU                256k
                                 zona rayada = solo con escalado a 2 GPU
```

- Escala = el mayor contexto de cualquier colocación (aquí 256k, en 2 GPU).
- Marca = el tope de la colocación de una GPU (aquí 96k).
- Con el modelo en una GPU, lo que hay por encima de la marca se pinta rayado:
  solo está disponible con una transición drenada (Fase 6 del plan de KV
  elástica).
- Con el modelo en dos GPU, toda la barra es utilizable y la marca se queda:
  indica a partir de dónde el modelo puede volver a una GPU
  (`downgrade_grace_seconds`, `minimum_residency_seconds`).
- Relleno claro = reservado por la admisión (el presupuesto completo de cada
  petición). Relleno sólido = ocupado de verdad, cuando la fuente lo dé.

### Niveles reales en producción (consultados el 2026-11-09)

Fuente: `GET /ocabra/profiles/{id}/runtime-presets/resolved`, que sale del
mismo `runtime_preset_capacity` que usa `plan_profile_runtime` para admitir. Así
la barra nunca promete un contexto que la admisión rechace. Esto ya nos ha
pasado: la deriva de slots y la guarda de presets.

| Perfil | Niveles (contexto / slots / GPU) |
|---|---|
| qwen3.8:27b | normal 96k/4 [1] · extended 256k/4 [0,1] |
| qwen3.8:27b-abliterated | normal 16k/4 [1] · extended 224k/4 [0,1] |
| qwen3:32b | normal 30k/4 [1] · dual-gpu 40k/4 [0,1] |
| qwen3-coder:30b | normal 116k/4 [1] · extended 256k/4 [0,1] |
| flint:40b | normal 64k/4 [1] · extended 256k/4 [0,1] |
| ornith-1.5:35b | normal 128k/4 [1] · extended 256k/4 [0,1] |
| ravenx-chaos-agent | normal 128k/4 [1] · extended 256k/4 [0,1] |
| nemotron-3-nano:30b | dual-gpu 1024k/4 [0,1] (un solo nivel) |
| nemotron-3.5-lightning:30b | normal 128k/4 [0,1] |
| qwen3:8b, bonsai-2:27b | un solo nivel en [1] |
| gemma4:26b, gemma4:12b, ravenx | sin presets → un solo nivel = `ctx_size` del modelo |
| perfiles `-ollama` | escalera rígida 32k/64k/128k/max con 1-2 slots |

Los niveles cuyo `fits` sea `false` no se pintan.

### Modos por backend

Comprobado el 2026-11-09: **los 17 modelos `llamacpp_mks` tienen
`kv_unified=True`**, así que el modo "pool" es el caso normal.

| Modo | Backends | Barra | Marca 1/2 GPU |
|---|---|---|---|
| `pool` | llamacpp_mks (`kv_unified`) | Un segmento por petición dentro del pool; datos reales de `/slots` | Real: umbral de escalado entre presets |
| `slots` | llama.cpp sin `kv_unified`, Ollama | N parcelas fijas (`ctx/parallel`); cambiar de nivel = recarga | Sale de la escalera de presets |
| `paged` | vLLM | Porcentaje de uso del pool, sin segmentos por petición | No aplica |
| `null` | whisper, diffusers, TTS, embeddings… | Sin barra | — |

### Contrato propuesto (normalizado en el backend; el frontend solo lo pinta)

```
context: {
  mode: "pool" | "slots" | "paged" | null,
  scaleMaxTokens,          // mayor nivel que cabe (256k)
  activeMaxTokens,         // nivel cargado ahora (96k o 256k)
  activePresetKey,
  tiers: [{presetKey, gpus:[1], maxTokens, slots}],
  segments: [{requestId, reservedTokens, usedTokens|null}],
  usagePct,                // solo paged
  source: "live" | "estimate"
}
```

## Fuentes de progreso por petición

| Señal | Dónde está | Estado |
|---|---|---|
| Lista de peticiones en vuelo | `mm.active_requests_snapshot()` | Existe |
| Tokens de salida en vivo y última actividad | `ActiveRequest.output_tokens_estimate` y `last_activity_at` (tee de streaming, ~2 Hz) | Existen pero **no se publican**: `/stats/recent` manda `inputTokens` y `outputTokens` a `null` en las filas en vuelo |
| Orden real de la cola | `worker_pool.peek_gate_priority` | Existe (ya lo usa `/stats/recent`) |
| Pool KV reservado y total | `FairSlotGate.snapshot()` → `kv_capacity_tokens`, `kv_reserved_tokens` | Existe |
| Fase prefill/generación | Momento del primer token | **Falta**: añadir `first_token_at` en el tee |
| Prefill real en curso (llama.cpp) | `GET /slots` del llama-server: `n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`, `next_token.n_decoded`, `n_remain`, `is_processing` | **Verificado en vivo** en llamacpp_mks (qwen3-coder:30b, slot a mitad de prefill de 105k) |
| `/metrics` del llama-server | — | Devuelve 501 (no se lanza con `--metrics`) |
| Ollama, vLLM | Sin progreso de prefill desde fuera | Barra estimada hasta el primer token |
| Peticiones sin streaming | Nada que leer hasta el final | Solo estimación, marcada como "estimado" en la UI |

### Problema abierto: emparejar slot y petición

`/slots` no sabe cuál es la petición de oCabra (`id_task` es interno del
llama-server). Opciones, por orden de preferencia:

1. **`return_progress: true` en el cuerpo de la petición en streaming**: el
   llama-server upstream emite `prompt_progress` en los chunks y el tee lo lee.
   Así el emparejamiento es exacto y no hace falta polling. *Falta verificar
   que el fork (743e3ee) lo soporta y que el chunk no rompe a los clientes:
   habría que quitarlo antes de reenviarlo.*
2. Fijar `id_slot` desde oCabra: emparejamiento exacto, pero se pierde la
   elección de slot por similitud de prompt cache, que hace el llama-server. No
   recomendado.
3. Emparejar de forma heurística por número de tokens del prompt y orden de
   admisión. Sirve solo de respaldo.

La ocupación agregada del pool (suma de `n_prompt_tokens + n_decoded` de los
slots activos) no necesita emparejamiento y vale para el relleno sólido de la
barra.

## Estimaciones

- La espera en cola de hoy es `max_restante + turnos × p50` del modelo
  (`api/internal/stats.py`). Mejora prevista: estimarla por petición con su
  prompt (`estimate_prompt_tokens`), los t/s de prefill y generación medidos
  de ese modelo y su reserva de salida.

## Transporte

Barras que se mueven a ~2 Hz con el polling actual saldrían caras (el
dashboard ya lanza muchas consultas periódicas). Se propone un snapshot "live"
único, enviado por el WebSocket que ya existe (`useWebSocket`): modelos (con
`context`, peticiones y cola), GPUs y CPU. Lo histórico sigue por REST.

## Fases

1. **Backend: snapshot live.** Publicar el progreso que ya existe, añadir
   `first_token_at`, el lector de `/slots` (con caché corta y tolerante a
   fallos), el objeto `context` normalizado, la cola con estimación por
   petición y el canal WebSocket. Con tests, incluido un test de invariante:
   los niveles de `context` deben coincidir con lo que admite
   `plan_profile_runtime`.
2. **Frontend.** Partir `pages/Dashboard.tsx` (1307 líneas) en
   `components/dashboard/`: card de modelo (con `BackendIcons`), barra de
   contexto, peticiones en curso, cola, widget de GPU/CPU, últimas peticiones
   y estadísticas. Quitar agentes y descargas.
3. **Verificación en vivo** con Playwright (ver
   `reference_playwright_headless_diagnostics`) y tráfico real: varias
   peticiones simultáneas a un modelo con escalera de 1→2 GPU, una petición
   que fuerce el escalado, una sin streaming y una petición a Ollama.
4. Bump MINOR, y entrada en el changelog solo si cambia algo de la API pública
   (el snapshot es API interna `/ocabra/*`, así que probablemente no haga
   falta).

## Decisiones del usuario (2026-11-09, segunda ronda)

- **Visibilidad:** los usuarios normales ven el dashboard completo. Solo los
  admin ven la **IP del cliente y el user agent** de las peticiones. El
  backend las omite para los demás; no basta con ocultarlas en la UI.
- **Widget de CPU:** solo lo que corre en la CPU local. Los peers de federación
  van en una sección propia, básica en esta ronda (online, habilitado, GPUs y
  modelos: es todo lo que sabemos del peer remoto).
- **Descargas:** indicador en la cabecera mientras haya alguna en curso.
- **Lo que se mantiene del dashboard actual** (inventario del 2026-11-09):
  - Card de modelo: iconos de capacidades, política de carga, VRAM en uso
    frente a la reservada con aviso de OOM, cuenta atrás de descarga por
    inactividad, hora y tiempo de carga, fase de carga, aviso "al límite"
    (diferenciando slots llenos y límite de cómputo), aviso de petición
    atascada, workers de whisper, botón de descargar.
  - GPU: temperatura, VRAM bloqueada, procesos (también los que no son de
    oCabra).
  - Host/CPU: todo (modelo y núcleos, CPU y potencia, RAM y swap, load
    average, sparklines de CPU, potencia y RAM).
  - Últimas peticiones: mensaje de error, IP y user agent (solo admin) y botón
    de cancelar (el backend ya limita a las propias para los no admin).
  - KPIs: revisados para que sean relevantes en vivo (ver abajo).
- **No mencionados, aplico por defecto:** los servicios de generación
  (ComfyUI, A1111, ACE-Step, TRELLIS.2) y el aviso de runtimes externos se
  mantienen como cards en la lista de modelos. Los permisos de descarga
  (unload) se quedan como hoy: el endpoint ya pide solo rol `user`.

### KPIs de cabecera revisados

Ahora (en vivo): peticiones en curso · en cola · t/s agregados de prefill y de
generación · potencia del servidor.
Última hora frente a 24 h: peticiones, tokens, porcentaje de errores
(429, 503 y 5xx), energía.
Se retiran: número de GPUs, modelos cargados, VRAM total y servicios (ya
salen en sus widgets), y el total histórico de tokens (pasa a `/stats`).

## Deudas y riesgos conocidos

- La observabilidad de KV upstream (Fase 4 del plan de KV elástica) sigue
  pendiente. Si `return_progress` no funciona en el fork, el "usado" por
  petición será heurístico.
- La reserva de admisión incluye un colchón de unos 930 MB respecto a
  `resolved` (ver `project_preset_fit_guard`). La barra muestra tokens, no
  VRAM, así que esto no le afecta, pero no hay que mezclar las dos cifras en
  la UI.

## Estado de implementación (2026-11-09)

### Fase 1 — backend · hecha

- `return_progress` en los streams llama.cpp (`WorkerPool._forward_stream_unfair`):
  se pide, se lee y se descarta antes de la lógica del "primer chunk", así
  que el gate sigue viendo el stream de antes. Verificado contra el binario de
  producción (`b1-743e3ee`): sin `return_progress`, llama-server solo manda
  cabeceras hasta el primer token; con él, chunks `prompt_progress`
  (`total`, `cache`, `processed`, `time_ms`).
- Contextvar `current_inflight_request_id` (`core/request_telemetry.py`),
  fijada por el middleware antes de `call_next`: llega a WorkerPool y al
  gate (test con BaseHTTPMiddleware + StreamingResponse).
- `ActiveRequest`: `request_kind`, `streaming`, `max_output_tokens`,
  `first_token_at` y el progreso de prefill.
- `FairSlotGate.requests_snapshot()`: leases y cola por peticion, con el
  orden de turno simulado sin tocar el estado.
- `stats/live_activity.py`: fase, t/s por fase, estimaciones (regresion del
  DurationEstimator + simulacion de liberacion de slots) y objeto `context`
  (niveles de `runtime_capacity_cache`, el mismo que usa la admision).
- `/ocabra/models/activity` ampliado (compatible) y empujado por
  `/ocabra/ws` con `{"subscribe": "models_activity"}`.
- Permisos de lectura abiertos a `user`: `/stats/recent` (IP/UA solo admin),
  `/stats/requests`, `/stats/tokens`, `GET /services`, `GET /downloads`.
- Tests: 1956 sin Postgres, 1971 con Postgres.

### Fase 2 — frontend · hecha

- `components/dashboard/`: KpiStrip, ModelLiveCard, ContextBar,
  RequestProgress, QueueSection, DeviceWidget (GPU y CPU), FederationSection,
  RecentRequestsTable, BackendIcon (simple-icons + lucide), useLiveActivity.
- Indicador de descargas en la cabecera (`layout/DownloadsIndicator.tsx`).
- Retirados: GpuCard, LoadedModelList, ConcentricGauge, MemoryBars,
  MiniTrends (sin uso tras el cambio).
- Revisado visualmente con Playwright: datos reales de produccion (GET
  reenviados con clave de solo lectura) + actividad simulada, que el backend
  desplegado aun no publica. Script en `~/ocabra-dashcheck/shot.cjs`.
- 70 tests de frontend en verde; `tsc -b` limpio.

### Fase 3 — pendiente

Desplegar y repetir la verificacion con trafico real: prefill largo en
llama.cpp (barra de progreso real), escalado 1→2 GPU de qwen3.8:27b (la marca
de 96k y el paso de rayado a utilizable), una peticion sin streaming, una a
Ollama y cola con dos usuarios.

### Deudas · cerradas el 2026-11-09

- **vLLM sin ocupación del pool** → `usagePct` lee `vllm:kv_cache_usage_perc`
  (o `gpu_cache_usage_perc` en el motor v0) de `/metrics` del worker, con
  caché de 2 s. Nombre comprobado en el vLLM 0.32 instalado.
- **VRAM repartida a partes iguales entre GPUs** → `vramByGpu`: NVML por
  proceso, cruzado con el PID del worker y sus descendientes. El reparto real
  de nemotron era 9,6 GB en la 3060 y 21,4 GB en la 3090, no 15,4 + 15,4. El
  reparto a partes iguales solo queda para Ollama, que comparte proceso entre
  modelos, y la UI lo dice.
- **La cola ignoraba el KV** → simulación por eventos con slots y KV libre,
  FIFO por usuario y la cabeza que no cabe deja pasar a otras, como hace el
  gate. Devuelve `predictedStartOrder`; la UI ordena por él y explica cuándo
  una petición adelanta o la adelantan. Si una petición no cabe ni con el pool
  vacío, no se le da estimación.
- **Prefill estimado fuera de llama.cpp** → al ritmo de prefill medido del
  modelo: mediana de las últimas 6 h, consultada contra la BD real (nemotron
  1219 t/s, bonsai 1066 t/s, gemma4:26b 715 t/s). Si no hay medida, se usa la
  regresión del estimador. No pasa del 99 % hasta el primer token real y se
  marca como estimado.
- **Indicador de descargas duplicado** → la píldora de la cabecera abre y
  cierra el panel existente (`DownloadQueue`, que es quien hace polling, SSE y
  cancelación). El panel ya no deja una barra plegada flotando, y la píldora
  avisa también de las descargas fallidas en la última hora.
- **Componentes muertos** → borrados `PowerBlock`, `PowerGauge`, `VramBar` y
  `MetricGauge`.
- **Chip de contexto distinto de la barra** → la card usa el contexto cargado
  (`activeMaxTokens`) antes que el configurado en el modelo base.

### Fase 3 — verificación en producción · hecha (v0.61.1)

Script: `~/ocabra-dashcheck/live_check.py`. Lanza peticiones reales en
streaming a qwen3.8:27b y consulta la actividad cada segundo.

- El cliente recibió 0 chunks `prompt_progress` (de 49 por petición) y todas
  las respuestas fueron correctas.
- Prefill real: 0 → 49.558 tokens a unos 1.100 t/s. El restante bajó de forma
  coherente (44 s → 0) y la petición pasó a `generating` con el primer token.
- `vramByGpu` real: qwen3.8:27b ocupaba 22.802 MB en la GPU 1. Antes de la
  prueba, gemma4:26b 22.378 MB y whisper 2.184 MB en la GPU 0.
- La 0.61.0 tenía dos fallos que solo salieron con tráfico real; los dos están
  arreglados en la 0.61.1:
  - **Petición en prefill sin haber empezado.** llama.cpp procesa los prompts
    de uno en uno, así que las peticiones admitidas que esperaban turno
    dentro del llama-server salían como "prefill 99 %" durante 100–200 s.
    Ahora tienen su propio stage, `backend_wait`: se detecta porque, al
    pedir progreso, llama.cpp manda un 0 % en cuanto empieza cada prompt.
  - **Restante a 0 en mitad de un prefill largo.** Se re-estima con el total
    real de tokens del prompt y nunca baja del prefill que falta.
- En la 0.61.2 (con commit, sin desplegar), el restante tampoco cuenta hacia
  atrás durante `loading` y `waiting_for_gpu`.
- El WebSocket se probó con tests (suscripción, desuscripción, caché
  compartida). En producción, las conexiones del navegador se aceptan. No se
  pudo probar desde fuera sin una sesión real: un token generado en otro
  proceso no vale, porque el secreto JWT vive en la BD.

### Hallazgo fuera del alcance del dashboard

**La admisión infravalora los prompts largos poco "naturales".** La reserva
de KV salió de 29.416 tokens para un prompt real de 49.558, y de 47.685 para
uno de 70.218. Sale del recuento aproximado de tokens, y es la "tokenización
exacta pendiente" del plan de KV elástica. No se ha tocado porque cambia la
admisión.

### Lo que sigue pendiente

- **Desplegar la v0.61.2** cuando termine el A/B de llama.cpp de otra sesión
  (pidió no reiniciar el api durante unos 90 min desde las 23:30 CET del
  2026-11-09).
- **Límite que no se puede cerrar desde oCabra:** Ollama y vLLM no publican el
  avance del prefill por petición. Para ellos se muestra una estimación con un
  ritmo medido.
