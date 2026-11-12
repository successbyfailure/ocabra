# Nuevo editor de modelos — propuesta de diseño

Estado: **implementado en la v0.74.0** (2026-11-12). Diseñado primero a partir
de lo que el backend acepta y expone, sin mirar el editor viejo; después se
repasó el editor viejo para no perder nada (ver §9). Decisiones del usuario:
página completa, con todas las mejoras y arreglos.

## 1. Qué hay que editar (inventario resumido)

Producción hoy: 57 modelos — 17 llamacpp_mks, 15 ollama, 5 upscaler, 5 agentes,
4 whisper, 4 piper, 2 diffusers y 1 de strata, flashvsr, acestep, vllm y
whisper_cpp. Cargas: 46 on_demand, 4 pin y 2 warm.

| Capa | Dónde vive | Qué contiene |
|---|---|---|
| Modelo lógico | `models` | nombre, backend por defecto, despliegues (uno por backend) |
| Despliegue | `model_configs` | backend, política de carga, pin duro, GPU preferida, `extra_config`, `artifact` (rutas, mmproj, draft) |
| `extra_config` universal | JSONB | `keep_alive_seconds`, `default_reasoning_effort`, `use_case`, `runtime_capabilities.max_parallelism`, `vram_estimate_mb` |
| `extra_config` por backend | JSONB, sección por backend | ~45 claves en llama.cpp, ~30 en vLLM, 8 en sglang, 6 en diffusers, 4 en whisper, etc. |
| Perfiles | `model_profiles` | `load_overrides`, `request_defaults`, `runtime_policy`, `assets` (voice_ref), routers con `routing_targets` |
| Presets de contexto | `profile_runtime_presets` | contexto fijo/máx, paralelismo, dtype KV, colocación, ejecución local/remota, prioridad, por defecto |
| Alias | `model_profile_aliases` | alias → (perfil, preset), oculto |
| Horarios | `eviction_schedules` | días + franja: desalojar y recargar |

Endpoints que permiten **previsualizar sin cargar** (la base de la UX visual):

- `POST /models/{id}/memory-estimate` con un `extra_config` en borrador: pesos, KV, contexto máximo, si cabe, concurrencia máxima.
- `GET /models/{id}/capacity?gpu=&slots=&kv_dtype=&context=`: arquitectura, MB de KV por 1k, tabla slots→contexto máximo, curva VRAM(contexto). Ejemplo real (ornith 35b): pesos 20,7 GB, 21,5 MB/1k tokens, 1 slot→142k, 2→70k, 4→33k.
- `POST /profiles/{pid}/runtime-presets/preview` y `GET …/resolved`: contexto efectivo, límites (modelo, hardware, política), GPUs destino, si cabe, avisos.
- `GET /models/{id}/speculative-candidates`: drafts compatibles por tokenizer.
- `POST /estimate`: tiempo esperado de una petición tipo.
- Con efectos (botón explícito): context-probe y benchmark por preset.

## 2. Principios

1. **Ver el efecto antes de guardar.** Cada control que cambia memoria o contexto
   recalcula una vista previa (con debounce, ~300 ms) usando los endpoints de
   arriba. Nunca se pide "guarda y carga para ver si cabe".
2. **Organizado por intención, no por backend.** Las secciones responden a
   preguntas del usuario: *¿cuándo está cargado?*, *¿dónde corre?*, *¿cuánto
   contexto y cuántos a la vez?*, *¿cómo responde?*. Las claves de cada backend
   se colocan en esas secciones; lo que no encaja va a *Avanzado*.
3. **Herencia visible.** Cada campo muestra de dónde sale su valor: por defecto
   global, del despliegue, del perfil o del preset. Lo sobrescrito se resalta y
   tiene un botón "volver al heredado".
4. **Solo lo que aplica.** Los campos dependen del backend y del estado (p. ej.
   `pin_permanent` solo con pin, `cache_type_v` solo con flash_attn, horarios
   solo con warm/pin). Lo que no aplica no se muestra o se muestra apagado con
   el motivo.
5. **Cambios como diff.** Un panel "cambios pendientes" lista cada cambio, su
   impacto (VRAM, contexto) y si **requiere recargar** o se aplica en caliente.
6. **Mismo lenguaje visual que el dashboard**: GpuBadge con color por GPU,
   ContextBar con la marca de 1 GPU, BackendIcon, colores de estado.

## 3. Estructura de la página

Página completa `/models/:id/edit` (no un modal: hay demasiado contenido y la
vista previa necesita espacio fijo).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [icono backend] ornith-1.5:35b   llamacpp_mks ▾   ● cargado en #1 RTX 3090   │
│ despliegues: [llamacpp_mks ★] [ollama]  + añadir         [Descartar] [Guardar]│
├────────────┬──────────────────────────────────────────┬──────────────────────┤
│ Secciones  │  Sección activa                          │  VISTA PREVIA (fija) │
│            │                                          │                      │
│ ● Resumen  │                                          │  Memoria por GPU     │
│ ● Carga    │                                          │  #0 3060 ▓▓░░░ 9/12  │
│ ● GPUs     │                                          │  #1 3090 ▓▓▓▓▒ 22/24 │
│ ● Contexto │                                          │   pesos·KV·extra     │
│ ● Presets  │                                          │                      │
│ ● Respuesta│                                          │  Contexto            │
│ ● Acelerac.│                                          │  ├──96k─┼────256k┤   │
│ ● Perfiles │                                          │  Concurrencia: 2     │
│ ● Avanzado │                                          │  ✓ Cabe   ⟳ recarga │
│            │                                          │                      │
│ (• = tiene │                                          │  Cambios (3)  ▸      │
│  cambios)  │                                          │                      │
└────────────┴──────────────────────────────────────────┴──────────────────────┘
```

- **Cabecera**: identidad, estado vivo (mismo que la card del dashboard),
  selector de despliegue del modelo lógico (★ = backend por defecto) y acciones.
- **Navegación izquierda** con un punto en las secciones con cambios o avisos.
- **Vista previa derecha, siempre visible**: memoria por GPU (barra apilada
  pesos / KV / overhead / proyector visión / draft, contra el total y lo libre
  ahora), contexto (ContextBar con tope de 1 GPU), concurrencia, veredicto
  (cabe / no cabe / cabe solo desalojando X) y contador de cambios pendientes.
  En móvil pasa a una barra inferior desplegable.

## 4. Secciones

### 4.1 Resumen

Lectura rápida y punto de entrada: capacidades (iconos), artefacto (GGUF, mmproj,
draft, tamaño en disco), rendimiento medido (t/s de prefill y generación, p75 de
salida, últimos 6 h), peticiones recientes y estado de cada preset. Atajos
"Probar en Playground" y "Medir (benchmark)".

### 4.2 Carga — *¿cuándo está en memoria?*

- **Política** como tres tarjetas grandes con un dibujo de línea de tiempo:
  - *Bajo demanda*: se carga con la primera petición, se descarga tras N min sin uso.
  - *Caliente (warm)*: se queda cargado; solo sale por presión o por horario.
  - *Fijado (pin)*: se carga al arrancar y se recarga si cae. Interruptor "pin
    duro" (no se desaloja ni como último recurso) visible solo aquí.
- **Tiempo de inactividad** (solo bajo demanda): deslizador con presets
  (1 min, 5, 15, 1 h, nunca) y la línea de tiempo animada "última petición →
  descarga". Muestra el valor global heredado.
- **Horarios**: rejilla semanal 7×24 en la que se pintan franjas con el ratón
  ("fuera de memoria de 9:00 a 14:00 L–V"). Solo activa con warm/pin; con
  bajo demanda se muestra el motivo (hoy el backend las ignora en silencio).
- **Coste de carga**: tiempo típico medido de arranque en frío, para decidir
  entre bajo demanda y warm.

### 4.3 GPUs — *¿dónde corre?*

- Tarjetas de las GPUs reales (nombre + color, VRAM total/libre, qué tiene
  cargado). Clic para elegir la **GPU preferida**; estrella = GPU principal;
  interruptor para **excluir** una GPU.
- Si se usan varias: **reparto** como una barra que se divide arrastrando el
  separador (`tensor_split`), con "a partes iguales" / "favorecer la
  principal", y el modo de división (capa / fila / tensor) con una frase de
  qué implica.
- **Capas en GPU** (`gpu_layers`): deslizador sobre un dibujo de las N capas
  del modelo, coloreadas GPU vs CPU, con el efecto en VRAM y RAM. MoE:
  `n_cpu_moe` con el mismo dibujo.
- Proyector de visión: en qué GPU.
- vLLM/sglang: tensor/pipeline parallel y fracción de memoria (barra de la GPU
  con la porción reservada al motor).

### 4.4 Contexto y concurrencia — *¿cuánto y cuántos a la vez?*

El corazón visual de la página.

- Tres controles: **contexto** (deslizador logarítmico con marcas 8k…256k y el
  máximo nativo del modelo), **peticiones simultáneas** (1–64) y **precisión
  de la KV** (fp16 / q8 / q4 como botones con su ahorro: "q8 = mitad de
  memoria").
- **Modo de la KV** (llama.cpp): *por slot* vs *pool compartido*
  (`kv_unified`), con el mismo dibujo que el dashboard: N cajas iguales frente
  a un depósito común.
- **Gráfica de capacidad**: eje X contexto, eje Y VRAM, una curva por número
  de slots (de `/capacity`), con líneas horizontales en el total de cada GPU y
  en 1 GPU / 2 GPU. El punto elegido se mueve al tocar los controles; la zona
  que no cabe sale sombreada. Debajo, la tabla slots → contexto máximo.
- Límites de la política: contexto máximo y salida máxima por petición.

### 4.5 Presets de contexto — *la escalera*

- Vista de **escalera** sobre un eje de contexto común (32k · 64k · 128k ·
  máx): cada peldaño es una tarjeta con contexto efectivo, paralelismo, dtype,
  GPUs destino (badges), ✓ cabe / ✗ no cabe con el motivo (límite del modelo,
  del hardware o de la política) y ★ por defecto. Datos de `…/resolved`.
- Editar un peldaño abre su panel con los controles de 4.4, previsualizado con
  `…/preview`; la guarda de "algún preset tiene que caber" se avisa **antes**
  de guardar en vez de devolver un 400.
- Alias colgados de cada peldaño (`modelo:128k` → preset 128k), con copiar.
- Política de selección: "preferir el ya cargado" / "preferir el por defecto",
  volver al por defecto tras inactividad, gracia de bajada y residencia mínima,
  mostradas como una línea de tiempo.
- Botones con efectos y confirmación: **sondear contexto** (avisa de que
  desaloja lo que haya en esas GPUs) y **benchmark**, con el resultado medido
  junto a la estimación.

### 4.6 Respuesta — *¿cómo contesta?*

- **Razonamiento**: esfuerzo por defecto (apagado / bajo / medio / alto) y,
  en llama.cpp, modo, formato y presupuesto. Aviso si la plantilla del modelo
  no lee el esfuerzo (lo sabemos: solo 3 de 14 lo hacen).
- **Valores por defecto de las peticiones** (`request_defaults`) como
  formulario de los parámetros comunes (temperature, top_p, top_k,
  max_tokens, penalizaciones, DRY…) con un recordatorio: "el cliente siempre
  gana". Para transcripción: diarizar sí/no, idioma.
- **Plantilla de chat**: integrada / nombre / Jinja propio, con jinja on/off.
- Voz (TTS): la referencia de voz como reproductor + subir otra.

### 4.7 Aceleración

- **Decodificación especulativa**: tipo (MTP propio / draft / eagle3) y draft
  elegido de una lista de candidatos compatibles, con el coste en VRAM en la
  vista previa y la última velocidad medida.
- flash attention, batch / ubatch, mlock / mmap, cache de prompts en RAM.
- **Visión**: siempre / bajo demanda / apagada, con su VRAM.
- vLLM: prefix caching, chunked prefill, eager, max_num_seqs / batched tokens.
- diffusers: modo de offload como tres opciones con el trade-off VRAM ↔
  velocidad; dtype; cuantización del text encoder.

### 4.8 Perfiles

- Lista de perfiles del modelo como tarjetas (nombre, categoría, por defecto,
  activado, alias). Cada perfil se edita con **las mismas secciones**, pero en
  modo "sobrescritura": todo aparece heredado del despliegue y solo se pinta
  lo que el perfil cambia.
- Routers: lista ordenable (arrastrar) de destinos, con el estado vivo de cada
  uno ("se usará este porque ya está cargado").
- Clonar, renombrar, borrar.

### 4.9 Avanzado

- `extra_config` en crudo (JSON) con resaltado de: claves conocidas, claves
  desconocidas (posible error tipográfico) y claves que se ignoran.
- `extra_args`, variables de entorno, `hf_overrides`.
- Vista **"comando de arranque"**: los flags exactos que recibirá el worker con
  la configuración en borrador.
- Mantenimiento on/off.

## 5. Guardar

- El borrador vive en el cliente; el panel de cambios muestra cada diff con su
  impacto y una etiqueta *en caliente* (política, pin, nombre, request
  defaults, contexto en vivo en mks) o *requiere recarga*.
- **Guardar** y, si el modelo está cargado y algo requiere recarga,
  **Guardar y recargar** (con aviso de peticiones en curso).
- Borrar una clave es explícito (el PATCH hace deep-merge: hay que mandar null).
- Al salir con cambios sin guardar, se pregunta.

## 6. De dónde salen los campos: esquema servido por el backend

Hoy las claves por backend solo existen en el código de cada backend, con
deriva entre schema, backend y worker. Propuesta: un endpoint
`GET /ocabra/backends/{backend}/config-schema` que devuelve, por campo:
clave, tipo, valor por defecto (incluido el global), rango o valores,
sección de la página, etiqueta y ayuda en español, si requiere recarga, si
afecta a la memoria (para lanzar la vista previa) y dependencias ("solo si
flash_attn"). El frontend genera los formularios desde ahí, con componentes
a medida para los campos visuales (GPUs, reparto, contexto, escalera). Un test
compara el esquema con lo que el backend lee realmente, para que no vuelva a
derivar. El mismo esquema valida el PATCH, que hoy acepta cualquier cosa.

## 7. Arreglos previos que el editor destapa

Encontrados en el inventario; conviene arreglarlos antes o a la vez, porque si
no el editor mostraría controles que no hacen nada:

1. `vramEstimateMb` del PATCH se descarta en silencio.
2. `keep_alive_seconds` en la sección `llama_cpp` se ignora en llamacpp_mks (lo lee en `llamacpp_mks`).
3. `split_mode="tensor"`: el backend lo acepta y el schema lo rechaza.
4. `load_overrides` de un perfil sustituyen la sección entera del backend en vez de fusionarse.
5. Horarios por modelo solo actúan sobre warm (con pin y bajo demanda no hacen nada).
6. Claves muertas: llama.cpp `runtime`, `runtime_policy.remote_warm_threshold_seconds`, ollama `num_parallel`, sglang vía `use_case` (escribe en `vllm`), vLLM `swap_space`.
7. bitnet usa `parallel` pero los slots se leen de `parallel_slots`; TensorRT-LLM `context_length` acaba en `--max-num-tokens`.
8. `pin_permanent` protege también modelos que no son pin.
9. Un horario mal formado da 500 (KeyError) en vez de 400.

## 8. Fases propuestas

1. **Esquema y arreglos previos (backend)**: endpoint de esquema por backend
   (empezando por llamacpp_mks, ollama, whisper, piper, upscaler y diffusers, que
   cubren el 90 % de los modelos), validación en el PATCH, arreglos de §7, un
   endpoint de vista previa unificado (memoria por GPU + capacidad + comando de
   arranque a partir de un borrador).
2. **Esqueleto de la página**: cabecera, navegación, vista previa fija, borrador
   + diff + guardar, secciones Carga y Avanzado.
3. **Lo visual**: GPUs, Contexto (gráfica de capacidad), Presets (escalera).
4. **Respuesta, Aceleración y Perfiles** (modo sobrescritura).
5. Resto de backends, retirar el editor viejo, verificación con Playwright
   contra producción.

## 9. Lo que se hizo (v0.74.0)

**Backend.**
- `core/config_schema.py`: esquema de opciones por backend.
- `GET /models/{id}/config-schema` y `POST /models/{id}/preview` (errores,
  claves desconocidas, estimación, reparto por GPU y comando exacto del worker
  en llama.cpp y vLLM).
- El PATCH valida contra el esquema y exige `model_manager`.
- Los nueve arreglos de §7 están hechos. Destaparon tres más:
  - `GET /models/{id}/profiles` y `/speculative-candidates` daban siempre 404,
    tapados por la ruta comodín.
  - El editor viejo gestionaba los grupos con el id del despliegue, cuando
    `group_models` guarda ids de perfil.
  - Los días de los horarios estaban desplazados uno (el botón «L» guardaba
    domingo).

**Frontend.**
- La página es `/models/edit/<backend>/<modelo>`, con un despliegue por
  pestaña. Componentes en `components/model-editor/`.
- El borrador se guarda como JSON Merge Patch exacto: los null solo van en lo
  que se borró, y solo se diffea lo que cambia, así que `spec_type` y las
  claves sin control se conservan.
- La vista previa usa la VRAM en vivo del propio modelo (`vramByGpu`) para no
  contarla como ocupada.
- Presets, alias, política, perfiles, ficheros y grupos se guardan al momento
  (son entidades aparte). Cada sección lo dice.
- Sin `model_manager` la página es de solo lectura.
- Se retiraron `ModelConfigModal`, `LogicalModelEditor`,
  `CapacityPlacementPanel`, `CapacityPanel` y `fieldPrimitives`.
- `ProfileModal` y `RuntimePresetsPanel` siguen vivos para `/profiles`, y
  `ScheduleEditor` para los horarios globales, con su error de días
  arreglado.

**Verificación.**
- Playwright con datos reales de producción (`~/ocabra-dashcheck/editor.cjs`)
  en llama.cpp mks, ollama, vLLM, whisper, difusores, piper y Strata, también
  en solo lectura.
- Tests: backend `tests/core/test_model_editor_backend.py`; frontend
  `components/model-editor/**/__tests__`.

**Deudas.**
- La vista previa supone las GPUs que elegiría el planificador. Con presets,
  la memoria por GPU es la de la configuración base; lo real lo dice cada
  preset (se muestra aparte).
- El reparto por GPU de la estimación solo es exacto en llama.cpp. Los demás
  backends dan un total.
- `runtime_policy.remote_warm_threshold_seconds` sigue sin consumidor (oculto
  en la UI).
- `api.profiles.update` no puede mandar null en `load_overrides` y
  `request_defaults`: vaciarlos manda `{}`.
- Las rutas de modelo y proyector solo se muestran. Cambiarlas sigue siendo
  editar el artefacto en Avanzado.
