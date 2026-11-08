# Plan: un modelo, varios backends · modelo lógico + despliegues

**Fecha de apertura:** 2026-11-07
**Estado:** ✅ Fases 0–6 hechas y deuda cerrada (v0.50.0); queda deuda mínima listada en la Fase 4/5
**Relacionado:** [`backend-distribution-plan.md`](backend-distribution-plan.md) (backend en el id
canónico, migración entre backends), [`llama-cpp-elastic-kv-plan.md`](llama-cpp-elastic-kv-plan.md)
(convención `*-ollama` que este plan elimina), `docs/CONTRACTS.md` §ids canónicos

---

## Decisiones de partida (usuario, 2026-11-07)

1. **Ollama se queda como backend**, pero secundario. El objetivo es servirlo todo con
   `llamacpp_mks`; Ollama queda como opción de respaldo y, sobre todo, como **fuente de
   distribución** de modelos (catálogo + `pull`).
2. **Ids limpios y coherentes.** Se renombran los ids canónicos actuales a un esquema único
   (`<backend>/<model_key>`), con migración de todas las tablas que los referencian. No se
   acepta la deuda de "dejar los ids viejos y poner alias encima".
3. Estrategia de migración: a criterio del implementador (ver Fase 1).

## Problema

Hoy el backend es parte de la **identidad** del modelo y no se puede cambiar:

- `model_configs` = una fila por (backend, pesos), id `backend/nombre`. El mismo GGUF está
  registrado **dos veces** (12 pares `ollama/x` + `llamacpp_mks/x`, el segundo apuntando al blob
  `sha256-…` de Ollama) con nombres internos distintos (`ollama/gemma4:26b` vs
  `llamacpp_mks/gemma4-26b`).
- El nombre público es un perfil con un `base_model_id` fijo. Para "cambiar de backend" se
  reapunta el perfil a mano y el otro se renombra con sufijo `-ollama`. Es convención manual
  (no existe en código) y ya es incoherente (`gemma4:26b-ollama` activo,
  `gemma4:26b-llamacpp` alias, routers mezclando variantes).
- `PATCH /models/{id}` descarta `backend_type` en silencio (`api/internal/models.py:905`); no
  hay endpoint de mover/renombrar.
- La duplicación se **genera sola**: `sync_ollama_inventory` (`core/model_manager.py:3749`)
  autoregistra cada tag de Ollama como `ollama/<tag>`, y el scanner (`registry/local_scanner.py:411`)
  ofrece los blobs de Ollama como candidatos `llama_cpp`.
- El `model_path` de llama.cpp apunta a `blobs/sha256-…` por ruta absoluta: un `ollama pull`
  que actualice el modelo, o una poda, lo deja colgando.
- La ACL (`group_models.model_id`, texto libre) mezcla `profile_id` e id canónico.
- La búsqueda/descarga solo sabe "GGUF→llama_cpp, safetensors→vllm"; las variantes GGUF tienen
  `backend_type="llama_cpp"` hardcodeado (`registry/huggingface.py:784`), así que `llamacpp_mks`
  —el de producción— nunca es el defecto; solo vllm tiene chequeo real de compatibilidad.

## Matriz de compatibilidad (verificada 2026-11-07)

Instalados hoy: acestep, bitnet, diffusers, flashvsr, llama_cpp, llamacpp_mks, piper, upscaler,
vllm, whisper, whisper_cpp, vibeasr + ollama (built-in). Con código pero no instalados: sglang,
tensorrt_llm, tts, chatterbox, voxtral, mage.

| Artefacto | Backends | Tareas |
|---|---|---|
| GGUF (LLM/embeddings, + mmproj, + draft) | `llamacpp_mks`, `llama_cpp`, `ollama` (si el artefacto viene de Ollama o se importa con Modelfile) | chat, completion, tools, visión, embeddings, reasoning |
| GGUF ternario/1-bit (i2_s, q1_0, bonsai) | `bitnet` (llama.cpp upstream quizá cargue algunos — sin verificar) | chat, completion |
| Repo HF safetensors (LLM/encoder) | `vllm`, `sglang`; `tensorrt_llm` previa compilación del engine | chat, embeddings, rerank, score, classify, pooling |
| Árbol diffusers | `diffusers`; `mage` si `_class_name == MageFlowPipeline` | imagen, edición |
| CTranslate2 / `.nemo` | `whisper` | STT (+ diarización) |
| ggml `.bin` | `whisper_cpp` (vive en `backends_dir`, no en `models_dir`) | STT CPU |
| Par GGUF VAE+LM | `vibeasr` | STT |
| HF dir TTS (Qwen3-TTS/Kokoro/Bark / Chatterbox / Voxtral) | `tts` / `chatterbox` / `voxtral` (familias disjuntas) | TTS |
| `.onnx` + `.onnx.json` | `piper` | TTS CPU |
| `.pth`/safetensors ESRGAN · safetensors FlashVSR | `upscaler` · `flashvsr` | upscaling de vídeo |
| checkpoints ACE-Step | `acestep` | música |

Conclusión: **compartir el mismo fichero** solo ocurre de verdad en GGUF (3 backends) y en HF
safetensors LLM (vllm/sglang). En STT/TTS/imagen, "el mismo modelo en otro backend" es **otro
artefacto** (whisper large-v3-turbo CT2 vs ggml q8). Por eso el artefacto va en el despliegue,
no en el modelo.

## Diseño

```
Model  (model_key = nombre público, p.ej. "gemma4:26b")
  │  display_name, category/tarea, default_backend, load_policy, pin_permanent, metadata upstream
  │  ACL, routers y perfiles apuntan aquí
  ├─ Deployment  llamacpp_mks/gemma4:26b   ← default
  │     artifact: {kind: gguf, source: ollama, ref: "gemma4:26b"}   (blob resuelto al cargar)
  │     extra_config: {llama_cpp: {...}}, preferred_gpu
  └─ Deployment  ollama/gemma4:26b
        artifact: {kind: ollama_tag, ref: "gemma4:26b"}
        extra_config: {ollama: {...}}
```

### Entidades

- **`models` (tabla nueva).** PK `model_key` (el nombre público; único espacio de nombres
  compartido con `model_profiles.profile_id` y aliases). Campos: `display_name`, `category`,
  `default_backend`, `load_policy`, `pin_permanent`, `registry_metadata` (sale de `extra_config`).
- **Despliegue = `model_configs` (se conserva la tabla y la clase `ModelConfig`**, para no
  renombrar cientos de usos sin valor; se documenta su nuevo significado). Cambios:
  - `model_key` FK → `models` (ON UPDATE CASCADE), único `(model_key, backend_type)`.
  - `model_id` pasa a ser **siempre** `<backend_type>/<model_key>`. Derivado, no editable.
  - `artifact` JSONB nuevo: descriptor del fichero (`kind`: `gguf`|`hf_repo`|`ollama_tag`|
    `ctranslate2`|`ggml`|`onnx_voice`|`diffusers_tree`|…; `source`: `hf`|`ollama`|`local`;
    `ref`/`repo`/`file`/`path`; `mmproj`/`draft` como sub-artefactos). Reemplaza a
    `extra_config.*.model_path`, `mmproj_model_id`, `speculative.draft_model_id` y al
    `::artifact` codificado en el id.
  - `load_policy`/`pin_permanent` se mueven a `models` (se aplican al despliegue por defecto;
    los no-default son `on_demand`). `preferred_gpu` se queda en el despliegue.
- **Perfiles** (`model_profiles`): `base_model_id` → `base_model_key` (FK a `models`) +
  `backend` opcional (pin: "este perfil siempre en ollama"). El perfil por defecto de un modelo
  tiene `profile_id == model_key`. Los perfiles variante (`*-diarized`) siguen siendo perfiles.
  `load_overrides` ya está namespaced por backend (`{"whisper": {...}}`): al resolver se aplica
  solo la sección del backend elegido.
- **Presets** (`profile_runtime_presets`) son semánticos (contexto, slots, kv dtype, placement) y
  se traducen por backend vía `use_case` (`core/runtime_coordinator.py:446`), así que se
  **comparten** entre despliegues. Lo que es por backend es la capacidad validada:
  `runtime_policy.validated_vram_mb` pasa a `{backend: {preset_key: mb}}`.

### Resolución de nombres

`nombre → alias oculto? → perfil (o modelo) → despliegue → worker`

1. Si el nombre es un id de despliegue (`backend/model_key`) → ese despliegue, con el perfil por
   defecto del modelo. Son las "etiquetas invisibles": aceptadas siempre, **no** listadas en
   `/v1/models` ni `/api/tags`. No necesitan filas de alias.
2. Alias oculto (`model_profile_aliases`) → como hoy.
3. `profile_id`/`model_key` → perfil; router → `RouterResolver` (targets = perfiles, modelos o ids
   de despliegue).
4. Backend = `profile.backend` si existe, si no `model.default_backend`. Si ese despliegue no está
   (backend desinstalado/deshabilitado) → **error explícito**, sin fallback silencioso a otro
   backend (el fallback es trabajo de los routers).

`worker_key`, `model:state:*` en Redis y `request_stats.model_id` siguen siendo por despliegue
(es lo que realmente corre en la GPU). Stats agregan por modelo vía join.

### ACL y routers

- `group_models.model_id` → semánticamente `model_key` o `profile_id` de router. Conceder un
  modelo concede todos sus despliegues y perfiles. Se elimina el doble chequeo
  `can_access_model(profile_id, base_model_id)` por uno solo sobre el `model_key`.
- `routing_targets` acepta `model_key`, `profile_id` o id de despliegue (para fijar backend, p. ej.
  la variante CPU de STT).

### Ollama como fuente

- `sync_ollama_inventory`: en vez de crear `ollama/<tag>` como modelo nuevo, **adjunta** el
  despliegue `ollama` al modelo cuyo artefacto ya es `{source: ollama, ref: <tag>}`. Si no existe,
  crea el modelo con dos despliegues (`llamacpp_mks` default + `ollama`) según la matriz.
- llama.cpp resuelve `artifact{source: ollama, ref}` → manifest → blob de la capa
  `vnd.ollama.image.model` (+ `projector` para mmproj) **en cada carga**. Se acaban las rutas
  `sha256-…` fijas.

### Motor de compatibilidad (`registry/compatibility.py`, nuevo)

Una sola función `compatible_backends(artifact_descriptor) -> list[{backend, status, installed,
reason}]` (`status`: `compatible`|`probable`|`incompatible`), con las reglas de la matriz:
formato (GGUF, ternario por nombre/cabecera, safetensors + `architectures`, `model_index.json`,
CT2, ggml, onnx), arquitectura (`_NATIVE_VLLM_ARCHITECTURES`, recipes), cuantización y estado de
instalación del backend. La consumen: tarjetas de búsqueda, variantes, scanner local, registro
local, descarga y el editor de modelo. Sustituye a `_infer_backend` + `_backend_options` y al
hardcode de `:784`. El probe real de vLLM sigue siendo opcional y bajo demanda.

## Migración de los datos actuales (Fase 1)

Mapeo propuesto (se genera con un script **dry-run que imprime la tabla para revisión** antes de
aplicar; backup `pg_dump` antes de `alembic upgrade`):

| Modelo (`model_key`) | Despliegues → default | Notas |
|---|---|---|
| gemma4:12b, gemma4:26b, mistral:7b, nemotron-3-nano:30b, qwen3:8b, qwen3:32b, qwen3-coder:30b, qwen3-embedding:{0.6b,4b,8b}, qwen3.8:27b, qwen3.8:27b-abliterated (12) | `llamacpp_mks` ★ + `ollama` | artifact llama.cpp pasa de ruta blob a `{source: ollama, ref: <tag>}` |
| nemotron3:33b | `ollama` ★ | solo Ollama hoy; se ofrece `llamacpp_mks` desde la UI |
| ravenx, ravenx-chaos-agent | `llamacpp_mks` ★ | artifact `{source: hf, repo, file}`; `model_key` = perfil actual |
| whisper-large-v3-turbo | `whisper` ★ + `whisper_cpp` (CPU q8) | `recomendado-stt` apunta al despliegue `whisper_cpp/…` explícitamente |
| resto (piper, upscaler, diffusers, acestep, flashvsr, bitnet, vllm reranker, whisper restantes, vibeasr) | 1 despliegue | `model_key` = `profile_id` por defecto actual |

- **Perfiles `*-ollama`** desaparecen. Se crean como **alias ocultos** apuntando al perfil del
  modelo + backend `ollama` (no se rompen clientes que los usen). Sus escaleras de presets
  (32k/64k/128k/max) se descartan: el modelo conserva los presets del perfil público actual
  (llama.cpp), que se aplican también al despliegue Ollama.
- **Routers** que apuntaban a `*-ollama` (`recomendado-ha-chatbots`: `qwen3.8:27b-abliterated-ollama`,
  `gemma4:12b-ollama`) → **modelo genérico** (`qwen3.8:27b-abliterated`, `gemma4:12b`), que sigue su
  backend por defecto (decisión del usuario 2026-11-07).
- **whisper_cpp**: dos perfiles para el mismo despliegue (`whisper-cpp-large-v3-turbo-cpu` y
  `whisper.cpp-large-v3-turbo-q8-0-cpu`) → se queda uno como alias.
- ~~**`whisper/microsoft/VibeVoice-ASR-BitNet`**~~: **borrado 2026-11-07** (solo BD, sin ficheros).
- Tablas a reescribir con el mapeo viejo→nuevo: `model_configs.model_id`,
  `model_profiles.base_model_id`, `model_profiles.routing_targets`, `model_profile_aliases`,
  `agents.base_model_id`, `group_models.model_id`, `request_stats.model_id` (138k filas, solo los
  ids que existen hoy; los históricos huérfanos como `ollama/qwen3.6:latest` se dejan),
  `model_load_stats.model_id`, `eviction_schedules.model_id`, `trtllm_compile_jobs.source_model`.
  Redis `model:state:*`: se purga (hay claves huérfanas `llama_cpp/…` de antes de la migración a
  `llamacpp_mks`); se regenera solo.
- `extra_config`: se normaliza a **snake_case** (hoy casi todas las claves de llama.cpp están
  duplicadas camelCase+snake), `model_path`/`mmproj`/`draft` salen a `artifact`,
  `registry_metadata` sale a `models`. Se añade validación Pydantic por backend al guardar.
- Migración con `downgrade()` real (mapeo inverso guardado en una tabla temporal
  `_model_id_migration_0034`), para poder volver atrás.


### Dry-run (2026-11-07) — `backend/scripts/model_unification_plan.py`

Solo lectura (`SET TRANSACTION READ ONLY`). Ejecutar con
`docker compose exec -T api python - [--json] < backend/scripts/model_unification_plan.py`.
Resultado sobre producción:

- **36 modelos**: 12 GGUF con `llamacpp_mks` ★ + `ollama`; `whisper-large-v3-turbo` con `whisper` ★ +
  `whisper_cpp`; el resto, un despliegue. **Ningún nombre público cambia.**
- Se agrupa por **digest del blob**, no por tag: `qwen3.8:27b-abliterated` estaba publicado en Ollama
  con dos tags (`hf.co/huihui-ai/…:UD-Q5_K_XL` y `qwen3.8:27b-abliterated`) sobre el mismo fichero.
- 13 perfiles `-ollama` y 2 de whisper.cpp → alias ocultos con backend fijado. Se descartan las
  escaleras de presets de 6 de ellos (32k/64k/128k/max de Ollama).
- Routers (decisión del usuario 2026-11-07): `recomendado-ha-chatbots` pasa a los modelos **genéricos** (`qwen3.8:27b-abliterated`,
  `gemma4:12b`), que siguen su backend por defecto; `recomendado-stt` conserva el despliegue CPU
  `whisper_cpp/whisper-large-v3-turbo` como fallback tras los de GPU. Regla: alias → modelo genérico, salvo que el modelo ya sea destino del router.
- ACL: todo pasa a `model_key`; **sin ampliación real de acceso** (los únicos grupos con variantes
  `-diarized` ya tenían también el modelo base).
- `request_stats.model_id` está **mezclado** (la mayoría guarda el nombre público, algunos el id
  canónico, otros el worker key `id::hash`). Propuesta: normalizar a `model_key` (`backend_type` ya
  dice el backend) y guardar lo que pidió el cliente en una columna nueva `requested_model`.
  ~10,8k filas de 185 nombres de modelos ya borrados se dejan como están.
- **Bug encontrado:** `ModelConfigModal` escribe `ctxSize` (camelCase) pero el backend da prioridad a
  `ctx_size`; con las dos claves presentes, la edición desde la UI se ignora en silencio.
  `llamacpp_mks/qwen3.8-27b-abliterated`: UI 229376, efectivo 40960. La normalización debe resolverlo
  y el frontend pasar a escribir snake_case.

- **Decisiones del usuario (2026-11-07) sobre el dry-run:**
  1. Contexto base de `qwen3.8:27b-abliterated` → **40960**. Los dos valores eran los dos presets
     (`normal`: 40960 solo en la 3090, 4 slots con KV unificado, 23,8 GB validados; `extended`:
     229376 en las dos GPUs, 32,9 GB). El base solo se usa al cargar sin preset. Regla general de la
     normalización: **en un choque camel/snake gana snake_case**, que es lo que leen hoy todos los
     lectores (`_get_option`, `resolve_llama_cpp_option`, `_get_whisper_option`).
  2. `load_policy` del modelo = **la más fuerte** de sus despliegues (pin > warm > on_demand):
     `qwen3-embedding:8b` queda `warm`.
  3. `request_stats.model_id` → `model_key`, más columna nueva `requested_model` con lo que pidió el
     cliente.
- **Requisito de despliegue conjunto de la normalización** (si no, la diarización se apaga sin
  avisar): los perfiles `-diarized` activan la diarización con `whisper.diarizationEnabled` en
  `load_overrides`; con la base ya en `whisper.diarization_enabled: false` el lector (snake primero)
  vería `false`. En el mismo despliegue que la migración hay que:
  - normalizar también `model_profiles.load_overrides` (ya en el plan);
  - cambiar a snake_case `build_diarized_extra_config` (`core/model_manager_helpers.py:32`);
  - cambiar el `ModelConfigModal` a leer/escribir snake_case (lee `whisper.diarizationEnabled`
    en `:597`, escribe `ctxSize` etc.), y validar `extra_config` con Pydantic al guardar.


### Fase 1 — ✅ desplegada 2026-11-07 17:07 (v0.46.0)

Despliegue: `pg_dump -Fc` previo en `/home/makespace/ocabra-backups/2026-11-07_pre-0034.dump`;
comparación antes/después repetida sobre clones de ese dump (48/48 despliegues coinciden salvo lo previsto). Nota: el dump (17:01) recogía `nemotron-3-nano:30b` a 524288, que otra sesión revirtió a 65536 a las 17:04, antes de la migración (17:07); producción conserva 65536. Resultado de la comparación: solo
las 18 diferencias previstas (nombre interno de 12 llama.cpp, `load_policy` de 3, snake_case de 3
whisper). Verificado en vivo como usuario real: `/v1/models` (60 entradas), chat `gemma4:26b`,
`recomendado-ha-chatbots`, `recomendado-embedding` (carga en frío de `qwen3-embedding:4b` vía
manifest de Ollama), id antiguo `llamacpp_mks/gemma4-26b` (alias), variante
`qwen3-embedding:0.6b-ollama`; `request_stats` con `model_id`=modelo y `requested_model`
rellenado en las 138k filas. Pendiente de UI: `ModelConfigModal.tsx:597` aún lee
`whisper.diarizationEnabled` (camelCase) — hoy todos los base valen `false`, así que se ve bien
por casualidad; arreglar en Fase 4.

Migración `0034_models_and_deployments` (esquema + datos en un paso; el entrypoint la aplica al
arrancar). La lógica vive en `ocabra/core/model_unification.py` (planificador puro +
`apply_plan`), los artefactos en `ocabra/core/artifacts.py`, la normalización de claves en
`ocabra/core/config_keys.py`.

Cambios respecto al diseño original, todos para **no cambiar el comportamiento en producción**:

- **Los perfiles `-ollama` NO pasan a alias**: se quedan como variantes fijadas a su despliegue
  de Ollama, con sus presets. Motivo: 176 peticiones en 30 días desde hasta 6 claves de API, con
  embeddings entre ellas (sus vectores deben seguir saliendo del mismo backend), y un alias no puede
  fijar backend hasta la Fase 2. Pasan a alias en la Fase 2.
- El fallback CPU de `recomendado-stt` también se queda como variante fijada (los routers solo
  aceptan perfiles como destino hasta la Fase 2).
- **Los ids antiguos siguen resolviendo**: un alias oculto por cada id de despliegue renombrado,
  apuntando al perfil de ese mismo despliegue (29 en producción).
- `backend_model_id` (lo que recibe el backend) = `artifact.ref`, que conserva el valor de antes en
  todos los despliegues salvo los GGUF de Ollama en llama.cpp (pasa a ser el tag: cosmético, el
  fichero se encuentra por la ruta resuelta).
- `model_path`/`mmproj`/`draft` **no se persisten**: se derivan del artefacto al construir el
  estado y en cada carga (sigue a un `ollama pull`). Una edición de esas rutas desde la UI
  actualiza el artefacto (`apply_path_edits`).
- `load_policy` se queda en el despliegue (el default lleva la del modelo, el resto `on_demand`):
  `ollama/qwen3.8:27b` y `ollama/qwen3-embedding:8b` pasan de `warm` a `on_demand`.
- ACL compatible con el chequeo actual: ids → `model_key`, variantes con entrada propia.
- Escrituras de `extra_config` y `load_overrides` normalizadas a snake_case (en un PATCH gana la
  grafía camelCase entrante, que es la edición). Diarización: `build_diarized_extra_config` escribe
  snake_case. Los lectores con doble grafía se mantienen.
- Stats: `request_stats.model_id` = `model_key`, `requested_model` = lo que pidió el cliente;
  `model_load_stats` por despliegue (se pliega al modelo en el agregador). Al arrancar se purgan
  las claves `model:state:*` de ids que ya no existen.

**Verificación sobre un clon de producción** (`ocabra_migold` → `ocabra_migtest`): lo que recibe
cada backend antes (código y datos actuales) y después (código nuevo sobre la base migrada) coincide
en los 48 despliegues — referencia, fichero resuelto y todas las opciones de llama.cpp — salvo los
cambios previstos de arriba. Script en `/home/makespace/ocabra-migcheck/`.

### Fase 2 + Fase 3 (API) + Fase 5 (backend) — implementadas 2026-11-07 (v0.47.0)

Sin migración: reutiliza `model_profiles.backend` y `model_profile_aliases.backend` de la 0034.

- **Invariante: cada despliegue tiene al menos un perfil.** Los perfiles con `backend` fijado son
  variantes ocultas: resuelven por nombre y por `<backend>/<model_key>`, pero no se listan en
  `/v1/models` ni `/api/tags`. `ensure_default_profiles` crea una variante oculta
  (`<model_key>-<backend>`) para todo despliegue no-default sin perfil. Así la resolución de
  nombres no necesita cambios: no se construyen perfiles "con backend forzado" al vuelo.
- **Cambiar el backend por defecto** (`core/model_deployments.set_default_backend`, expuesto en
  `PATCH /ocabra/logical-models/{key}`): reapunta los perfiles públicos (`backend IS NULL`) al
  despliegue nuevo, mueve `load_policy`/`pin` al nuevo default (el viejo pasa a `on_demand`),
  aparca/restaura la capacidad validada por backend (`runtime_policy.validated_by_backend`: los
  lectores siguen leyendo los mapas planos, que siempre describen el backend actual) y da una
  variante oculta al antiguo default. El worker viejo se deja a la expulsión por inactividad.
- **Añadir/quitar despliegues** (`POST/DELETE /ocabra/logical-models/{key}/deployments`): el
  artefacto se deriva del existente con `core/compatibility.derive_artifact` (solo backends que
  pueden leer ese mismo fichero); quitar el default de un modelo con más despliegues cambia antes el
  default (los perfiles caen en cascada con su despliegue). `DELETE /ocabra/models/{id}` usa la
  misma lógica.
- **ACL por modelo**: `_expand_model_grants` amplía en la autenticación cada `model_key` concedido a
  sus despliegues, perfiles (ocultos incluidos) y alias; las 15 comprobaciones `can_access_model`
  no cambian. Conceder solo una variante sigue concediendo solo esa variante.
- `resolve_model` (Ollama/legacy) ya no toma el primer `backend_model_id` que coincide: prefiere el
  despliegue default del modelo (`ModelManager.is_default_deployment`).
- Federación con routers: elige el primer destino caliente (local o en un peer), como
  `RouterResolver`, en vez del primero de la lista.
- **Compatibilidad** (`core/compatibility.py`): una sola tabla de reglas. Las tarjetas y variantes
  de HF/Ollama/BitNet llevan `compatible_backends` (`installed` según el instalador; si el
  sugerido no está instalado se cae al primero compatible que sí). Las variantes GGUF sugieren
  `llamacpp_mks` (antes `llama_cpp` fijo).
- **Descarga**: `register_config` acepta `model_key`, `default_backend` y `backends`. Un pull de
  Ollama se sirve por defecto con `llamacpp_mks` (artefacto = tag) y conserva el despliegue
  `ollama` como respaldo.
- Pendiente: Fase 4 (UI de modelos) y la parte de UI de la Fase 5.

### Fase 4 + Fase 5 (UI) — implementadas 2026-11-07 (v0.48.0)

- **Modelos**: vista "Por modelo" (por defecto) junto a la de siempre "Por despliegue". Cada modelo
  muestra sus backends (`BackendIcons`: ★ default, sólido = desplegado, discontinuo = compatible,
  "?" = probable, tachado = no instalado) y una fila por despliegue; acciones: backend por defecto
  (PATCH), añadir backend compatible e instalado (POST), quitar despliegue con confirmación (DELETE).
  Clic en un despliegue abre el `ModelConfigModal` de siempre.
- **`ModelConfigModal` en snake_case**: lee snake con fallback camel y escribe solo snake. Arregla
  dos bugs reales: la edición del tipo de caché KV se perdía (se enviaban `cache_type_k` nuevo y
  `cacheTypeK` viejo, y en un choque el servidor se queda con el camelCase de la edición) y
  `default_reasoning_effort` se mostraba vacío.
- **Explore**: iconos de compatibilidad en tarjetas HF/BitNet/Ollama y variantes; el diálogo de
  descarga pide "Nombre del modelo" (`model_key`), qué backends desplegar y cuál es el default.
  De paso, la búsqueda de Ollama no se mapeaba y los tamaños salían "?".

Deuda de la primera versión, **cerrada en v0.50.0**:
- Editor de modelo (`LogicalModelEditor`): pestaña "General" (nombre visible, backend por defecto,
  añadir backend) y una pestaña por despliegue con su formulario embebido (`ModelConfigForm`,
  extraído de `ModelConfigModal`, que sigue envolviéndolo para la vista "Por despliegue").
- `/logical-models` devuelve `preferredGpu`; se muestra por despliegue y se edita en su pestaña.
- Vaciar un campo envía `null` y el servidor borra la clave (`_deep_merge_dicts`, JSON Merge Patch).
- whisper.cpp: campos `threads` y `diarization_model_id` (la diarización se pide por petición).
- El escáner local sugiere `llamacpp_mks`.

Queda (menor): claves antiguas en la raíz de `extra_config` que el formulario mueve a su sección
no se borran al vaciarlas (solo se omiten); dos diálogos Radix apilados al quitar un despliegue desde
el editor, no probado en navegador real.

## Fases

| Fase | Contenido | Riesgo |
|---|---|---|
| **0** | Fixes independientes encontrados en el análisis (ver abajo). Se pueden hacer ya. | bajo |
| **1** | Esquema (`models`, columnas nuevas), script dry-run de mapeo, migración Alembic + downgrade, normalización de `extra_config`. | **alto** — toca producción; ventana de mantenimiento |
| **2** | Resolución y runtime: resolver nombre→despliegue, coordinator/presets con backend, `validated_vram_mb` por backend, resolución de artefactos Ollama al cargar, `sync_ollama_inventory` que adjunta, ACL por `model_key`. Federación: peers emparejan por `model_key` (hoy por id canónico; `federation_enabled=false` en este nodo). | alto |
| **3** | API interna: `/ocabra/models` devuelve modelos con sus despliegues; CRUD de despliegues, `PATCH default_backend`; motor de compatibilidad expuesto. Revisar consumidores externos de `/ocabra/models` (gateway, LiteLLM sync) antes de cambiar la forma. | medio |
| **4** | Frontend Models: una fila por modelo con iconos de backend (★ default, gris = no instalado), editor con pestaña por despliegue + selector de backend por defecto + añadir/quitar despliegue. | medio |
| **5** | Frontend Explore/descarga: iconos de compatibilidad en tarjetas (HF, Ollama, BitNet) y variantes; diálogo de descarga con qué despliegues crear y cuál es el default; registro local (`StorageSettings`) con el mismo motor. | medio |
| **6** | Docs: `CONTRACTS.md` (ids, resolución), `ROADMAP.md`, este plan; bump MINOR. | bajo |

### Fase 0 — fixes independientes · ✅ cerrada 2026-11-07 (`d222e34`, v0.43.1)

Hechos: 1, 4, 5, 6 y un bug extra (bitnet solo leía `model_path` de la raíz y el registro local
lo escribe en la sección `bitnet`). **Movidos a Fase 2**: 2 y 3, porque esa fase reescribe justo
ese emparejamiento y la federación está desactivada en este nodo (sin impacto hoy).


1. `_refresh_registry_metadata` (`api/internal/downloads.py:412`) re-infiere el backend e ignora
   el elegido por el usuario → metadata escrita a un `model_id` que no existe.
2. Federación con routers: `resolve_federated` (`core/federation.py:716`) toma el primer target
   existente, no la elección de `RouterResolver`.
3. `resolve_model` legacy por `backend_model_id` (`api/openai/_deps.py:172`) toma el primer match;
   ambiguo con dos backends. (Se resuelve del todo en Fase 2.)
4. Descarga escribe `model_path` en la raíz de `extra_config` (`downloads.py:386`) y el registro
   local dentro de `llama_cpp` (`registry.py:235`).
5. ~~`server_config.download_dir = "/tmp/test-downloads"`~~ **Diagnosticado 2026-11-07:** no era
   un directorio temporal sino `test_patch_config_applies_runtime_values` persistiendo en la BD de
   producción (los tests de integración usaban el `DATABASE_URL` real). Arreglado en
   `backend/tests/conftest.py`: los tests van siempre a `<db>_test`, creada y migrada al arrancar.
   Queda limpiar los valores contaminados de `server_config` y las filas `test/*` de
   `model_load_stats` en producción.
6. ~~`whisper/microsoft/VibeVoice-ASR-BitNet`~~ borrado 2026-11-07 (sin ficheros; el bueno es
   `vibeasr/…`).

## Cuestiones abiertas

- ¿`model_key` para modelos que vienen de HF? Propuesta: slug corto editable en el diálogo de
  descarga (por defecto, nombre del repo sin org y en minúsculas, o el tag de Ollama).
- ¿Dos despliegues del mismo backend con distinta cuantización (Q4/Q8)? Propuesta: **no**; una
  cuantización distinta es otro modelo (`qwen3:8b` vs `qwen3:8b-q8`). Mantiene `(model_key,
  backend)` único y el id derivable.
- `llama_cpp` (upstream) frente a `llamacpp_mks`: con el modelo nuevo, tener los dos como
  despliegues de un mismo modelo es justo el A/B que motivó separarlos.

## Avances / deudas / pendiente

- **Hecho (2026-11-07):** análisis de backends, artefactos, flujo de búsqueda/descarga/resolución
  y datos reales de producción; este plan.
- **Hecho (2026-11-07):** Fase 0; tests aislados en `ocabra_test` (`e5234cc`); VibeVoice duplicado borrado;
  `DIFFUSERS_OFFLOAD_MODE=model` en `.env` (sin efecto en los 2 modelos actuales, que fijan `none`).
- **Hecho (2026-11-07):** Fase 1 desplegada y verificada en vivo.
- **Pendiente:** Fases 2–6 (incluye pasar los perfiles `-ollama` a alias y la lectura snake_case de la UI).

## Incidente 2026-11-08 · presets que "no caben" tumban el modelo

Un preset `fixed` de 262144 para `gemma4:26b` (verificado a mano con una petición de 230k) dejó el
perfil sin ningún preset que el planificador diese por bueno (concede ~177k en la 3090: reserva para 4
peticiones simultáneas + el borrador MTP, deliberadamente conservador desde los OOM del 27/10 y 04/11).
`plan_profile_runtime` rechazó todas las peticiones 2 minutos. Arreglo: preset borrado (gemma4:26b
corre sin presets, a 262144 en la 3090) y **guarda en los endpoints de presets** (`_unfit_preset_reason`
en `api/internal/profiles.py`): crear/modificar/borrar un preset que deje al perfil sin ningún preset
habilitado que quepa se deshace con 400. **No ponerle preset a `gemma4:26b`** mientras el planificador
cuente así; un preset `context_mode: max` se adaptaría en vez de rechazar.
