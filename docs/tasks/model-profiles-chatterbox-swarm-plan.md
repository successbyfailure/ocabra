# Plan operativo — Model Profiles + Chatterbox TTS

Fecha: 2026-04-06
Estado: activo
Objetivo: ejecutar Fase 6 (Model Profiles) y el soporte inicial de Chatterbox TTS, dejando fuera por ahora la UI de fine-tuning de voz.

## Alcance de esta oleada

Incluye:
- separación interna entre modelos base y perfiles públicos
- resolución por `profile_id` en `/v1/*` y `/api/*`
- CRUD de perfiles en `/ocabra/*`
- grupos de acceso vinculados a perfiles, no a modelos base
- soporte de assets por perfil
- para perfiles TTS, subida de audio de referencia desde la configuración del perfil
- backend `chatterbox` con síntesis normal, streaming y voice cloning zero-shot vía `voice_ref`

Queda fuera:
- motor genérico de fine-tuning
- UI wizard de entrenamiento
- jobs de fine-tuning, datasets y checkpoints
- auto-creación de perfiles tras entrenamiento

## Principios de implementación

1. Los modelos base siguen siendo internos y administrables.
2. Los clientes consumen solo `profile_id`.
3. `profile_id` pasa a ser la superficie pública de `/v1/models` y de la inferencia.
4. `voice_ref` se modela como asset del perfil, no como ruta libre enviada por el cliente.
5. Los perfiles TTS deben poder subir audio desde la UI del perfil y reutilizarlo en runtime.
6. Workers se comparten solo si `load_overrides` es idéntico; si cambia, worker dedicado.

## Decisiones ya cerradas

- No se implementa todavía la UI de fine-tuning de voz.
- Sí se implementa upload de audio de referencia desde configuración del perfil TTS.
- `voice_ref` debe resolverse internamente a una ruta controlada por oCabra.
- El soporte Chatterbox entra como backend first-class, no como hack dentro de `tts_backend.py`.

## Oleadas

### Oleada 1 — Foundation de perfiles

Entrega:
- modelo SQLAlchemy `ModelProfile`
- storage de assets por perfil
- endpoints internos CRUD de perfiles
- respuesta de `/ocabra/models` con `profiles[]`
- migración inicial desde modelos actuales a perfiles default

### Oleada 2 — Resolución pública y runtime

Entrega:
- `resolve_profile()` en OpenAI/Ollama
- `/v1/models` y `/api/tags` exponiendo perfiles
- fallback legacy configurable desde model_id canónico al perfil default
- `ModelManager` y `WorkerPool` soportando worker key por `(base_model_id, load_overrides_hash)`
- merge de `request_defaults` y `assets` en el forwarding

### Oleada 3 — Chatterbox

Entrega:
- backend `chatterbox`
- worker FastAPI propio
- `/voices`, `/synthesize`, `/synthesize/stream`
- voice cloning con `voice_ref`
- detección básica de modelos Chatterbox en scanner/registry

### Oleada 4 — UI y cierre

Entrega:
- gestión de perfiles desde `Models`
- modal de perfil con upload de audio de referencia
- indicadores de worker compartido/dedicado
- contratos y documentación actualizados
- tests backend/frontend del flujo completo

## Equipo de agentes

### Agente A — Profiles DB/API

Ownership principal:
- `backend/ocabra/db/model_config.py`
- nueva migración Alembic
- nuevo módulo `backend/ocabra/core/profile_registry.py`
- `backend/ocabra/api/internal/models.py`
- nuevo router `backend/ocabra/api/internal/profiles.py` si conviene separar

Responsabilidades:
- definir `ModelProfile`
- decidir si los assets viven en `JSONB` dentro del perfil o en tabla aparte
- implementar CRUD de perfiles
- implementar upload/delete de assets de perfil
- guardar audio de referencia en ruta controlada, por ejemplo `/data/profiles/{profile_id}/...`
- respuesta de `/ocabra/models` con lista anidada de perfiles
- migración inicial creando perfil default por cada modelo accesible

Restricciones:
- no tocar aún la resolución de `/v1/*` ni `/api/*`
- no tocar aún `WorkerPool`

### Agente B — Profile resolution runtime

Ownership principal:
- `backend/ocabra/api/openai/_deps.py`
- `backend/ocabra/api/openai/models.py`
- `backend/ocabra/api/openai/chat.py`
- `backend/ocabra/api/openai/completions.py`
- `backend/ocabra/api/openai/audio.py`
- `backend/ocabra/api/openai/embeddings.py`
- `backend/ocabra/api/openai/images.py`
- `backend/ocabra/api/openai/pooling.py`
- `backend/ocabra/api/ollama/_mapper.py`
- `backend/ocabra/api/ollama/tags.py`
- `backend/ocabra/api/ollama/chat.py`
- `backend/ocabra/api/ollama/generate.py`
- `backend/ocabra/api/ollama/embeddings.py`
- `backend/ocabra/core/model_manager.py`
- `backend/ocabra/core/worker_pool.py`

Responsabilidades:
- introducir `resolve_profile()`
- hacer que `/v1/models` y `/api/tags` publiquen perfiles, no modelos raw
- implementar fallback legacy opcional desde model_id canónico al perfil default
- mover el control de grupos a perfiles
- añadir worker key derivada de `base_model_id + load_overrides_hash`
- compartir worker cuando el hash coincida
- mergear `request_defaults` y `assets` en el forwarding

Restricciones:
- no diseñar storage de assets; consumir el contrato del Agente A
- no implementar UI

### Agente C — Chatterbox backend

Ownership principal:
- nuevo `backend/ocabra/backends/chatterbox_backend.py`
- nuevo `backend/workers/chatterbox_worker.py`
- `backend/ocabra/main.py`
- `backend/ocabra/config.py`
- `backend/pyproject.toml` o venv dedicado si se decide ese camino
- `backend/ocabra/registry/huggingface.py`
- `backend/ocabra/registry/local_scanner.py`

Responsabilidades:
- backend first-class para Chatterbox
- worker con `/health`, `/info`, `/voices`, `/synthesize`, `/synthesize/stream`
- soporte `voice_ref` como path controlado por oCabra
- mapping de voces OpenAI a speakers Chatterbox
- estimación de VRAM y health check
- registro del backend en `main.py`
- detección básica en scanner/registry

Restricciones:
- no meter fine-tuning
- no redefinir contratos de perfiles sin coordinar con A/B

### Agente D — Models UI / Profiles UI

Ownership principal:
- `frontend/src/pages/Models.tsx`
- nuevos componentes en `frontend/src/components/models/`
- `frontend/src/api/client.ts`
- `frontend/src/types/index.ts`

Responsabilidades:
- vista de modelos con perfiles anidados
- crear/editar/borrar perfiles
- toggle enabled/default
- edición de `request_defaults` y `load_overrides`
- upload de audio de referencia para perfiles TTS
- preview claro: el cliente verá `model="<profile_id>"`
- mostrar si un perfil comparte worker o usa worker dedicado

Restricciones:
- no abordar fine-tuning UI
- no mover Settings salvo que sea imprescindible

### Agente E — Contratos, tests y docs

Ownership principal:
- `docs/CONTRACTS.md`
- `docs/PLAN.md`
- `docs/ROADMAP.md`
- `docs/DOC_AUDIT.md`
- tests backend/frontend asociados

Responsabilidades:
- documentar el contrato `ModelProfile`
- documentar endpoints y payloads de upload de assets
- ajustar roadmap para separar claramente Fase 6 de Fase 7B fine-tuning
- añadir tests de:
  - resolución de perfiles
  - fallback legacy
  - cascada al borrar modelo
  - upload/remove de `voice_ref`
  - worker compartido vs dedicado
  - Chatterbox synthesize/stream

## Dependencias y orden

1. A define modelo de datos, endpoints internos y contrato de assets.
2. E actualiza contratos mínimos en paralelo con A antes de que B/D codifiquen encima.
3. B depende de A para consumir `ModelProfile` y el contrato de assets.
4. C puede avanzar en paralelo a A/B si fija `voice_ref` como path local controlado.
5. D depende de A para CRUD/upload y de B para semántica final visible al cliente.
6. Integración final: A -> B/C -> D -> E.

## Riesgos principales

### 1. Colisión entre grupos y perfiles

Hoy los grupos parecen vincularse a `model_id`. Con perfiles, el control debe pasar a `profile_id`.
Esto toca auth, filtros de `/v1/models` y `/api/tags`, y potencialmente estadísticas.

### 2. Estado del worker vs estado del perfil

Un perfil no debe copiar el runtime completo del modelo base. La UI necesita dejar claro:
- estado del perfil
- modelo base asociado
- si comparte worker con otros perfiles

### 3. Assets como rutas arbitrarias

No se debe aceptar `voice_ref` libre desde cliente final. La UI admin sube un audio y oCabra guarda
la ruta interna. El runtime solo recibe paths ya controlados por el servidor.

### 4. Legacy compatibility

Hay que decidir y aplicar bien el fallback:
- v0 inicial: permitir model_id raw -> perfil default con warning
- siguiente versión: apagarlo por defecto

### 5. Chatterbox dependencies

Es probable que Chatterbox necesite aislamiento tipo Voxtral. Mejor asumir desde el inicio:
- venv dedicado o dependencia opcional aislada
- no ensuciar el runtime principal si rompe otras dependencias TTS

## Criterio de Done

Se considerará terminado este bloque cuando:
- `/v1/models` liste perfiles y la inferencia funcione por `profile_id`
- exista CRUD completo de perfiles desde admin
- un perfil TTS permita subir audio de referencia y usarlo en inferencia
- Chatterbox cargue, sintetice y haga streaming
- el sistema comparta workers cuando toca y separe cuando `load_overrides` cambian
- docs y contratos queden alineados

## Prompts de lanzamiento sugeridos

### Prompt Agente A

Implementa la foundation backend de Model Profiles en oCabra. Tu scope es BD, migración, registry/CRUD interno y assets de perfil. Añade `ModelProfile`, migración, storage de assets de perfil con upload seguro, endpoints internos y `profiles[]` en `/ocabra/models`. No toques aún OpenAI/Ollama resolution ni WorkerPool. Usa `docs/PLAN.md` Fase 6 como intención, pero adapta al código real. Tests obligatorios.

### Prompt Agente B

Implementa la resolución pública por perfiles en oCabra. Tu scope es `/v1/*`, `/api/*`, `ModelManager` y `WorkerPool`. Debes resolver por `profile_id`, soportar fallback legacy configurable, mover el filtrado de acceso a perfiles y soportar worker sharing por `(base_model_id, load_overrides_hash)`. Consume el contrato de `ModelProfile` ya definido por backend. Tests obligatorios. No hagas UI.

### Prompt Agente C

Implementa soporte inicial de Chatterbox como backend first-class en oCabra, sin fine-tuning. Crea backend, worker, config y registro. Debe soportar `/voices`, `/synthesize`, `/synthesize/stream` y `voice_ref` como audio de referencia controlado por oCabra. Aísla dependencias si es necesario. Tests obligatorios.

### Prompt Agente D

Implementa la UI de Model Profiles en la página Models. Debe listar modelos con perfiles anidados, permitir CRUD de perfiles, editar defaults/overrides y subir audio de referencia para perfiles TTS. No abordes fine-tuning ni cambios amplios en Settings. Alinea tipos y cliente API con el backend real. Tests frontend obligatorios.

### Prompt Agente E

Actualiza contratos, roadmap, plan y tests transversales para Model Profiles y Chatterbox. Tu foco es evitar deriva documental y asegurar cobertura de resolución, fallback legacy, upload de assets y runtime TTS. No abras nuevas funcionalidades fuera de este scope.
