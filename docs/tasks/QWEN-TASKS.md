# Tareas para Qwen

Antes de empezar cualquier tarea: lee `AGENTS.md`, `docs/CONTRACTS.md` y `docs/CONVENTIONS.md`.
Las tareas de Qwen son las más acotadas y con el contrato más explícito. Cada paso está detallado para minimizar ambigüedad.

---

## TAREA Q-1 — Stream 3-B: Ollama API Compatibility

**Estado:** ESPERAR a que `feat/2-A-vllm` Y `feat/1-B-model-manager` estén mergeadas
**Rama:** `feat/3-B-ollama-api`
**Briefing completo:** `docs/agents/stream-3B-ollama-api.md`

### Cuándo entrar
Cuando `docs/agents/stream-2A-vllm.md` tenga `[x] Completado`.
Puede correr en paralelo con la tarea C-5 (OpenAI API de Claude).

### Qué hacer, paso a paso

**Paso 1 — Name Mapper**
Crear `backend/ocabra/api/ollama/name_mapper.py`:

```python
class OllamaNameMapper:
    # Mapa predefinido de nombres Ollama → model_id interno
    KNOWN_MAP = {
        "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
        "phi4:14b": "microsoft/phi-4",
        "gemma2:9b": "google/gemma-2-9b-it",
        "gemma2:27b": "google/gemma-2-27b-it",
        "nomic-embed-text:latest": "nomic-ai/nomic-embed-text-v1.5",
        "mxbai-embed-large:latest": "mixedbread-ai/mxbai-embed-large-v1",
    }

    def to_internal(self, ollama_name: str) -> str:
        """llama3.2:3b → meta-llama/Llama-3.2-3B-Instruct.
        Si no está en el mapa, usa el nombre tal cual como model_id."""

    def to_ollama(self, model_id: str) -> str:
        """meta-llama/Llama-3.2-3B-Instruct → llama3.2:3b.
        Si no está en el mapa inverso, convierte: org/Name-Size → name:size."""
```

**Paso 2 — GET /api/tags**
Crear `backend/ocabra/api/ollama/tags.py`:
- Llama a `model_manager.list_states()`.
- Filtra: solo modelos con `backend_type="vllm"` (Ollama no tiene imagen/audio).
- Traduce cada `ModelState` al formato Ollama tags (ver briefing §GET /api/tags).
- Campos a rellenar:
  - `name`: `name_mapper.to_ollama(model.model_id)`
  - `model`: igual que `name`
  - `modified_at`: `model.loaded_at` o fecha de fichero del modelo
  - `size`: suma de ficheros del directorio del modelo en bytes
  - `digest`: `sha256:` + los primeros 12 chars del hash del `model_id`
  - `details.parameter_size`: inferir del nombre del modelo (buscar patrón `7B`, `14B`, `70B`, etc.)
  - `details.quantization_level`: `F16` por defecto

**Paso 3 — POST /api/show**
Crear `backend/ocabra/api/ollama/show.py`:
- Lee `ModelState` del `model_manager`.
- Retorna: `modelfile` (generar uno básico), `parameters` (de la config vLLM), `template` (del `tokenizer_config.json` si existe, si no una plantilla genérica de chat), `details`, `model_info`.

**Paso 4 — POST /api/pull**
Crear `backend/ocabra/api/ollama/pull.py`:
- Lee `name` del body.
- Convierte a `model_ref` con `name_mapper.to_internal()`.
- Llama a `download_manager.enqueue(source="ollama", model_ref=model_ref)`.
- Si `stream=true` (default): retorna stream NDJSON con progreso suscrito a Redis.
  - Mensajes esperados:
    ```
    {"status": "pulling manifest"}
    {"status": "pulling layer", "digest": "sha256:...", "total": N, "completed": M}
    {"status": "verifying sha256 digest"}
    {"status": "writing manifest"}
    {"status": "success"}
    ```

**Paso 5 — POST /api/generate (completions)**
Crear `backend/ocabra/api/ollama/generate.py`:
- Traduce request Ollama → OpenAI completions:
  - `model` → `name_mapper.to_internal(model)`
  - `prompt` → `prompt`
  - `options.temperature` → `temperature`, etc. (usar tabla OPTION_MAP del briefing)
  - `images[]` → añadir como imagen en el prompt si el modelo tiene `vision=True`
- Llama a `ensure_loaded(model_id)` (importar de openai compat o duplicar lógica).
- Proxy a vLLM `/v1/completions`.
- Si `stream=true`: convierte SSE → NDJSON Ollama en tiempo real.
- Si `stream=false`: retorna respuesta completa en formato Ollama generate.

**Paso 6 — POST /api/chat**
Crear `backend/ocabra/api/ollama/chat.py`:
- Traduce messages Ollama → messages OpenAI:
  - `{"role": "user", "content": "texto"}` → igual
  - `{"role": "user", "content": "texto", "images": ["base64..."]}` → añade `image_url` en el content
- Proxy a vLLM `/v1/chat/completions`.
- Si `stream=true`: convierte SSE OpenAI → NDJSON Ollama.
  - Chunk OpenAI: `data: {"choices":[{"delta":{"content":"texto"}}]}\n\n`
  - Chunk Ollama: `{"model":"...","created_at":"...","message":{"role":"assistant","content":"texto"},"done":false}\n`
  - Último chunk Ollama incluye: `"done":true,"total_duration":N,"prompt_eval_count":N,"eval_count":N,"eval_duration":N`

**Paso 7 — POST /api/embeddings**
Crear `backend/ocabra/api/ollama/embeddings.py`:
- Traduce `{"model": "...", "input": "texto"}` → OpenAI embeddings format.
- Proxy a vLLM `/v1/embeddings`.
- Respuesta Ollama: `{"model": "...", "embeddings": [[0.1, 0.2, ...]]}`.

**Paso 8 — DELETE /api/delete**
Crear `backend/ocabra/api/ollama/delete.py`:
- `{"name": "llama3.2:3b"}` → `model_manager.delete(name_mapper.to_internal(name))`.

**Paso 9 — Router**
Crear `backend/ocabra/api/ollama/__init__.py` con un `APIRouter` que incluya todos los endpoints.
Registrar en `backend/ocabra/main.py` en la sección `# ROUTERS`.

**Paso 10 — Tests**
Crear `backend/tests/test_ollama_api.py`:
- Test de `name_mapper`: round-trip `llama3.2:3b → internal → llama3.2:3b`.
- Test de `GET /api/tags`: mock de `model_manager`, verificar campos de respuesta.
- Test de stream NDJSON: mock de SSE de vLLM, verificar que se convierte correctamente.
- Test de `POST /api/pull`: mock de download_manager, verificar mensajes NDJSON de progreso.

### Señal de finalización
Commit en rama `feat/3-B-ollama-api`:
```
feat(3-B): implement Ollama API compatibility layer
```
Marcar `docs/agents/stream-3B-ollama-api.md` → `[x] Completado`.

---

## TAREA Q-2 — Stream 4-D: Settings UI

**Estado:** ESPERAR a que `feat/1-D-frontend-base` esté mergeada
**Rama:** `feat/4-D-settings`
**Briefing completo:** `docs/agents/stream-4-frontend-features.md §Stream 4-D`
**Paralelo con:** X-5, X-6, X-7 de Codex

### Cuándo entrar
Cuando `docs/agents/stream-1D-frontend-base.md` tenga `[x] Completado`.

### Qué hacer, paso a paso

**Paso 1 — Settings page scaffold**
Crear `frontend/src/pages/Settings.tsx`:
- Layout con tabs verticales o secciones: General, GPUs, LiteLLM, Storage, Schedules Globales.
- Carga config inicial: `useEffect` que llama `api.config.get()` al montar.
- Botón "Guardar" que llama `api.config.patch(changes)` y muestra toast de éxito/error.

**Paso 2 — GeneralSettings**
Crear `frontend/src/components/settings/GeneralSettings.tsx`:
- Input: carpeta de modelos (MODELS_DIR). Solo informativo si no es editable en runtime.
- Select: nivel de log (DEBUG, INFO, WARNING, ERROR).
- Input numérico: idle timeout por defecto (segundos) para modelos `on_demand`.
- Input numérico: buffer de VRAM reservada (MB, default 512).

**Paso 3 — GPUSettings**
Crear `frontend/src/components/settings/GPUSettings.tsx`:
- Radio buttons: GPU preferida por defecto (GPU 0 - RTX 3060 12GB | GPU 1 - RTX 3090 24GB).
- Slider: umbral de presión de VRAM (% de VRAM usada que trigger evicción, default 90%).
- Input numérico: temperatura máxima de alerta (°C, default 80).

**Paso 4 — LiteLLMSettings**
Crear `frontend/src/components/settings/LiteLLMSettings.tsx`:
- Input: URL del proxy LiteLLM (e.g., `http://litellm:4000`).
- Input (password): API key de admin de LiteLLM.
- Toggle: sync automático al añadir/eliminar modelos.
- Botón "Sincronizar ahora": llama `api.config.syncLiteLLM()`, muestra resultado (`"12 modelos sincronizados"`).
- Estado de última sync: timestamp + icono de éxito (verde) o error (rojo) + mensaje de error si aplica.

**Paso 5 — StorageSettings**
Crear `frontend/src/components/settings/StorageSettings.tsx`:
- Lista de modelos instalados con uso de disco (barra + GB).
- Datos de `api.registry.listLocal()`.
- Botón "Limpiar caché HuggingFace": llama `DELETE /ocabra/registry/hf/cache` (si ese endpoint existe, si no mostrar instrucción manual).
- Total de espacio usado por todos los modelos.

**Paso 6 — GlobalSchedules**
Crear `frontend/src/components/settings/GlobalSchedules.tsx`:
- Explicación: "Los schedules globales se aplican a todos los modelos salvo que tengan su propio schedule."
- Lista de schedules globales existentes con botón editar/borrar.
- Botón "Añadir schedule": abre el mismo componente `ScheduleEditor` de los modelos (importar de `components/models/ScheduleEditor.tsx`, si no existe aún, crear una versión simplificada aquí).
- Llama a `PATCH /ocabra/config` con la lista de schedules actualizada.

**Paso 7 — Tests**
Crear `frontend/src/__tests__/Settings.test.tsx`:
- Test de carga inicial: mock de `api.config.get()`, verificar que los campos muestran los valores correctos.
- Test de LiteLLM sync: mock de `api.config.syncLiteLLM()`, verificar mensaje de resultado.
- Test de guardar cambios: mock de `api.config.patch()`, verificar llamada con los campos modificados.

### Señal de finalización
Commit en rama `feat/4-D-settings`:
```
feat(4-D): implement settings UI with GPU, LiteLLM and storage config
```
Marcar `docs/agents/stream-4-frontend-features.md` → `[x] 4-D completado`.

---

## Resumen de orden de ejecución para Qwen

```
[ ESPERAR: feat/2-A-vllm mergeada ]
         ↓
     Q-1 (Ollama API)    ← lanzar en cuanto vLLM esté listo

[ ESPERAR: feat/1-D-frontend-base mergeada ]
         ↓
     Q-2 (Settings UI)   ← en paralelo con Q-1 si los tiempos coinciden
```

Ambas tareas son independientes entre sí y pueden ejecutarse a la vez si las dependencias de cada una están cumplidas.
