# Convenciones de código

Describe lo que hace el repositorio hoy (medido el 2026-11-19). Donde el
código existente no sigue una regla, se dice; el código nuevo sigue la regla.

## Python (backend)

### Estilo
- Formateador: **ruff format** (compatible Black, `line-length = 100`). Formatea
  solo los ficheros que tocas: no pases `ruff format` a directorios enteros.
- Linter: **ruff check** con `select = ["E", "F", "I", "UP", "B", "ASYNC"]`,
  ignorando `E501` y `UP042` (los `(str, Enum)` se persisten y serializan; ver
  `backend/pyproject.toml`). `alembic/versions` queda excluido.
- `Depends`, `Query`, `Body`… y `require_role` están exentos de B008 por nombre
  (`extend-immutable-calls`): es el idiom de FastAPI.
- Python 3.11 en las imágenes (`requires-python = ">=3.11"`).
- Type hints en toda función pública. Retornos anotados siempre.
- No `Any` salvo en código de proxy/reenvío donde el tipo es genuinamente dinámico y algunos módulos legacy ya existentes; para código nuevo, preferir tipos explícitos.

### Naming
```python
# Módulos y ficheros: snake_case
gpu_manager.py
vllm_backend.py

# Clases: PascalCase
class GPUManager:
class VLLMBackend(BackendInterface):

# Funciones y métodos: snake_case, verbos
async def load_model(model_id: str) -> ModelState: ...
async def get_gpu_state(index: int) -> GPUState: ...

# IDs de modelo: canónico backend/model
# Ejemplos: vllm/meta-llama/Llama-3.3-70B-Instruct, llama_cpp/gemma4-26b
# Si necesitas el id nativo del runtime usa backend_model_id. Los clientes
# piden por profile_id (perfil), no por el id canónico.

# Constantes: UPPER_SNAKE_CASE
MAX_VRAM_BUFFER_MB = 512
DEFAULT_IDLE_TIMEOUT_S = 300

# Variables de instancia privadas: _prefijo
self._worker_process: asyncio.subprocess.Process | None = None
```

### Estructura de un módulo backend
```python
# 1. Imports stdlib
import asyncio
from dataclasses import dataclass

# 2. Imports third-party
import httpx
from pydantic import BaseModel

# 3. Imports internos, absolutos desde el paquete
from ocabra.backends.base import BackendInterface
from ocabra.core.gpu_manager import GPUState

# 4. Constants

# 5. Clases de datos / schemas

# 6. Clase principal

# 7. Funciones helper privadas (si aplica)
```

El orden de los bloques de imports lo impone ruff (regla `I`). Los imports
internos son absolutos (`from ocabra…`: ~1.100 líneas); los relativos (~56, en
20 ficheros) se usan solo dentro de un mismo paquete de rutas (`api/openai/`,
`api/ollama/`: `from ._deps import …`) y no se deben extender fuera de ahí.
Muchos módulos importan dentro de la función para evitar ciclos o cargas
pesadas al arrancar; es deliberado.

### Async
- Todos los handlers FastAPI son `async def`.
- No `requests` library; usar `httpx.AsyncClient`. En inferencia se abre un
  cliente por petición a propósito (`worker_pool.py`); no lo sustituyas por
  uno compartido.
- No `time.sleep`; usar `asyncio.sleep`. Lo bloqueante (NVML, ficheros
  grandes, inferencia en CPU) va a `asyncio.to_thread`.
- Los procesos de workers se lanzan con `asyncio.create_subprocess_exec`.
- Contextos de vida (`lifespan`) para startup/shutdown, no `@app.on_event`.
  Excepción heredada: `ocabra/docker_agent/server.py` todavía usa
  `@app.on_event("startup")`.

### SQLAlchemy
- Usar siempre `AsyncSession`.
- Modelos SQLAlchemy en `ocabra/db/` (33 tablas; índice en
  `docs/CONTRACTS.md` §7). Todo cambio de esquema va con una migración de
  Alembic nueva; las existentes no se tocan ni se borran.
- Schemas Pydantic compartidos en `ocabra/schemas/` (81 clases). Los cuerpos
  que solo usa un router viven en el propio módulo de `ocabra/api/` (58
  clases): es la práctica real, aceptable mientras no se compartan; si otro
  módulo los necesita, muévelos a `schemas/`.
- Tablas en plural y snake_case (`model_configs`, `request_stats`); las de
  unión, con los dos nombres (`user_groups`, `group_models`).

### Pydantic
- Pydantic v2 (`model_config = ConfigDict(...)`; no hay `class Config`).
- Schemas de petición: `*Request`, `*Create`, `*Update` o `*Patch`.
- Schemas de respuesta: `*Response` o `*Out`.
- Los cuerpos de la API interna usan alias camelCase (`populate_by_name`)
  cuando el panel los manda así; mira el router vecino antes de elegir.
- Ajustes: un campo en `ocabra/config.py` **y** su línea en `.env.example`
  (lo comprueba `tests/test_env_example.py`).

### Tests
- Framework: **pytest + pytest-asyncio** (`asyncio_mode = "auto"`: no hace
  falta marcar los tests async).
- Ficheros: `backend/tests/test_<tema>.py`. Hoy hay 155 en la raíz de `tests/`
  y el resto en `core/`, `agents/`, `integrations/` y `workers/`; nombres en
  inglés y en español conviven. Para código nuevo, pon el test en el
  subdirectorio que espeje el paquete (`tests/core/` para `ocabra/core/`).
- Fixtures compartidas en `backend/tests/conftest.py`; las rutas efectivas de
  la app, con `tests/_app_routes.effective_routes` (FastAPI 0.163 no aplana
  `app.routes`).
- Mocks de GPU/procesos con `unittest.mock.AsyncMock`.
- Se ejecutan con y sin Postgres (`env -u DATABASE_URL pytest`): los tests
  que necesitan BD se saltan solos sin ella.
- Invariantes del repositorio como tests: la versión y el changelog,
  `ALLOWED_ENDPOINTS` de Batches frente a `/docs`, todas las rutas en
  `docs/API.md` (`test_api_md.py`), `.env.example` frente a `config.py`.

---

## TypeScript / React (frontend)

### Estilo
- No hay formateador configurado (Prettier no está instalado): sigue el estilo
  del fichero (comillas dobles, sin punto y coma, 2 espacios).
- Linter: **ESLint 9** (`eslint.config.js`) con `@eslint/js`,
  `typescript-eslint` (recommended) y `eslint-plugin-react-hooks`
  (`rules-of-hooks` como error, `exhaustive-deps` como aviso). No hay
  `eslint-plugin-react`. `npm run lint`.
- TypeScript estricto (`strict: true`); `npm run build` ejecuta `tsc -b`.
- Imports internos con el alias `@/` (`@/stores/modelStore`); relativos solo
  entre ficheros vecinos.

### Naming
```tsx
// Componentes: PascalCase, ficheros .tsx
GpuCard.tsx
ModelBadges.tsx

// Hooks: camelCase con prefijo "use"
useWebSocket.ts
useGpuStats.ts

// Stores (Zustand): camelCase con sufijo "Store"
gpuStore.ts
modelStore.ts

// API client methods: camelCase, verbos
getModels()
loadModel(modelId: string)
streamChatCompletion(...)

// Types/Interfaces: PascalCase, en src/types/ (index.ts y un fichero por
// dominio grande: agents.ts, backends.ts)
interface GPUState { ... }
interface ModelState { ... }
type LoadPolicy = "pin" | "warm" | "on_demand"
```

### Estructura de componente
```tsx
// 1. Imports React
import { useState, useEffect } from "react"

// 2. Imports de librerías
import { toast } from "sonner"

// 3. Imports internos
import { ConfirmDialog } from "@/components/common/ConfirmDialog"
import { useModelStore } from "@/stores/modelStore"
import type { ModelState } from "@/types"

// 4. Types locales (si son solo para este componente)

// 5. Componente: export function con nombre (así están ~175 de 180)
export function ModelCard({ model }: { model: ModelState }) {
  // hooks primero
  // handlers
  // render
}
```

### Estado global (Zustand)
- Un store por dominio en `src/stores/`: `agentsStore`, `authStore`,
  `backendsStore`, `downloadStore`, `gpuStore`, `mcpStore`, `modelStore`,
  `serviceStore`.
- Llamadas HTTP en `src/api/` (`client.ts` y un módulo por dominio grande);
  hooks en `src/hooks/` (`useWebSocket`, `useSSE`…).
- Actions en el propio store, no en componentes
- Tipos explícitos para el state y las actions

### Estilos
- TailwindCSS utility classes exclusivamente
- Primitivas de Radix UI (`@radix-ui/react-*`) directamente; los componentes base propios viven en `src/components/common/` (no hay shadcn/ui)
- No CSS modules ni styled-components
- Responsive: mobile-first, aunque el target principal es desktop

### Tests
- Framework: **Vitest + Testing Library**
- Ficheros: `src/__tests__/<Componente>.test.tsx` (36), o `__tests__/`
  junto al componente en las carpetas con muchos (`model-editor/`,
  `federation/`, `common/`).
- Los mocks de API se declaran en cada test con `vi.mock(...)`; no hay
  carpeta de mocks compartidos.
- e2e con Playwright (`@playwright/test`), aparte de `npm test`.

---

## Docker

- Imágenes base:
  - api y docker-agent (`backend/Dockerfile.slim`, la de por defecto):
    `python:3.11-slim-bookworm` más `cuda-nvcc-13-0` y `libcublas-dev-13-0`
    del repositorio de NVIDIA (para compilar backends en el sitio). No trae
    backends: se instalan en `/data/backends`.
  - Imagen fat heredada (`backend/Dockerfile`, `docker-compose.fat.yml`):
    `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
  - Backends nativos publicados: `backend/Dockerfile.backend-native`
    (`nvidia/cuda:<ver>-devel-ubuntu22.04` para compilar).
  - Frontend: `node:20-alpine` (build) + `nginx:alpine` (serve), multi-stage.
  - gateway `python:3.11-slim`, hw-monitor `python:3.12-slim`.
- La api corre sin root (`user: ${OCABRA_UID}:${OCABRA_GID}`) y sin socket de
  Docker: las operaciones de Docker van por `docker-agent`.
- Variables: los secretos y ajustes viven en `.env`. Hoy conviven
  `env_file: .env` (api, docker-agent, acestep) y bloques `environment:` con
  `${VAR:-defecto}`; los servicios que no deben ver todos los secretos
  (postgres, redis, hw-monitor, gateway) reciben solo sus variables.
- Healthcheck: lo tienen api, docker-agent, postgres, redis y los servicios
  opcionales; frontend, hw-monitor, ollama, gateway y caddy aún no. Un
  servicio nuevo debe traerlo.

---

## Git

### Commit messages
```
<tipo>(<scope>): <descripción corta>

[cuerpo opcional]
```

En español, describiendo el efecto («borrar el fichero de entrada de un batch
terminado ya no da 500»). Si el cambio sube versión, la versión va al final
entre paréntesis (`(v0.118.1)`).

Tipos: `feat` | `fix` | `refactor` | `test` | `docs` | `chore` | `perf`

Scopes de ejemplo: `gpu-manager`, `vllm-backend`, `openai-api`, `dashboard`, `docker`, `deps`

```
fix(batches): borrar el fichero de entrada de un batch terminado ya no da 500
refactor(vram): una sola cascada de expulsión, con el orden de la de presión
docs(vram): docstring del módulo del árbitro al día
```

### Branches

`main` es la rama de trabajo y de producción (no hay rama `dev`). Para cambios
largos o en paralelo con otras sesiones, usa una rama `feat/<tema>` o
`fix/<tema>` en un worktree propio y fusiona en `main` cuando esté verificada.
