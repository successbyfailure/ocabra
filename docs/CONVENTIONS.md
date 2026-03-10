# Convenciones de código

## Python (backend)

### Estilo
- Formateador: **ruff format** (compatible Black, line-length 100)
- Linter: **ruff check** con reglas: E, F, I, UP, B, ASYNC
- Type hints en toda función pública. Retornos anotados siempre.
- No `Any` salvo en código de proxy/reenvío donde el tipo es genuinamente dinámico.

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

# 3. Imports internos (relativos)
from ..core.gpu_manager import GPUState
from .base import BackendInterface

# 4. Constants

# 5. Clases de datos / schemas

# 6. Clase principal

# 7. Funciones helper privadas (si aplica)
```

### Async
- Todos los handlers FastAPI son `async def`.
- No `requests` library; usar `httpx.AsyncClient`.
- No `time.sleep`; usar `asyncio.sleep`.
- Los procesos de workers se lanzan con `asyncio.create_subprocess_exec`.
- Contextos de vida (`lifespan`) para startup/shutdown, nunca `@app.on_event`.

### SQLAlchemy
- Usar siempre `AsyncSession`.
- Schemas Pydantic en `ocabra/schemas/` (no en `db/`).
- Modelos SQLAlchemy en `ocabra/db/`.
- Nombrar tablas en plural, snake_case: `model_configs`, `request_stats`.

### Pydantic
- Usar Pydantic v2 (`model_config = ConfigDict(...)`).
- Schemas de request: `*Request` o `*Create` / `*Update`.
- Schemas de response: `*Response` o `*Out`.

### Tests
- Framework: **pytest + pytest-asyncio**
- Ficheros: `backend/tests/test_<módulo>.py`
- Fixtures compartidas en `backend/tests/conftest.py`
- Mocks de GPU/procesos con `unittest.mock.AsyncMock`

---

## TypeScript / React (frontend)

### Estilo
- Formateador: **Prettier** (config en `frontend/.prettierrc`)
- Linter: **ESLint** con plugin react + typescript
- Siempre TypeScript estricto (`strict: true` en tsconfig)

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

// Types/Interfaces: PascalCase, en fichero types.ts por dominio
interface GPUState { ... }
interface ModelState { ... }
type LoadPolicy = "pin" | "warm" | "on_demand"
```

### Estructura de componente
```tsx
// 1. Imports React
import { useState, useEffect } from "react"

// 2. Imports de librerías
import { Badge } from "@/components/ui/badge"

// 3. Imports internos
import { useModelStore } from "@/stores/modelStore"
import type { ModelState } from "@/types/models"

// 4. Types locales (si son solo para este componente)

// 5. Componente (arrow function)
export const ModelCard = ({ model }: { model: ModelState }) => {
  // hooks primero
  // handlers
  // render
}
```

### Estado global (Zustand)
- Un store por dominio: `gpuStore`, `modelStore`, `statsStore`, `downloadStore`
- Actions en el propio store, no en componentes
- Tipos explícitos para el state y las actions

### Estilos
- TailwindCSS utility classes exclusivamente
- Componentes base de shadcn/ui (en `src/components/ui/`)
- No CSS modules ni styled-components
- Responsive: mobile-first, aunque el target principal es desktop

### Tests
- Framework: **Vitest + Testing Library**
- Ficheros: `src/__tests__/<Componente>.test.tsx`
- Mocks de API en `src/__tests__/mocks/`

---

## Docker

- Imágenes base:
  - Backend: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + python
  - Frontend: `node:20-alpine` (build) + `nginx:alpine` (serve)
- Multi-stage builds obligatorios para frontend
- Variables de entorno via `env_file: .env`, nunca hardcodeadas en `docker-compose.yml`
- Healthchecks en todos los servicios

---

## Git

### Commit messages
```
<tipo>(<scope>): <descripción corta en imperativo>

[cuerpo opcional]
```

Tipos: `feat` | `fix` | `refactor` | `test` | `docs` | `chore` | `perf`

Scopes de ejemplo: `gpu-manager`, `vllm-backend`, `openai-api`, `dashboard`, `docker`, `deps`

```
feat(gpu-manager): add VRAM pressure detection with configurable threshold
fix(vllm-backend): handle process crash on OOM error
docs(contracts): add Redis key conventions
```

### Branches
```
main           # producción / estado estable
dev            # integración de streams
feat/1-A-gpu-manager
feat/1-B-model-manager
feat/2-A-vllm
feat/3-A-openai-api
fix/<descripción>
```

Cada stream trabaja en su propia rama `feat/<stream>` y hace PR a `dev`.
