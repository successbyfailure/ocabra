# Briefing: Fase 0 — Fundación

**Agente único, trabajo secuencial. Todo lo demás depende de esta fase.**

## Objetivo

Crear la estructura base funcional del proyecto: Docker Compose operativo,
FastAPI con health endpoint, base de datos con migraciones, Redis conectado,
y React app con layout y routing. Al terminar, cualquier otro agente puede
arrancar su stream sin bloquearse en infra.

## Entregables (Definition of Done)

- [ ] `docker compose up` levanta todos los servicios sin errores
- [ ] `GET /health` retorna `{"status": "ok", "version": "0.1.0"}`
- [ ] Alembic: migración inicial aplicada, tablas creadas
- [ ] Redis: cliente async conectado, pub/sub helper funcional
- [ ] React app: arranca en dev, tiene routing con 5 páginas stub, layout con sidebar
- [ ] `.env.example` completo con todas las variables definidas en `docs/CONTRACTS.md`

## Ficheros a crear

### Raíz
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `.env.example`

### Backend
- `backend/Dockerfile`
- `backend/pyproject.toml`
- `backend/alembic.ini`
- `backend/alembic/env.py`
- `backend/alembic/versions/` (vacío con `.gitkeep`)
- `backend/ocabra/__init__.py`
- `backend/ocabra/main.py`
- `backend/ocabra/config.py`
- `backend/ocabra/database.py`
- `backend/ocabra/redis_client.py`
- `backend/ocabra/db/__init__.py`
- `backend/ocabra/db/model_config.py`
- `backend/ocabra/db/stats.py`
- `backend/ocabra/db/server_config.py`
- `backend/ocabra/schemas/__init__.py`
- `backend/ocabra/api/__init__.py`
- `backend/ocabra/api/health.py`
- `backend/tests/__init__.py`
- `backend/tests/conftest.py`

### Frontend
- `frontend/Dockerfile`
- `frontend/nginx.conf`
- `frontend/package.json`
- `frontend/tsconfig.json`
- `frontend/vite.config.ts`
- `frontend/tailwind.config.ts`
- `frontend/postcss.config.js`
- `frontend/index.html`
- `frontend/src/main.tsx`
- `frontend/src/App.tsx`
- `frontend/src/components/ui/` (shadcn/ui init básico)
- `frontend/src/pages/Dashboard.tsx` (stub)
- `frontend/src/pages/Models.tsx` (stub)
- `frontend/src/pages/Explore.tsx` (stub)
- `frontend/src/pages/Playground.tsx` (stub)
- `frontend/src/pages/Stats.tsx` (stub)
- `frontend/src/pages/Settings.tsx` (stub)
- `frontend/src/components/layout/Sidebar.tsx`
- `frontend/src/components/layout/Layout.tsx`
- `frontend/src/types/` (ficheros de tipos base vacíos)
- `frontend/src/api/client.ts` (estructura base, métodos placeholder)
- `frontend/src/stores/gpuStore.ts` (estructura base)
- `frontend/src/stores/modelStore.ts` (estructura base)

### Caddy
- `caddy/Caddyfile`

## Docker Compose — servicios requeridos

```yaml
services:
  api:
    build: ./backend
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
    ports: ["8000:8000"]
    depends_on: [postgres, redis]

  frontend:
    build: ./frontend
    ports: ["3000:80"]

  postgres:
    image: postgres:16-alpine
    volumes: [postgres_data:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine

  caddy:
    image: caddy:2-alpine
    ports: ["80:80", "443:443"]
    volumes: [./caddy/Caddyfile:/etc/caddy/Caddyfile]
```

## Dependencias Python (pyproject.toml)

```toml
[project]
name = "ocabra"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "alembic>=1.14",
    "redis[asyncio]>=5.2",
    "httpx>=0.28",
    "pynvml>=11.5",
    "apscheduler>=3.10",
    "structlog>=24.4",
    "huggingface-hub>=0.27",
]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ASYNC"]
```

## Dependencias Frontend (package.json)

```json
{
  "dependencies": {
    "react": "^18.3",
    "react-dom": "^18.3",
    "react-router-dom": "^6.28",
    "zustand": "^5.0",
    "recharts": "^2.14",
    "@radix-ui/react-*": "latest",
    "lucide-react": "^0.469",
    "clsx": "^2.1",
    "tailwind-merge": "^2.6"
  },
  "devDependencies": {
    "typescript": "^5.7",
    "vite": "^6.0",
    "@vitejs/plugin-react": "^4.3",
    "tailwindcss": "^3.4",
    "autoprefixer": "^10.4",
    "eslint": "^9.17",
    "vitest": "^2.1",
    "@testing-library/react": "^16.1"
  }
}
```

## Notas

- El backend levanta en modo `--reload` en desarrollo.
- En producción, Caddy hace reverse proxy: `/v1/*`, `/api/*`, `/ocabra/*` → `api:8000`, todo lo demás → `frontend:80`.
- El primer Alembic migration debe crear las 4 tablas definidas en `docs/CONTRACTS.md §7`.
- `main.py` debe usar `lifespan` context manager, no `@app.on_event`.

## Estado

- [x] En progreso
- [x] Completado — mergeado en `main` (commit 80c71d7)
