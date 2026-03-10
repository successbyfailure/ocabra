# CLAUDE.md — Instrucciones para agentes Claude Code

Este fichero es leído automáticamente por Claude Code al abrir el proyecto.

## Proyecto: oCabra

Servidor de modelos de IA multi-GPU compatible con APIs OpenAI y Ollama.
Lee `docs/PLAN.md` para el plan completo y `AGENTS.md` para las reglas generales de colaboración multi-agente.

## Stack rápido

| Capa | Tecnología |
|------|-----------|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2.0 async, Pydantic v2 |
| DB | PostgreSQL 16 + Alembic (migraciones) |
| Cache | Redis 7 |
| GPU | pynvml |
| Backends IA | vLLM, Diffusers, faster-whisper, Transformers |
| Frontend | React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui |
| Deploy | Docker Compose + Caddy |

## Comandos clave

```bash
# Levantar el stack completo
docker compose up -d

# Solo desarrollo backend (fuera de Docker)
cd backend && python -m uvicorn ocabra.main:app --reload --port 8000

# Migraciones
cd backend && alembic upgrade head
cd backend && alembic revision --autogenerate -m "descripcion"

# Frontend dev
cd frontend && npm run dev

# Tests backend
cd backend && pytest

# Tests frontend
cd frontend && npm test

# Linting / formato
cd backend && ruff check . && ruff format .
cd frontend && npm run lint
```

## Convenciones obligatorias

- Sigue `docs/CONVENTIONS.md` para nombres, estructura y estilo.
- Sigue `docs/CONTRACTS.md` antes de implementar interfaces entre módulos.
- **No toques código fuera de tu stream asignado** (ver `AGENTS.md` y `docs/agents/`).
- Todos los endpoints nuevos deben tener docstring con descripción, parámetros y respuesta.
- Usa `async/await` en todo el backend. No código síncrono en handlers de FastAPI.
- Los modelos SQLAlchemy van en `backend/ocabra/db/`. Los schemas Pydantic van en `backend/ocabra/schemas/`.

## Estructura de directorios resumida

```
ocabra/
├── backend/ocabra/
│   ├── api/openai/       # Compatibilidad OpenAI /v1/*
│   ├── api/ollama/       # Compatibilidad Ollama /api/*
│   ├── api/internal/     # API interna /ocabra/*
│   ├── core/             # GPU manager, model manager, scheduler, worker pool
│   ├── backends/         # vLLM, Diffusers, Whisper, TTS
│   ├── registry/         # HuggingFace, Ollama registry, scanner local
│   ├── integrations/     # LiteLLM sync
│   ├── stats/            # Métricas, energía, agregación
│   └── db/               # SQLAlchemy models
├── frontend/src/
│   ├── pages/            # Dashboard, Models, Explore, Playground, Stats, Settings
│   ├── components/       # Componentes reutilizables
│   ├── api/              # Cliente API tipado
│   └── stores/           # Zustand stores
└── workers/              # Scripts de workers de backends IA
```

## Lo que NO debes hacer

- No hagas `git push` sin indicación explícita del usuario.
- No borres migraciones de Alembic existentes.
- No almacenes datos de conversaciones (el servidor es stateless).
- No instales dependencias globales; usa el entorno del proyecto.
- No modifiques `docker-compose.yml` sin entender el impacto en otros servicios.
- No uses `time.sleep()` en código async; usa `asyncio.sleep()`.
- No escribas ficheros de modelos IA (*.safetensors, *.gguf, etc.) en el repo.

## Flujo de trabajo recomendado

1. Lee tu briefing de stream en `docs/agents/`.
2. Consulta `docs/CONTRACTS.md` para las interfaces que consumes o produces.
3. Implementa siguiendo `docs/CONVENTIONS.md`.
4. Escribe tests para lo que implementas.
5. Marca tu progreso en el briefing de tu stream si está disponible.
