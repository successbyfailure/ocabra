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

## Versionado

El número de versión vive en **tres** sitios y **deben coincidir siempre**:

- `backend/ocabra/config.py` → `Settings.app_version` (default; se expone en `/health` y en `app.version`).
- `.env` (y `.env.example`) → `APP_VERSION` — **esto sobreescribe el default de `config.py`** (`pydantic-settings` lee env vars por encima del default), así que si solo tocas `config.py` el valor servido en producción no cambia. Súbelo siempre a la vez.
- `frontend/package.json` → `"version"` — el footer del sidebar (`src/components/layout/Sidebar.tsx`) lo lee en build-time vía `__APP_VERSION__` (inyectado por `vite.config.ts`), así que basta con subir `package.json`; no hay un cuarto sitio que tocar a mano (antes era un string `"v0.5.0"` hardcodeado y por eso se desincronizaba).

Tras cualquier cambio con entidad (feature, fix no trivial, cambio de
comportamiento visible), sube **los tres** a la vez, al mismo valor, y
regenera el lockfile para que no quede desincronizado: `cd frontend && npm
install --package-lock-only`. Sigue semver (`MAJOR.MINOR.PATCH`): `PATCH`
para fixes aislados, `MINOR` para features/cambios de comportamiento,
`MAJOR` solo si el usuario lo pide explícitamente. Cambios triviales
(typos, comentarios, refactor sin efecto observable) no requieren bump.

Si el cambio que motiva el bump lo puede notar alguien que use la API, anota
también la entrada en el changelog que sirve el servidor (ver la sección
siguiente). Hay un test que falla si subes de versión sin hacerlo.

## Documentación que sirve el propio servidor

Además del Swagger en `/docs`, oCabra publica dos páginas en prosa dirigidas a
**quien construye aplicaciones y agentes contra la API**, no a quien desarrolla
oCabra:

| Ruta | Contenido | Fuente |
|------|-----------|--------|
| `/docs/changelog` | Cambios de comportamiento entre versiones, parámetros nuevos y errores corregidos | `backend/ocabra/docs_content/changelog.md` |
| `/docs/models` | Cómo rinde cada modelo en este despliegue | `backend/ocabra/docs_content/models.md` |

Las dos se sirven también en Markdown crudo añadiendo `.md` (misma fuente, sin
maquetar: es lo que consumen los agentes). El renderizador vive en
`backend/ocabra/api/docs_pages.py` y soporta un subconjunto de Markdown a
propósito —encabezados, párrafos, listas, reglas, bloques de código, tablas,
negrita, código en línea y enlaces— para no añadir una dependencia; si
necesitas más sintaxis, amplíalo ahí.

**Cuándo tienes que actualizar el changelog.** Siempre que subas de versión por
algo que un cliente pueda notar. En concreto:

- Cambia lo que devuelve o acepta un endpoint, aunque sea compatible.
- Cambia un valor por defecto, un límite o un tope (contexto, salida,
  concurrencia, timeouts).
- Arreglas un bug que el usuario estaba sufriendo: dilo aunque el arreglo sea
  interno, porque es lo que le explica por qué algo fallaba.
- Añades o retiras un endpoint, un backend o una capacidad.

Marca con **Acción** lo que obligue a tocar código del cliente, y escribe desde
su punto de vista: qué le pasaba antes y qué le pasa ahora. Los refactors,
cambios de UI y deuda interna NO van aquí.

Hay dos tests que lo sostienen y que fallarán si te lo saltas:

- `test_el_changelog_menciona_la_version_mas_reciente` — el changelog tiene que
  mencionar la serie de la versión servida, para que no se quede viejo al subir
  versión.
- `test_la_referencia_documenta_todos_los_endpoints_de_batches` — la tabla de
  Batches de la descripción de `/docs` (en `backend/ocabra/main.py`,
  `_API_DESCRIPTION`) tiene que listar todo lo que hay en `ALLOWED_ENDPOINTS`.

Ojo con el proxy: las subrutas de `/docs` necesitan su propia regla en
`caddy/Caddyfile` (`handle /docs/*`). Los tests cubren el enrutado de FastAPI,
no el de Caddy, así que una página puede pasar los tests y dar 404 de puertas
afuera.

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
