# AGENTS.md — Guía para agentes de IA

Lo lee cualquier agente (Claude, GPT, Gemini…) que trabaje en este repositorio.
Las instrucciones operativas están en `CLAUDE.md`: léelo entero primero. Si hay
conflicto, manda la seguridad y el alcance que pida el usuario, luego
`CLAUDE.md`, luego este fichero.

## Proyecto

**oCabra** es un servidor de modelos de IA multi-GPU compatible con las APIs de
OpenAI (`/v1/*`) y Ollama (`/api/*`), con API de administración propia
(`/ocabra/*`), autenticación propia (JWT + API keys, grupos de acceso a
modelos), carga/descarga dinámica con políticas `pin`/`warm`/`on_demand`,
federación entre nodos y backends instalables en caliente. Hardware de
producción: RTX 3060 (12 GB) + RTX 3090 (24 GB).

## Documentación

- `docs/ROADMAP.md` — backlog y orden de ejecución (**fuente de verdad del trabajo pendiente**).
- `docs/PLAN.md` — arquitectura y fases (histórico en parte).
- `docs/CONTRACTS.md` — interfaces entre módulos; si cambias uno, actualízalo.
- `docs/CONVENTIONS.md` — estilo y naming.
- `docs/tasks/` — planes en curso; `docs/archive/` — planes ya entregados (solo consulta).
- `/docs`, `/docs/changelog`, `/docs/models` — documentación que sirve el propio servidor.

## Reglas de trabajo

- **Producción corre del checkout `/docker/ocabra`** y puede haber otras sesiones
  con cambios sin commitear: trabaja en un worktree propio, haz `git add` solo de
  tus ficheros y construye imágenes siempre desde un worktree limpio.
- No hay ownership por "streams" ni rama `dev`: el reparto lo decide el usuario
  en cada tarea. No toques ficheros que te haya dicho que están en obras.
- Commits: `<tipo>(<scope>): <descripción>` (`feat`, `fix`, `refactor`, `test`,
  `docs`, `chore`, `perf`), en español como el historial, uno por unidad lógica.
- Tests en `backend/tests/` o `frontend/src/__tests__/`; no des por terminado
  nada con tests en rojo.
- Variables de entorno nuevas: en `.env.example` con comentario; en el código,
  siempre `settings.<campo>` (pydantic-settings).
