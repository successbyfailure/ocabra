# AGENTS.md — Guía de colaboración multi-agente

Este fichero es leído por cualquier agente de IA (Claude, GPT-4o, Gemini, etc.)
que trabaje en este repositorio.

## Proyecto

**oCabra** es un servidor de modelos de IA multi-GPU con las siguientes características:
- Compatible con APIs OpenAI (`/v1/*`) y Ollama (`/api/*`)
- Gestión dinámica de carga/descarga de modelos con políticas `pin`, `warm`, `on_demand`
- Soporte para LLM, imagen, audio, TTS y modelos multimodal
- Hardware objetivo: RTX 3060 (12 GB) + RTX 3090 (24 GB)
- LiteLLM Proxy actúa como capa externa de autenticación

**Documentación esencial:**
- `docs/PLAN.md` — Plan completo, arquitectura, fases
- `docs/CONTRACTS.md` — Interfaces entre módulos (LEE ESTO ANTES DE IMPLEMENTAR)
- `docs/CONVENTIONS.md` — Estilo de código y naming
- `docs/agents/` — Briefing específico de cada stream de trabajo

## Reglas de colaboración multi-agente

### 1. Ownership de streams

Cada agente trabaja en un stream asignado. **No modifiques ficheros de otro stream
sin coordinación explícita.** Los límites están definidos en `docs/agents/`.

| Stream | Módulos propios |
|--------|----------------|
| **Fase 0** | Toda la fundación (un agente secuencial) |
| **1-A GPU** | `backend/ocabra/core/gpu_manager.py`, `scheduler.py` |
| **1-B Model** | `backend/ocabra/core/model_manager.py`, `worker_pool.py` |
| **1-C Registry** | `backend/ocabra/registry/`, `backend/ocabra/api/internal/downloads.py` |
| **1-D Frontend base** | `frontend/src/` estructura base, stores, API client |
| **2-A vLLM** | `backend/ocabra/backends/vllm_backend.py`, `workers/vllm_worker.py` |
| **2-B Diffusers** | `backend/ocabra/backends/diffusers_backend.py`, `workers/diffusers_worker.py` |
| **2-C Audio** | `backend/ocabra/backends/whisper_backend.py`, `tts_backend.py`, `workers/` |
| **3-A OpenAI API** | `backend/ocabra/api/openai/` |
| **3-B Ollama API** | `backend/ocabra/api/ollama/` |
| **4-A Models UI** | `frontend/src/pages/Models.tsx`, `Explore.tsx` y sus componentes |
| **4-B Playground** | `frontend/src/pages/Playground.tsx` y sus componentes |
| **4-C Stats UI** | `frontend/src/pages/Stats.tsx` y sus componentes |
| **4-D Settings UI** | `frontend/src/pages/Settings.tsx` y sus componentes |
| **5 Integrations** | `backend/ocabra/integrations/`, `stats/`, polish general |

### 2. Ficheros compartidos — protocolo de modificación

Los siguientes ficheros son compartidos y requieren cuidado especial:

- `backend/ocabra/main.py` — Solo añade routers en la sección marcada con `# ROUTERS`
- `backend/ocabra/db/` — Añade modelos nuevos en ficheros propios, no modifiques los ajenos
- `docker-compose.yml` — No modificar sin consenso
- `frontend/src/App.tsx` — Solo añade rutas en la sección marcada con `# ROUTES`
- `frontend/src/api/client.ts` — Añade métodos, no cambies la estructura base

### 3. Contratos de interfaz

Antes de implementar cualquier interfaz entre módulos, verifica `docs/CONTRACTS.md`.
Si necesitas cambiar un contrato existente, documenta el cambio en ese fichero primero.

### 4. Commits

- Prefijo de commit según tipo: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Incluye el stream en el scope: `feat(1-A): add GPU pressure detection`
- Un commit por unidad lógica de trabajo, no commits masivos

### 5. Tests

- Cada módulo nuevo debe tener tests en `backend/tests/` o `frontend/src/__tests__/`
- Nomenclatura: `test_<módulo>.py` o `<Componente>.test.tsx`
- No marques trabajo como completo si los tests fallan

### 6. Variables de entorno

Todas las variables de entorno nuevas deben añadirse a `.env.example` con comentario explicativo.
Nunca escribas valores reales en el código; usa siempre `settings.<variable>` (pydantic-settings).

## Dependencias entre streams (no inicies un stream sin que el anterior esté listo)

```
Fase 0  (fundación completa)
  ├─→ 1-A (GPU Manager)      ─┐
  ├─→ 1-B (Model Manager)    ─┤─→ 2-A (vLLM)     ─┐
  ├─→ 1-C (Registry)         ─┤─→ 2-B (Diffusers) ─┤─→ 3-A (OpenAI API) ─┐
  └─→ 1-D (Frontend base)    ─┘─→ 2-C (Audio)     ─┘─→ 3-B (Ollama API) ─┤─→ 4-A,B,C,D ─→ 5
                                                                            └──────────────────┘
```

1-D puede empezar en paralelo con mocks antes de que 3-A/3-B estén listos.

## Estado del proyecto

Ver el estado actualizado de cada stream en `docs/agents/<stream>.md`.
