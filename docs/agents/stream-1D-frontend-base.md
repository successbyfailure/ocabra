# Briefing: Stream 1-D — Frontend Base

**Prerequisito: Fase 0 completada.**
**Rama:** `feat/1-D-frontend-base`

## Objetivo

Construir la infraestructura del frontend: cliente API tipado, stores Zustand,
hook de WebSocket, y las páginas Dashboard y Layout completos. Las otras páginas
(Models, Explore, Playground, Stats, Settings) pueden usar mocks hasta que los
streams 3-A/3-B estén listos.

## Ficheros propios

```
frontend/src/api/client.ts
frontend/src/api/types.ts
frontend/src/stores/gpuStore.ts
frontend/src/stores/modelStore.ts
frontend/src/stores/downloadStore.ts
frontend/src/hooks/useWebSocket.ts
frontend/src/hooks/useSSE.ts
frontend/src/components/layout/Layout.tsx
frontend/src/components/layout/Sidebar.tsx
frontend/src/components/layout/Header.tsx
frontend/src/components/gpu/GpuCard.tsx
frontend/src/components/gpu/PowerGauge.tsx
frontend/src/components/gpu/VramBar.tsx
frontend/src/components/models/ModelStatusBadge.tsx
frontend/src/components/models/LoadPolicyBadge.tsx
frontend/src/components/common/CapabilityBadge.tsx
frontend/src/pages/Dashboard.tsx    ← implementación completa
frontend/src/pages/Models.tsx       ← stub con lista básica
frontend/src/pages/Explore.tsx      ← stub
frontend/src/pages/Playground.tsx   ← stub
frontend/src/pages/Stats.tsx        ← stub
frontend/src/pages/Settings.tsx     ← stub
```

## Tipos base (api/types.ts)

Espeja exactamente los contratos de `docs/CONTRACTS.md`:

```typescript
export interface GPUState {
  index: number
  name: string
  totalVramMb: number
  freeVramMb: number
  usedVramMb: number
  utilizationPct: number
  temperatureC: number
  powerDrawW: number
  powerLimitW: number
  lockedVramMb: number
}

export type ModelStatus = "discovered"|"configured"|"loading"|"loaded"|"unloading"|"unloaded"|"error"
export type LoadPolicy = "pin"|"warm"|"on_demand"
export type BackendType = "vllm"|"diffusers"|"whisper"|"tts"

export interface ModelCapabilities {
  chat: boolean
  completion: boolean
  tools: boolean
  vision: boolean
  embeddings: boolean
  reasoning: boolean
  imageGeneration: boolean
  audioTranscription: boolean
  tts: boolean
  streaming: boolean
  contextLength: number
}

export interface ModelState {
  modelId: string
  displayName: string
  backendType: BackendType
  status: ModelStatus
  loadPolicy: LoadPolicy
  autoReload: boolean
  preferredGpu: number | null
  currentGpu: number[]
  vramUsedMb: number
  capabilities: ModelCapabilities
  lastRequestAt: string | null
  loadedAt: string | null
}

export interface DownloadJob {
  jobId: string
  source: "huggingface" | "ollama"
  modelRef: string
  status: "queued"|"downloading"|"completed"|"failed"|"cancelled"
  progressPct: number
  speedMbS: number | null
  etaSeconds: number | null
  error: string | null
  startedAt: string
  completedAt: string | null
}

// WebSocket events
export type WSEvent =
  | { type: "gpu_stats"; data: GPUState[] }
  | { type: "model_event"; data: { event: string; modelId: string; status: ModelStatus } }
  | { type: "download_progress"; data: { jobId: string; pct: number; speedMbS: number } }
  | { type: "system_alert"; data: { level: "warn"|"error"; message: string } }
```

## API Client (api/client.ts)

```typescript
// Todas las llamadas a /ocabra/* (API interna)
const api = {
  gpus: {
    list: (): Promise<GPUState[]>
    get: (index: number): Promise<GPUState>
  },
  models: {
    list: (): Promise<ModelState[]>
    get: (modelId: string): Promise<ModelState>
    load: (modelId: string): Promise<ModelState>
    unload: (modelId: string): Promise<void>
    patch: (modelId: string, patch: Partial<ModelState>): Promise<ModelState>
    delete: (modelId: string): Promise<void>
  },
  downloads: {
    list: (): Promise<DownloadJob[]>
    enqueue: (source: string, modelRef: string): Promise<DownloadJob>
    cancel: (jobId: string): Promise<void>
    streamProgress: (jobId: string): EventSource   // SSE
  },
  registry: {
    searchHF: (q: string, task?: string): Promise<HFModelCard[]>
    getHFDetail: (repoId: string): Promise<HFModelDetail>
    searchOllama: (q: string): Promise<OllamaModelCard[]>
    listLocal: (): Promise<LocalModel[]>
  },
  stats: {
    requests: (params: StatsParams): Promise<RequestStats>
    energy: (params: StatsParams): Promise<EnergyStats>
    performance: (modelId?: string): Promise<PerformanceStats>
  },
  config: {
    get: (): Promise<ServerConfig>
    patch: (patch: Partial<ServerConfig>): Promise<ServerConfig>
    syncLiteLLM: (): Promise<{ syncedModels: number }>
  }
}
```

## WebSocket Hook

```typescript
// hooks/useWebSocket.ts
export const useWebSocket = () => {
  // Conecta a ws://host/ocabra/ws
  // Parsea eventos y los despacha al store correspondiente
  // Auto-reconexión con backoff exponencial
  // Retorna: { connected: boolean, lastEvent: WSEvent | null }
}
```

## Stores Zustand

```typescript
// gpuStore: actualizado por WS events "gpu_stats"
interface GPUStore {
  gpus: GPUState[]
  setGpus: (gpus: GPUState[]) => void
  lastUpdated: Date | null
}

// modelStore: actualizado por WS events "model_event" + polling inicial
interface ModelStore {
  models: Record<string, ModelState>
  setModels: (models: ModelState[]) => void
  updateModel: (modelId: string, patch: Partial<ModelState>) => void
  loadModel: (modelId: string) => Promise<void>
  unloadModel: (modelId: string) => Promise<void>
}

// downloadStore: actualizado por WS events "download_progress"
interface DownloadStore {
  jobs: DownloadJob[]
  addJob: (job: DownloadJob) => void
  updateJob: (jobId: string, patch: Partial<DownloadJob>) => void
}
```

## Dashboard — implementación completa

El Dashboard es la única página totalmente implementada en este stream:

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│ GPU Cards (2 cards, una por GPU)                        │
│  - Nombre, VRAM bar, utilización, temperatura, potencia │
│  - PowerGauge: arco visual de W actuales / W límite     │
├─────────────────────────────────────────────────────────┤
│ Modelos activos (solo los LOADED)                       │
│  - ModelStatusBadge + LoadPolicyBadge + GPU chip        │
│  - Botón unload                                         │
├─────────────────────────────────────────────────────────┤
│ Descargas activas (si las hay)                          │
│  - Progress bar animada + velocidad + ETA               │
└─────────────────────────────────────────────────────────┘
```

Todos los datos son en tiempo real vía WebSocket.

## Diseño visual

- Dark mode por defecto (Tailwind `dark:` classes)
- Sidebar colapsable en móvil
- Colores de estado:
  - `loaded` → verde
  - `loading` / `unloading` → amarillo (animado)
  - `error` → rojo
  - `configured` → gris
- GPU utilización alta (>80%) → badge rojo parpadeante
- Temperatura alta (>80°C) → texto en naranja

## Tests requeridos

- Test de useWebSocket: mock de WebSocket, verificar que eventos actualizan stores
- Test de GpuCard: render con datos mock, snapshot
- Test de Dashboard: carga inicial de datos, actualización en tiempo real

## Estado

- [ ] En progreso
- [x] Completado
