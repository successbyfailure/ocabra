# Briefing: Stream 4 — Frontend Features (4 sub-streams paralelos)

**Prerequisito: Streams 3-A y 3-B completados (o mocks disponibles). Stream 1-D completado.**

Los 4 sub-streams pueden trabajar en paralelo en sus páginas respectivas.

---

## Stream 4-A — Models UI & Explore

**Rama:** `feat/4-A-models-ui`

### Ficheros propios
```
frontend/src/pages/Models.tsx
frontend/src/pages/Explore.tsx
frontend/src/components/models/ModelCard.tsx
frontend/src/components/models/ModelConfigModal.tsx
frontend/src/components/models/ScheduleEditor.tsx
frontend/src/components/explore/HFModelCard.tsx
frontend/src/components/explore/OllamaModelCard.tsx
frontend/src/components/explore/SearchFilters.tsx
frontend/src/components/downloads/DownloadQueue.tsx
```

### Models page
- Lista de todos los modelos (cualquier status)
- Columnas: nombre, tipo, policy badge, GPU, VRAM, status badge, actions
- Acciones por modelo: load/unload, pin/unpin, configurar, eliminar
- Filtros: por status, por tipo (LLM/imagen/audio), por GPU
- Modal de configuración: load_policy, preferred_gpu, auto_reload, schedules

### ScheduleEditor component
- UI para añadir/editar/borrar ventanas horarias de evicción
- Selector de días de semana + hora inicio/fin
- Preview: "Los lunes, miércoles y viernes de 02:00 a 06:00 este modelo se descargará automáticamente"

### Explore page
- Dos tabs: HuggingFace | Ollama
- Barra de búsqueda con debounce 300ms
- Filtros HF: task (text-generation, image-generation, etc.), tamaño, gated
- Cards con: nombre, descripción, downloads, VRAM estimada, backend sugerido, botón Instalar
- Al hacer clic en Instalar: abre modal con opciones (carpeta destino, load_policy) y lanza descarga
- DownloadQueue flotante (bottom-right) con progreso en tiempo real via SSE

---

## Stream 4-B — Playground

**Rama:** `feat/4-B-playground`

### Ficheros propios
```
frontend/src/pages/Playground.tsx
frontend/src/components/playground/ChatInterface.tsx
frontend/src/components/playground/ImageInterface.tsx
frontend/src/components/playground/AudioInterface.tsx
frontend/src/components/playground/ModelSelector.tsx
frontend/src/components/playground/ParamsPanel.tsx
frontend/src/components/playground/ToolCallRenderer.tsx
frontend/src/components/playground/MessageBubble.tsx
```

### Layout Playground
```
┌─────────────────────────────────────────────────────────┐
│ ModelSelector (dropdown con capability badges)          │
├──────────────────────────────┬──────────────────────────┤
│                              │ ParamsPanel               │
│  Interface según capability  │  - temperature            │
│  (Chat / Image / Audio)      │  - max_tokens             │
│                              │  - top_p                  │
│                              │  - system prompt          │
│                              │  - response_format        │
└──────────────────────────────┴──────────────────────────┘
```

### ChatInterface
- Markdown rendering (react-markdown + syntax highlighting)
- Streaming con animación de cursor
- Tool calls: renderizado especial con nombre de tool + args + resultado
- Vision: drag & drop de imagen en el chat
- Botón "Copy as OpenAI API call" (muestra el curl/código equivalente)

### ImageInterface
- Prompt + negative prompt
- Sliders: steps, guidance, width/height (presets de tamaño)
- Seed (con botón random)
- Galería de resultados con botón de descarga

### AudioInterface
- Transcripción: recorder de audio (MediaRecorder API) o upload de fichero
- Mostrar transcripción con timestamps si response_format=verbose_json
- TTS: textarea de texto, selector de voz, slider de velocidad, reproductor de audio

---

## Stream 4-C — Stats UI

**Rama:** `feat/4-C-stats`

### Ficheros propios
```
frontend/src/pages/Stats.tsx
frontend/src/components/stats/RequestsChart.tsx
frontend/src/components/stats/TokensChart.tsx
frontend/src/components/stats/EnergyPanel.tsx
frontend/src/components/stats/PerformanceTable.tsx
frontend/src/components/stats/DateRangePicker.tsx
```

### Layout Stats
- DateRangePicker: últimas 1h, 24h, 7d, 30d, o rango custom
- Selector de modelo (o "todos")

### Gráficos (Recharts)
- Requests/minuto: LineChart en tiempo real
- Tokens de entrada/salida: BarChart apilado
- Latencia P50/P95/P99: LineChart
- Tokens/segundo por modelo: BarChart horizontal

### EnergyPanel
```
┌─────────────────────────────┐
│ GPU 0 (3060)                │
│  Consumo actual: 45 W       │
│  kWh esta sesión: 0.23      │
│  Coste estimado: €0.03      │
├─────────────────────────────┤
│ GPU 1 (3090)                │
│  Consumo actual: 180 W      │
│  kWh esta sesión: 0.87      │
│  Coste estimado: €0.13      │
└─────────────────────────────┘
```

### PerformanceTable
- Tabla por modelo: total requests, avg latencia, tokens/s, errores, uptime %
- Ordenable por columna
- Exportable como CSV

---

## Stream 4-D — Settings UI

**Rama:** `feat/4-D-settings`

### Ficheros propios
```
frontend/src/pages/Settings.tsx
frontend/src/components/settings/GeneralSettings.tsx
frontend/src/components/settings/GPUSettings.tsx
frontend/src/components/settings/LiteLLMSettings.tsx
frontend/src/components/settings/StorageSettings.tsx
frontend/src/components/settings/GlobalSchedules.tsx
```

### Secciones

**General**
- Carpeta de modelos (MODELS_DIR)
- Log level
- Idle timeout por defecto (para on_demand)
- Buffer de VRAM reservada (MB)

**GPU**
- GPU preferida por defecto (select: GPU 0 / GPU 1)
- Umbral de presión de VRAM (% para trigger evicción)
- Temperatura máxima de alerta

**LiteLLM Sync**
- URL del proxy LiteLLM
- API key de admin
- Toggle: sync automático al añadir/eliminar modelos
- Botón: sync manual → muestra resultado ("12 modelos sincronizados")
- Estado: última sync (timestamp + éxito/error)

**Storage**
- Uso de disco por modelo (lista con barras)
- Botón limpiar caché HuggingFace
- Carpeta de descarga de modelos

**Schedules globales**
- ScheduleEditor para schedules que afectan a todos los modelos
- Precedencia: schedule de modelo individual > schedule global

---

## Convenciones comunes para todos los streams 4-*

- Usar el API client de `src/api/client.ts`, nunca fetch directo
- Todos los estados de loading/error deben mostrar feedback visual (skeleton, toast)
- Usar `sonner` o el componente Toast de shadcn para notificaciones
- Confirmar acciones destructivas (eliminar modelo, cancelar descarga) con Dialog
- Los datos se refrescan automáticamente cada 30s + por WS events cuando aplica

## Estado

- [x] 4-A completado
- [x] 4-B completado
- [x] 4-C completado
- [x] 4-D completado
