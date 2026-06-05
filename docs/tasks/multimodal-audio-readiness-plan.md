# Plan: Multimodal Audio Readiness — oCabra

**Fecha:** 2026-06-06
**Estado:** Activo
**Origen:** auditoría tras validar audio nativo de Gemma 4 (E4B) en Ollama vía `/v1/chat/completions` con `type: "input_audio"`.
**Objetivo:** Que oCabra y su frontend puedan explotar plenamente modelos con entrada nativa de audio (Gemma 4, Gemma 3n, Qwen2.5-Omni…) tanto en chat REST como en modo Realtime.

---

## Resumen de la auditoría

| Capa | Estado | Detalle |
|---|---|---|
| Backend REST `/v1/chat/completions` | ✅ Listo | Detecta y pasa verbatim `input_audio`, gating por capability, schemas permissivos, federación verbatim. |
| Backend Realtime WebSocket | ❌ Roto para multimodal nativo | `RealtimeSession` siempre Whisper→texto→LLM; ignora `audio_input` aunque el LLM lo declare. |
| Frontend tipos | ✅ | `ModelCapabilities.audioInput` + mapper en `client.ts`. |
| Frontend `ModelCard` | ⚠️ | Badge `audioInput` definido pero **no renderizado**. |
| Frontend `ChatInterface` | ❌ | Solo construye `text` + `image_url`; sin subida/grabación de audio. |
| Frontend `ModelConfigModal` | ⚠️ | `extra_config.limit_mm_per_prompt` solo editable como JSON crudo. |
| Frontend UI Realtime | ❌ | Sin página/modal, sin grabador, sin selector de LLM. |

---

## Tareas

### T1 — Renderizar badge `audioInput`/`videoInput` en `ModelCard`

**Stream:** 1D Frontend.
**Tamaño:** S (1-2 h).
**Archivos:**
- `frontend/src/components/models/ModelCard.tsx` — añadir los badges junto a los demás `CapabilityBadge`.
- `frontend/src/components/common/CapabilityBadge.tsx` — verificar que ya cubre `audioInput`/`videoInput` (parece que sí).

**Aceptación:**
- Cualquier modelo con `capabilities.audioInput=true` muestra el chip "Audio in" en el listado.
- Idem `videoInput` → chip "Video in".
- `npm run lint` limpio.

---

### T2 — Campo dedicado para `limit_mm_per_prompt` en `ModelConfigModal`

**Stream:** 1D Frontend.
**Tamaño:** S (2-3 h).
**Archivos:**
- `frontend/src/components/models/ModelConfigModal.tsx` — añadir, en la sección de vLLM, tres inputs numéricos `image`, `audio`, `video` cuyo conjunto se serializa como `extra_config.limit_mm_per_prompt`.
- Solo visible cuando el modelo es vLLM (o cuando alguna capability multimodal está activa).

**Aceptación:**
- Al guardar, `extra_config` contiene `{"limit_mm_per_prompt": {"image": N, "audio": M, "video": K}}` solo si al menos uno > 0.
- Valor vacío en los tres → la clave se omite (no rompe modelos existentes).
- Tooltip/help text explicando para qué sirve.

---

### T3 — Entrada de audio en `ChatInterface` (Playground)

**Stream:** 1D Frontend.
**Tamaño:** M (medio día - 1 día).
**Archivos:**
- `frontend/src/components/playground/ChatInterface.tsx` — drag-drop + botón "🎤 grabar" + adjuntar archivo de audio.
- `frontend/src/components/playground/ChatInterface.tsx` `buildOpenAIMessages()` — emitir `{type:"input_audio", input_audio:{data:base64, format:"wav"|"mp3"}}` además de `text`.
- Mostrar chip de audio adjunto en el mensaje (icono + nombre + duración).
- Validar contra `model.capabilities.audioInput` antes de enviar: si el modelo no lo soporta, mostrar warning bloqueante.

**Aceptación:**
- Usuario puede adjuntar `.wav`/`.mp3`/`.m4a` por drag-drop o file picker.
- Usuario puede grabar audio del micro (MediaRecorder API, formato webm/opus convertido o aceptado tal cual).
- El mensaje llega al backend como `input_audio` y el modelo (Gemma 4 E4B en Ollama, ya disponible) responde correctamente.
- Si el modelo seleccionado no tiene `audioInput`, el botón está deshabilitado con tooltip explicativo.

---

### T4 — `RealtimeSession`: bypass STT para LLMs con audio nativo

**Stream:** 3A OpenAI API (backend).
**Tamaño:** L (1-2 días).
**Archivos:**
- `backend/ocabra/core/realtime_session.py` — detectar `audio_input=True` en el LLM seleccionado; saltar la llamada a Whisper y pasar el buffer de audio como `input_audio` al LLM directamente.
- `backend/ocabra/api/openai/realtime.py` — opcional: nueva propiedad en `session.update` para forzar bypass o auto.
- Mantener el camino actual STT→LLM→TTS como fallback para LLMs sin `audio_input`.
- Sin romper tests existentes.

**Aceptación:**
- Con LLM = `ollama/gemma4:e4b-it-q8_0` (capability audio_input=true), el `RealtimeSession` envía el audio directo al LLM, sin Whisper. Latencia esperada: menor que la pipeline actual.
- Con LLM = `ollama/qwen3:8b` (sin audio_input), el pipeline sigue siendo Whisper→texto→LLM como antes.
- Log debug en cada turno indica qué ruta tomó.
- Cobertura de tests si los hay para `RealtimeSession`; añadir uno mínimo para la nueva rama.

**Salida documentable (para T5):** contrato de `session.update` (campos nuevos si los hubiera) y formato exacto del WebSocket mensaje que el cliente debe enviar para audio nativo.

---

### T5 — UI Realtime en Playground

**Stream:** 1D Frontend.
**Tamaño:** L (1-2 días). Depende del contrato producido por T4.
**Archivos:**
- `frontend/src/pages/Playground.tsx` o componente nuevo — tab "Realtime".
- Grabador continuo (MediaRecorder + VAD client-side opcional).
- Selector de LLM (filtrar por audio-capable opcionalmente).
- Cliente WebSocket contra `/v1/realtime?model=…`, envío de eventos `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `response.create`.
- Visualización de transcripción + respuesta + reproducción TTS.

**Aceptación:**
- Sesión Realtime funcional contra modelo audio-nativo (T4) y contra pipeline clásico (fallback).
- Botón Push-to-Talk + modo VAD continuo.
- Indicadores de estado: conectado / grabando / pensando / hablando.
- Errors visibles si el modelo se cae.

---

## Orden de ejecución

```
T1 ─┐
T2 ─┼─ paralelo (worktrees aislados)
T3 ─┤
T4 ─┘
        └─→ T5 (consume contrato de T4)
```

## Deudas conocidas (no entran en este plan)

- vLLM 12B Unified audio no funciona hoy en upstream (esperar tag `gemma4-unified` de la imagen). Ver `[[gemma4-audio-asr]]` en memoria.
- Bug cold-start `gemma4:12b` en Ollama (primer fwd pass tras carga dice "no audio").
- Modo thinking de Gemma 4 12B muy verboso por defecto.

## Pruebas pendientes (no entran en este plan)

- Multilingüe español con Gemma 4 E4B/12B.
- `nothink` mode en Gemma 4.
- Comparativa de latencias E4B vs Whisper-large-v3 en español.
