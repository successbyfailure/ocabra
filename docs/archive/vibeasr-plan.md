# Backend VibeASR — STT ternario (microsoft/VibeVoice-ASR-BitNet)

Estado: **implementado + validado end-to-end en CPU** (transcripción real correcta).

## Qué es

STT ternario 1.58-bit de Microsoft: VAE encoder (`i8_s`) + LM decoder (`i2_s` + embeddings `q6_k`),
~1.7 GB, **CPU-first, tiempo real** (RTF ~1). Distinto de faster-whisper: es otro pipeline/runtime.
Runtime: binario nativo de [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) —
**`asr_stream_server`**, servidor persistente por **stdin/stdout** (no HTTP):

    stdin  : una ruta de audio por línea ("EXIT" para terminar)
    stdout : "---READY---" al cargar; luego tokens (uno/línea) + "---END---" por petición

## Arquitectura en oCabra

Como el endpoint `/v1/audio/transcriptions` hace POST multipart a `http://127.0.0.1:<port>/transcribe`,
el backend nuevo `vibeasr` (modalidad `AUDIO_TRANSCRIPTION`) lanza un **worker shim FastAPI**
que envuelve el servidor persistente:

- `backend/ocabra/backends/vibeasr_backend.py` — `VibeAsrBackend`: resuelve binario (`asr_stream_server`),
  los 2 GGUF (VAE `i8_s` / LM `i2_s`) del dir del modelo, y lanza el worker. `install_spec` = build nativo
  de VibeASR.cpp (git + `install_vibeasr.sh`, `extra_bins.stream_server`), `include_core_runtime=True` (FastAPI).
- `backend/workers/vibeasr_worker.py` — arranca `asr_stream_server` una vez (modelos cargados), espera `---READY---`,
  y por petición: transcodifica a **24 kHz mono WAV con ffmpeg**, escribe la ruta por stdin, lee hasta `---END---`,
  une los tokens. `/health` + `/transcribe` (+ `/info`). Serializa con un `asyncio.Lock`.
- Config (`config.py`): `vibeasr_stream_server_bin`, `vibeasr_python_bin`, `vibeasr_threads` (8), `vibeasr_startup_timeout_s`.
- Registro en `main.py` (`register_backend("vibeasr", ...)`), detección en `local_scanner._detect_hf_backend`
  (`vibeasr`/`vibevoice-asr` por nombre o `model_type`), y `"vibeasr"` añadido al `Literal` de `schemas/registry.py`.
- `backend/scripts/install_vibeasr.sh` — build CPU (cmake) de `asr_stream_server` + `asr_infer`, copia libs con symlinks.

## Validación real (2026-08-01, en `ocabra-api-1`, CPU)

- Modelo descargado a `/data/models/microsoft--VibeVoice-ASR-BitNet/` (VAE 703 MB + LM 993 MB).
- VibeASR.cpp compilado (CPU, sin CUDA) → `asr_stream_server` + `asr_infer`.
- Worker real (mi código) probado con `jfk.wav` (11 s de voz humana, vía HTTP `/transcribe`):
  - `/health` OK tras ~5 s (carga de modelos).
  - **json** → `{"text":"And so, my fellow american, ask not what your country can do for you. Ask what you can do for your country."}` ✅
  - **text** → mismo texto en `PlainTextResponse`. 2ª petición reusó el servidor persistente (sin recargar). RTF ~1.04.
  - El transcode ffmpeg (a 24 kHz mono) funcionó con el wav original.

## Ventaja frente a Bonsai (build)

A diferencia del fork CUDA de Bonsai, VibeASR es **CPU puro (ggml)** → la ruta de instalación modular
(`install_vibeasr.sh` dentro del contenedor api) **sí compila** (no necesita headers cuBLAS). No compite por VRAM.

## Deudas

- **Desplegar en `ocabra-api-1`**: la validación fue con el worker copiado a mano; para enrutado automático por la API
  de oCabra hay que desplegar esta rama (el worker vive en la imagen, no está bind-mounted).
- **ffmpeg en la imagen**: el worker depende de ffmpeg para transcodificar; el `install_spec` ya lo pide en `apt_packages`,
  pero la imagen OCI/runtime debe incluirlo. `Dockerfile.vibeasr` pendiente (mirror de `Dockerfile.bitnet`).
- **Idiomas — español NO soportado**: la ficha lista EN/ZH/FR/IT/KO/PT/VI (español NO nombrado). Validado (2026-08-01):
  inglés (jfk, voz humana) perfecto; francés casi perfecto incluso con voz robótica espeak; **español el peor, tira a portugués**.
  → Para **STT en español usar faster-whisper** (ya en oCabra). VibeASR = opción CPU-barata para SUS idiomas. Decisión de enrutado por idioma.
  (Nota: espeak es síntesis robótica y degrada todos los idiomas; el inglés con voz humana real salió perfecto.)
- **Streaming de tokens**: el servidor emite tokens incrementales; ahora se agregan y se devuelven de una vez.
  Se podría exponer streaming real en el futuro.

## Relación

Encaja con el [plan de voz](voice-pipeline-plan.md) Fase 1 (STT CPU, libera GPU para LLM/TTS).
Ternario hermano de [Bonsai](bonsai-ternary-plan.md).
