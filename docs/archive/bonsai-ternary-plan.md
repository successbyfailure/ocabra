# Bonsai 27B en el backend ternario (`bitnet`)

Estado: **implementado + validado end-to-end en GPU** (fork compilado, modelo descargado, generación real correcta).

## Validación real (2026-07-31)

- Modelo `Bonsai-27B-Q1_0.gguf` (3.8 GB) descargado a `/data/models/prism-ml--Bonsai-27B-gguf/`.
- Fork `PrismML-Eng/llama.cpp` compilado con **CUDA 12.4 en imagen devel** (sm_86) → `bonsai-server`.
- Ejecutado en RTX 3090 (`CUDA_VISIBLE_DEVICES=1`, `--n-gpu-layers 99 --ctx-size 4096`):
  - Carga del 27B en **~14 s**, **~4.7 GB VRAM**.
  - Generación **72 t/s** (prompt eval 128 t/s).
  - Modelo de **razonamiento** (`thinking=1`, chat format `peg-native`): emite `reasoning_content` + `content`.
    Prueba "17×4 + saludo" → reasoning correcto (68) y `content = "Hola, 68."`, `finish_reason=stop`. ✅
- Detección de oCabra validada contra los ficheros reales (working tree del host): `Bonsai-27B-Q1_0.gguf`
  → `backend_type=bitnet`, `_is_prismml_model=True`; `dspark` y **`mmproj` (projector de visión)** excluidos.

## Contexto

El backend `bitnet` servía solo GGUFs de **microsoft/BitNet** (kernels `i2_s` / TL1/TL2, CPU-first).
**Bonsai 27B** (PrismML, extrema-cuantización de Qwen3.6 27B) usa un layout distinto —
`Q1_0_g128` con kernels *hybrid-attention*— que **solo** compila el fork
[`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) (CUDA/Metal).
Los ficheros `*dspark*` del repo NO son el modelo: son *drafters* de speculative decoding.

Modelos de referencia:
- `prism-ml/Bonsai-27B-gguf` (1-bit, ~3.8 GB `Q1_0`)
- `prism-ml/Ternary-Bonsai-27B-gguf` (1.58-bit ternario, ~6.7 GB)

## Qué se ha hecho (avances)

Se **extendió** el backend `bitnet` (decisión: no crear backend nuevo; ambos son `llama-server` HTTP):

- **`backends/bitnet_backend.py`**
  - `_is_prismml_model(path)`: detecta Bonsai/PrismML por nombre (`bonsai` / `q1_0` / `prismml`).
  - Selección de binario por modelo: `_select_server_bin()` → `bonsai-server` (PrismML) para Bonsai,
    `bitnet-server` (Microsoft) para el resto. Resolución del binario PrismML:
    metadata modular (`prismml_server`) → sibling `prismml/bonsai-server` → `settings.bitnet_prismml_server_bin`.
    Si es Bonsai y no hay binario PrismML, `load()` falla con mensaje claro.
  - Default **GPU-first** para Bonsai: `gpu_layers=99` (`_PRISMML_DEFAULT_GPU_LAYERS`) salvo override.
  - `get_capabilities()`: `tools=reasoning=True` para Bonsai (modelo agéntico/razonador).
  - `LD_LIBRARY_PATH` usa el dir del binario elegido (el fork vive en `prismml/` con sus libs para evitar choques de ABI).
- **`config.py`**: `bitnet_prismml_server_bin` (default `/usr/local/bin/bonsai-server`).
- **Detección** (`registry/bitnet_registry.py`, `registry/local_scanner.py`):
  reconocen `bonsai` / `ternary` / `q1_0` y **excluyen `dspark`**. `suggested_backend="bitnet"`.
- **Build del fork** (best-effort, controlado por `BITNET_BUILD_PRISMML=true`, default on):
  - `backend/scripts/install_bitnet.sh` (ruta modular): clona+compila PrismML → `${BIN_DIR}/prismml/bonsai-server`.
  - `backend/scripts/build_bitnet.sh` + `backends/dockerfiles/Dockerfile.bitnet` (imagen OCI): paridad → `/backend/bin/prismml/`.
- **Tests**: `tests/test_bitnet_backend.py` (selección de binario, gpu_layers, error sin fork),
  `tests/test_bitnet_registry.py` (card Bonsai, exclusión de dspark). 15 verdes; ruff check + format OK.

## Rendimiento y contexto (medido en 3090)

- **Paralelismo** (`--parallel 8`, continuous batching): throughput agregado escala ~2× con rendimientos
  decrecientes — 1→65 t/s, 2→97, 4→113, 8→126 t/s; latencia/petición sube casi lineal (3s→12.7s).
  Punto dulce ~`parallel=4`.
- **CPU vs GPU**: en CPU (16 hilos) ~5.3 t/s vs 72 t/s en la 3090 (~14×). Bonsai corre en CPU pero su camino
  rápido es GPU/Metal (los kernels `Q1_0_g128` son CUDA/Metal). BitNet b1.58 / Falcon-Edge sí son CPU-first.
- **Contexto**: nativo **262.144 (256K)**. Footprint fijo ~3.9 GB; KV fp16 ~65 KB/token (~16.4 GB para 256K),
  KV q8_0 ~39 KB/token. Escalera en 3090:
  - fp16: OK hasta 96K (~18.7 GB usados con el resto de la tarjeta ocupado).
  - **q8_0 + `--flash-attn on`: los 256K completos entran en la 3090 (~22.4 GB)**. Opciones nuevas del backend:
    `bitnet_cache_type_k/v` (fuerzan flash-attn).
- **Evicción**: la escalera GPU se disparó sola cuando el scheduler liberó la 3090 (idle) — de 1 GB a 15.6 GB libres.

## Hallazgos de despliegue (importantes)

- **Sintaxis de flags divergente entre binarios**: el fork PrismML (llama.cpp reciente) exige `--flash-attn on`
  (valor explícito on/off/auto); el binario Microsoft usa `--flash-attn` booleano. El backend ramifica por
  `is_prismml`. Ojo con otras flags si se actualizan los forks.

- **El build CUDA necesita imagen devel**: el contenedor `ocabra-api-1` (runtime) tiene `nvcc` pero **no los headers de cuBLAS**,
  así que la ruta modular `install_bitnet.sh` NO puede compilar el fork CUDA dentro del api. La vía soportada para Bonsai GPU
  es la **imagen OCI `Dockerfile.bitnet`** (base CUDA devel). Se validó compilando en un `nvidia/cuda:12.4.1-cudnn-devel` one-off.
- **Link contra el stub del driver**: el link final necesita `libcuda.so` → usar el stub
  (`/usr/local/cuda/lib64/stubs`), que los scripts activan cuando `BITNET_ENABLE_CUDA=true`.
- **Libs de runtime a empaquetar**: el contenedor destino aporta solo `libcuda.so.1` (driver, vía NVIDIA runtime).
  `bonsai-server`/`libggml-cuda` también enlazan `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` y **`libnccl.so.2`**,
  que NO están de sistema en el api (solo en venvs pip). Los scripts ahora **copian esas libs junto al binario** (dir `prismml/`)
  y preservan los **symlinks soname** (`cp -a build/bin/*.so*`, no `find -type f`). El backend ya mete ese dir en `LD_LIBRARY_PATH`.
- **`BITNET_ENABLE_CUDA` default = false**: para Bonsai GPU hay que construir con `BITNET_ENABLE_CUDA=true`
  (si no, el fork se compila CPU-only). Coupla con el build de Microsoft (que es CPU-first por diseño).

## Deudas

- **Desplegar el código nuevo en el api**: la validación de generación se hizo lanzando `bonsai-server` a mano;
  para que oCabra lo enrute solo, el contenedor `ocabra-api-1` debe correr esta rama (imagen con los cambios del backend)
  y tener el `prismml/bonsai-server` instalado (vía OCI image o copiado al volumen `ocabra_backends_data`).
- **Visión de Bonsai**: es multimodal (usa `Bonsai-27B-mmproj-*.gguf`). Ahora `vision=False` — no cableada la entrada de imágenes.
- **dspark / speculative decoding**: los drafters se excluyen como modelo servible; NO se explota aún el 1.34× de speculative decoding.
- **Coste del build**: instalar `bitnet` ahora compila también el fork PrismML (pesado). Desactivable con `BITNET_BUILD_PRISMML=false`
  (en ese caso Bonsai no arranca hasta apuntar `settings.bitnet_prismml_server_bin` a un binario válido).

## Cuestiones pendientes

- ¿Fijar Bonsai 27B como modelo `on_demand` o PIN de producción? (VRAM ~5.2 GB @4K ctx).
- ¿Merece la pena el drafter dspark (1.34× decode) dado el coste de doble modelo en VRAM?
- Confirmar que el `llama-server` del fork PrismML mantiene la API OpenAI-compat que consume oCabra (chat/completions + streaming).

## Siguiente paso (fuera de este cambio)

**VibeVoice-ASR-BitNet** (STT ternario, Microsoft): backend nuevo `vibeasr` (modalidad `AUDIO_TRANSCRIPTION`)
con worker shim HTTP sobre el CLI `asr_infer` de VibeASR.cpp. Ver plan de voz (`docs/tasks/voice-pipeline-plan.md`).
