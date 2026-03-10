# Briefing: Stream 2-B — Diffusers Backend (Generación de imagen)

**Prerequisito: Stream 1-B completado.**
**Rama:** `feat/2-B-diffusers`

## Objetivo

Implementar el backend de generación de imagen usando Diffusers de HuggingFace.
Compatible con el endpoint OpenAI `/v1/images/generations`.

## Ficheros propios

```
backend/ocabra/backends/diffusers_backend.py
workers/diffusers_worker.py
backend/tests/test_diffusers_backend.py
```

## Modelos soportados

| Familia | Pipeline | VRAM aprox |
|---------|----------|-----------|
| Stable Diffusion 1.5 | `StableDiffusionPipeline` | ~4 GB |
| Stable Diffusion XL | `StableDiffusionXLPipeline` | ~7 GB |
| FLUX.1-schnell | `FluxPipeline` | ~12 GB (bfloat16) |
| FLUX.1-dev | `FluxPipeline` | ~24 GB |

## Arquitectura del worker

A diferencia de vLLM, el worker de Diffusers es un script FastAPI propio
(`workers/diffusers_worker.py`) que expone:

```
POST /generate
  body: {
    prompt: str,
    negative_prompt?: str,
    width?: int,          # default 1024
    height?: int,         # default 1024
    num_inference_steps?: int,  # default 20
    guidance_scale?: float,     # default 7.5
    seed?: int,
    num_images?: int      # default 1
  }
  response: {
    images: [{ b64_json: str }],
    generation_time_ms: int,
    seed_used: int
  }

GET /health
GET /info   → { model_id, pipeline_type, vram_used_mb }
```

## diffusers_backend.py

```python
class DiffusersBackend(BackendInterface):

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        """
        Lanza workers/diffusers_worker.py como subproceso con:
        CUDA_VISIBLE_DEVICES={gpu_indices[0]}  (no tensor parallel para imagen)
        --model-path /data/models/{model_id}
        --port {assigned_port}
        Espera healthcheck hasta 180s (carga lenta en Diffusers).
        """

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """Siempre retorna: image_generation=True, el resto False."""

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        """Suma ficheros .safetensors × 1.3 de overhead."""

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """
        Traduce el formato OpenAI images.generations a formato interno del worker:
        - OpenAI: {prompt, n, size, response_format}
        - Worker: {prompt, width, height, num_images, ...}
        Retorna en formato OpenAI: {created, data: [{b64_json|url}]}
        """
```

## workers/diffusers_worker.py

```python
"""
FastAPI worker para generación de imagen.
Carga el pipeline al arrancar, sirve requests de generación.

Uso: python diffusers_worker.py --model-path /data/models/flux-schnell --port 18010
"""

import argparse
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, ...
from fastapi import FastAPI

def detect_pipeline_class(model_path: Path) -> type:
    """Lee model_index.json para detectar el tipo de pipeline."""

app = FastAPI()

@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    # Usar asyncio.get_event_loop().run_in_executor para no bloquear
    ...
```

## Traducción formato OpenAI → worker

```python
# Input OpenAI:
{
  "prompt": "a cat in space",
  "n": 1,
  "size": "1024x1024",   # "256x256" | "512x512" | "1024x1024" | "1792x1024" | "1024x1792"
  "response_format": "b64_json"  # | "url"
}

# Output OpenAI:
{
  "created": 1234567890,
  "data": [{"b64_json": "..."}]
}
```

## Optimizaciones recomendadas

- `torch.compile()` en el pipeline si CUDA disponible
- `enable_model_cpu_offload()` para modelos grandes cuando hay presión de VRAM
- `torch.float16` / `bfloat16` según la GPU (bfloat16 mejor en 3090)
- Batch de requests si llegan varios simultáneamente al mismo modelo

## Tests requeridos

- Mock del subprocess: test de load/unload
- Test de traducción de tamaños: "1792x1024" → width=1792, height=1024
- Test de detección de pipeline_class por model_index.json

## Dependencias adicionales (añadir a pyproject.toml)

```toml
# En sección opcional [project.optional-dependencies]
diffusers = [
    "diffusers>=0.31",
    "accelerate>=1.2",
    "transformers>=4.47",
    "Pillow>=11.0",
    "sentencepiece>=0.2",
]
```

## Estado

- [ ] En progreso
- [x] Completado
