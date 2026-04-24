# oCabra Installation Guide

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 8 GB VRAM (CUDA compute capability >= 7.0) | 24 GB+ VRAM (e.g. RTX 3090, RTX 4090, A5000) |
| RAM | 16 GB | 32 GB+ |
| Disk | 50 GB free (OS + containers) | 200 GB+ (model storage) |
| CPU | 4 cores | 8+ cores |

Multiple GPUs are supported. oCabra will detect and manage all available NVIDIA GPUs.

## Software Prerequisites

1. **Linux** (Ubuntu 22.04+ recommended; other distros work with Docker support)
2. **NVIDIA drivers** >= 535 (check with `nvidia-smi`)
3. **Docker Engine** >= 24.0 ([install guide](https://docs.docker.com/engine/install/))
4. **Docker Compose** v2 plugin (included with modern Docker Desktop/Engine)
5. **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

Verify the toolkit is configured:

```bash
# Should show GPU info inside the container
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> ocabra
cd ocabra

# 2. Run the setup script
chmod +x setup.sh
bash setup.sh
```

The script will:
- Check all prerequisites
- Create data directories
- Generate `.env` with random secrets
- Prompt for admin credentials and DB password
- Build and start all Docker services
- Wait for the health check to pass

Once complete, open **http://localhost:8484** in your browser.

## Manual Setup

If you prefer to set things up step by step:

```bash
# 1. Create .env
cp .env.example .env

# 2. Generate a JWT secret and add it to .env
echo "JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env

# 3. Create data directories
mkdir -p data/postgres_data caddy/caddy_data caddy/caddy_config

# 4. Create model storage directories
# Default path is /docker/ai-models — change AI_MODELS_ROOT in .env if needed
sudo mkdir -p /docker/ai-models/ocabra/{models,hf_cache}
sudo mkdir -p /docker/ai-models/ollama

# 5. Build and start
docker compose build api frontend postgres redis caddy
docker compose up -d

# 6. Verify
curl http://localhost:8484/health
```

## Image variants (Bloque 15 — Fase 4)

The `api` service ships in two flavours; the default since 2026-04-25 is **slim**.

| Variant | Image | Size | Backends |
|---------|-------|------|----------|
| **slim** (default) | `ocabra-api:slim` | ~987 MB | None pre-installed; install on demand via `POST /ocabra/backends/{type}/install` |
| **fat** (rollback) | `ocabra-api:fat`  | ~51 GB  | All backends pre-installed in the image (legacy behaviour) |

### Default — slim

```bash
docker compose up -d --build api
```

After the api comes up, install the backends you need from the **Backends** page in the UI (or via the API). Each backend gets its own `/data/backends/<name>/venv/` and consumes disk only when installed. Backends that have not yet been migrated to `install_spec` (acestep, bitnet, llama_cpp, tensorrt_llm) appear as "built-in" but are NOT actually present on slim — for those you still need fat.

### Fallback — fat

```bash
docker compose -f docker-compose.yml -f docker-compose.fat.yml up -d --build api
```

Use this when you depend on backends that are not yet installable on slim (the four listed above), or when you want the historical "everything works out of the box" experience. The fat image enables `BACKENDS_FAT_IMAGE=true` so the UI shows every backend as `built-in/installed`.

### Switching between variants

The on-disk install state for slim lives in the `backends_data` Docker volume (`/data/backends`). Switching between slim and fat does not delete it; whatever you installed on slim stays available the next time you boot slim, and is simply ignored when you boot fat (the fat image uses its own pre-installed copies).

## Configuration Reference

Key variables in `.env` (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `OCABRA_ADMIN_USER` | `ocabra` | Admin login username |
| `OCABRA_ADMIN_PASS` | `ocabra` | Admin login password |
| `JWT_SECRET` | *(generated)* | Secret for signing JWT tokens. If empty, regenerated on every restart (invalidates sessions). |
| `POSTGRES_PASSWORD` | `change_me_in_production` | PostgreSQL password |
| `AI_MODELS_ROOT` | `/docker/ai-models` | Host path for model storage |
| `DEFAULT_GPU_INDEX` | `1` | Preferred GPU for loading models |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | Fraction of VRAM vLLM reserves (0.0-1.0) |
| `IDLE_TIMEOUT_SECONDS` | `300` | Seconds before unloading idle models |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models |
| `GATEWAY_PORT` | `9001` | Host port for the services gateway |
| `LOG_LEVEL` | `INFO` | Backend log level |

### Changing the model storage path

The default `docker-compose.yml` mounts `/docker/ai-models/ocabra` into containers. To use a different path:

1. Set `AI_MODELS_ROOT` in `.env`
2. Update the volume paths in `docker-compose.yml` to match

## First Steps After Install

1. **Log in** at `http://localhost:8484` with your admin credentials.
2. **Generate an API key** from Settings to use the OpenAI-compatible API.
3. **Add models** from the Explore page or pull Ollama models.
4. **Test** with curl:
   ```bash
   curl http://localhost:8484/v1/models \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

## Troubleshooting

### GPU not detected

```
Error: nvidia-smi not found
```

- Install NVIDIA drivers: `sudo apt install nvidia-driver-535` (or newer)
- Reboot after installation

```
Error: NVIDIA Container Toolkit not found
```

- Install it: `sudo apt install nvidia-container-toolkit`
- Configure Docker: `sudo nvidia-ctk runtime configure --runtime=docker`
- Restart Docker: `sudo systemctl restart docker`

### Port conflicts

If port 8484 is in use, change it in `docker-compose.yml` under the `caddy` service:

```yaml
ports:
  - "YOUR_PORT:80"
```

### API won't start / unhealthy

```bash
# Check logs
docker compose logs api

# Common causes:
# - PostgreSQL not ready yet (usually resolves on retry)
# - Missing .env file
# - Invalid DATABASE_URL
```

### Out of disk space during build

The backend image is large (~15 GB) due to CUDA + vLLM + model runtimes. Ensure at least 50 GB free on the Docker storage partition.

```bash
# Clean up old images
docker system prune -a
```

### Permission denied on data directories

```bash
# Fix ownership (match FRONTEND_UID/GID in .env, default 1000)
sudo chown -R 1000:1000 data/
```

### Models fail to load / VRAM errors

- Lower `VLLM_GPU_MEMORY_UTILIZATION` (e.g. `0.7`) to leave more VRAM headroom
- Increase `IDLE_TIMEOUT_SECONDS` or set to `0` to unload models immediately when idle
- Use smaller quantized models (GGUF/AWQ/GPTQ) for GPUs with limited VRAM

## Stopping and Restarting

```bash
# Stop all services
docker compose down

# Start again
docker compose up -d

# Rebuild after code changes
docker compose build && docker compose up -d
```
