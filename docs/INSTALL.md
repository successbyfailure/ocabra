# oCabra Installation Guide

oCabra runs as a Docker Compose stack. `setup.sh` is the entry point for a new
install on Linux or Windows/WSL2; everything below describes what it does and
how to do the same by hand.

What a default install runs (`docker-compose.yml`):

| Service | Role |
|---------|------|
| `api` | FastAPI server (OpenAI `/v1`, Ollama `/api`, internal `/ocabra`), image `ocabra-api:slim`. Runs **as a normal user** (`OCABRA_UID`, not root) and has **no Docker socket**. |
| `docker-agent` | Same image, different process: the only container with the Docker socket. Runs oCabra's own Docker operations (optional services, the Ollama daemon, self-update, TensorRT-LLM builds) for the api, authenticated with a token on a shared volume. |
| `frontend` | The panel (nginx serving the React build). |
| `postgres`, `redis` | State. Redis requires `REDIS_PASSWORD` when set. Neither is published on the host. |
| `hw-monitor` | Privileged sidecar: GPU/CPU sampling, power limits and the thermal cap. |
| `ollama` | One shared Ollama daemon (official image) for every Ollama model. |
| `caddy` | Front door on host port **8484**: routes `/v1`, `/api`, `/ocabra`, `/health`, `/ready`, `/version`, `/docs`, `/metrics` to the api and everything else to the panel. |

Optional services (compose profile `services`, started with
`bash setup.sh --with-services`): `gateway`, `trellis2`, `comfyui`, `a1111`,
`acestep`, `unsloth-studio`.

Inference backends (llama.cpp, vLLM, Whisper, Diffusers, …) are **not** in the
image: each one is installed separately into `/data/backends` from the panel's
**Backends** page
(see [Backends](#backends)).

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU, 8 GB VRAM | 24 GB+ (RTX 3090/4090, A5000…); several GPUs are supported |
| RAM | 16 GB | 32 GB+ |
| Disk | 50 GB free for images and backends | 200 GB+ for models |
| CPU | 4 cores | 8+ cores |

Disk, roughly: the api image is ~3.2 GB, each installed backend adds from a few
hundred MB (llama.cpp, whisper.cpp) to several GB (vLLM, torch-based ones),
and models take what they take. A model download refuses to start with less
than `DOWNLOAD_MIN_FREE_GB` (30) free.

### Software

1. **Linux** (Ubuntu 22.04+ or Debian 12+) or **Windows 11 with WSL2**.
2. **NVIDIA driver** with CUDA 13 support (**R580 or newer**). The api image
   compiles with `nvcc` 13.0 and the published backends are CUDA 13 builds;
   an older driver can see the GPU but not run them.
3. **Docker Engine** (Linux) or current **Docker Desktop** with the WSL2
   backend (Windows).
4. **Docker Compose v2.30+** (`setup.sh` refuses older ones: portable GPU
   requests need it).
5. Linux only: **NVIDIA Container Toolkit**
   ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
   Docker Desktop provides GPU access on WSL2.
6. `git`, `curl`, and preferably `python3` or `openssl` (used to generate secrets).

`setup.sh` checks GPU access the same way you can:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
git clone https://github.com/successbyfailure/ocabra
cd ocabra
bash setup.sh                 # or: bash setup.sh --with-services
```

Then open **http://localhost:8484**. The API is at `http://localhost:8484/v1`
(OpenAI) and `http://localhost:8484/api` (Ollama); Swagger at `/docs`.

What `setup.sh` does, in order:

1. Checks Docker, Compose 2.30+, `curl`, `nvidia-smi` and GPU access from a
   CUDA container. On WSL2 it stops if the repo is on a Windows drive.
2. Creates `.env` from `.env.example` if there is none.
3. On WSL2: moves `AI_MODELS_ROOT` and `ACESTEP_ROOT` under
   `~/.local/share/ocabra/` and sets `THERMAL_CAP_ENABLED=false` and
   `GPU_THROTTLE_ENABLED=false`.
4. Fills every empty secret with a random value: `JWT_SECRET`,
   `GATEWAY_SERVICE_TOKEN`, `REDIS_PASSWORD`, `OCABRA_SECRETS_KEY` and
   `POSTGRES_PASSWORD`, and writes `DATABASE_URL` from the `POSTGRES_*` values.
5. On the first interactive run, asks for the admin user and password. A blank
   password is replaced by a random one that is **only** written to `.env`:
   read it with `grep OCABRA_ADMIN_PASS .env`.
6. Sets `OCABRA_UID`/`OCABRA_GID` to the user running the script (the api runs
   as that user) and creates the data folders (`data/…`, `caddy/…`, and under
   `AI_MODELS_ROOT` and `ACESTEP_ROOT`). It warns about folders that user does
   not own.
7. Validates the compose file (`docker compose config -q`), builds `api`,
   `frontend`, `hw-monitor` (plus the optional services whose sources exist),
   starts the core services and waits up to 120 s for
   `http://localhost:8484/health`.

The api applies the database migrations itself on every start
(`alembic upgrade head` in `backend/docker-entrypoint.sh`; set
`OCABRA_SKIP_MIGRATIONS=true` to skip them).

**Back up `.env`.** `OCABRA_SECRETS_KEY` encrypts the secrets stored in the
database (federation keys, MCP credentials); it lives only in `.env`, and
without it they cannot be read.

## Windows 11 / WSL2

oCabra runs as Linux containers through Docker Desktop's WSL2 backend; there
is no native Windows mode. Install the current Windows NVIDIA driver and
Docker Desktop with WSL integration enabled for your distribution, then:

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
```

Inside the Ubuntu shell:

```bash
git config --global core.autocrlf input
mkdir -p ~/src && cd ~/src
git clone https://github.com/successbyfailure/ocabra && cd ocabra
bash setup.sh
```

- **Clone inside WSL** (`~/src`), never on a Windows drive (`/mnt/c`,
  `/mnt/d`…): those cannot hold Linux permissions, so Postgres refuses to
  start, the non-root api cannot own its data, and Docker is very slow there.
  `setup.sh` stops with an error if it finds itself under `/mnt/<letter>`.
- **Line endings.** The repo's `.gitattributes` (`* text=auto eol=lf`) keeps
  `setup.sh` and the scripts copied into the images (`docker-entrypoint.sh`…)
  in LF even in a Windows clone; with CRLF they fail with
  `env: 'bash\r': No such file or directory`. `core.autocrlf input` avoids
  converting anything else.
- Do not install a Linux NVIDIA driver inside WSL.
- **Power and thermal control are off.** WSL's NVML bridge does not allow
  changing power limits or persistence mode, so `setup.sh` disables the
  thermal cap and throttle. `hw-monitor` still reports GPUs; CPU energy (RAPL,
  `/sys/class/powercap`) does not exist on WSL2, which is why compose no longer
  bind-mounts it (a privileged container already sees the host's sysfs).
- On multi-GPU Windows hosts, check that the CUDA and NVML device order match
  before relying on per-GPU placement.

## Manual setup

The same steps as `setup.sh`, by hand:

```bash
cp .env.example .env
# Secrets: JWT_SECRET, OCABRA_SECRETS_KEY, REDIS_PASSWORD, GATEWAY_SERVICE_TOKEN,
# POSTGRES_PASSWORD (e.g. python3 -c 'import secrets; print(secrets.token_hex(32))')
# and DATABASE_URL=postgresql+asyncpg://ocabra:<password>@postgres:5432/ocabra
# (URL-encode the password). Also OCABRA_ADMIN_USER / OCABRA_ADMIN_PASS.
echo "OCABRA_UID=$(id -u)" >> .env; echo "OCABRA_GID=$(id -g)" >> .env

# Folders, owned by that user (AI_MODELS_ROOT defaults to /docker/ai-models)
mkdir -p data/{postgres_data,openai_files,image_outputs,ollama} caddy/{caddy_data,caddy_config}
mkdir -p "$AI_MODELS_ROOT"/ocabra/{models,hf_cache} "$AI_MODELS_ROOT"/ollama "$ACESTEP_ROOT"/checkpoints

docker compose config -q
docker compose build api frontend hw-monitor
docker compose up -d api docker-agent frontend postgres redis hw-monitor ollama caddy
curl http://localhost:8484/health
```

`.env.example` documents every variable, grouped, with its default. Lines left
commented keep the default. Settings saved later from the panel (Settings
page, stored in the `server_config` table) are applied at every api start and
**override `.env`**.

## Backends

Each inference backend is installed on its own from the **Backends** page
(or `POST /ocabra/backends/{type}/install`) into
`/data/backends/<backend>/` — the `backends_data` volume, which survives image
rebuilds and updates. Every backend except Ollama (its own container) is
installed this way, including llama.cpp (`llama_cpp`, `llamacpp_mks`), BitNet,
ACE-Step and TensorRT-LLM.

Install methods (`core/backend_installer.py`):

- **oci** — download the backend already built, as published in
  `BACKEND_OCI_REPOSITORY` (`ghcr.io/successbyfailure/ocabra-backends`, one
  tag per backend, see `core/oci_registry.py`). No compilers, minutes instead
  of tens of minutes. If that repository is private, set `BACKEND_OCI_TOKEN`
  (a GitHub token with `read:packages`).
- **source** — create a venv and pip-install, or compile (llama.cpp, BitNet,
  whisper.cpp) with the CUDA toolchain in the image. Build options for BitNet
  are the `BITNET_*` variables in `.env.example`.
- **auto** (default) — oci when the backend is published, otherwise or if
  that fails, source.

Reinstalling an installed backend updates it: the new copy is unpacked next to
the old one and swapped in, so a failure keeps the old one. A backend cannot be
uninstalled while one of its models is loaded.

ACE-Step also needs its checkout and weights at `ACESTEP_ROOT` (mounted
read-only, and `checkpoints/` read-write).

### Image variants

| Image | Size | Notes |
|-------|------|-------|
| `ocabra-api:slim` (default, `backend/Dockerfile.slim`) | ~3.2 GB | `python:3.11-slim-bookworm` + CUDA 13 compiler; no backends (`BACKENDS_FAT_IMAGE=false`). |
| `ghcr.io/successbyfailure/ocabra-api:X.Y.Z` | ~3.2 GB | The same image, published per release (see [Updating](#updating)). |
| `ocabra-api:fat` (`backend/Dockerfile` + `docker-compose.fat.yml`) | tens of GB | Legacy: every backend baked into the image. Kept as a rollback; not needed any more, since every backend installs on slim. |
| `ghcr.io/successbyfailure/ocabra-node:X.Y.Z` (`backend/Dockerfile.node`) | ~2.3 GB | Self-contained node for rented GPUs (see [Federation](#federation-and-cloud-nodes)). |

## Ports and network

| Host port | Service | Variable |
|-----------|---------|----------|
| 8484 | Caddy: panel + API | fixed in `docker-compose.yml` (`caddy.ports`) |
| 8485 | Caddy: services portal | `SERVICES_PORTAL_PORT` |
| 127.0.0.1:7869 | Ollama, host only (it has no authentication) | `OLLAMA_BIND` |
| 9000 | gateway (optional) | `GATEWAY_PORT` |
| 8481 / 8188 / 7860 / 7862 / 8889 | TRELLIS.2 / ComfyUI / A1111 / ACE-Step / Unsloth Studio (optional) | `TRELLIS2_PORT`, `COMFYUI_PORT`, `A1111_PORT`, `ACESTEP_GRADIO_PORT`, `UNSLOTH_STUDIO_PORT` |

The api itself (8000), Postgres and Redis are only reachable on the compose
network. `/metrics` answers only to private addresses (Caddy returns 403
otherwise).

**Behind another reverse proxy.** Caddy only trusts `X-Forwarded-For` from the
proxies listed in `caddy/Caddyfile` (`trusted_proxies`; the shipped file lists
`10.0.8.5`, the openresty in front of the reference deployment). Replace it
with your proxy's address, or the api will see every request as coming from
the proxy (wrong client IPs in the stats, and `/metrics` reachable through it
if the proxy is on a private network). Then set `PUBLIC_URL` (absolute links
such as `response_format=url` images), `USE_HTTPS=true` if the panel is served
over HTTPS, and `AUTH_COOKIE_DOMAIN` if the panel and the service portals
share a parent domain.

## File ownership (non-root api)

Since v0.106.2 the api runs as `OCABRA_UID:OCABRA_GID` (1000 by default), so it
must own what it writes: `data/openai_files`, `data/image_outputs`,
`data/ollama`, `AI_MODELS_ROOT/ocabra/{models,hf_cache}`, the ACE-Step
checkpoints and the `backends_data` volume.

An install from before v0.106.2 has all that owned by root. Fix it once, from
the repo root, with the api stopped (or right before recreating it):

```bash
backend/scripts/chown-api-data.sh --dry-run   # list what would change
backend/scripts/chown-api-data.sh             # chown the api's read-write mounts
```

It reads the mounts from `docker compose config` and only changes owners (no
data is copied). For a root shell in the api container use
`docker compose exec -u 0 api bash`.

## Configuration essentials

The full list, with defaults, is `.env.example`. The ones most installs touch:

| Variable | Default | Description |
|----------|---------|-------------|
| `OCABRA_ADMIN_USER` / `OCABRA_ADMIN_PASS` | `ocabra` / `ocabra` | First admin; only used when the users table is empty. |
| `AI_MODELS_ROOT` | `/docker/ai-models` | Host folder holding `ocabra/models`, `ocabra/hf_cache` and `ollama`. All model mounts derive from it; change it before the first start. |
| `ACESTEP_ROOT` | `/docker/ACE-Step-1.5` | ACE-Step checkout and weights. |
| `HF_TOKEN` | *(empty)* | Hugging Face token for gated/private models. |
| `DEFAULT_GPU_INDEX` | `1` | GPU used when a model has no preference (NVML index, PCI order). |
| `REQUIRE_API_KEY_OPENAI` / `REQUIRE_API_KEY_OLLAMA` | `true` | Require an API key on `/v1` and `/api`. |
| `IDLE_TIMEOUT_SECONDS` | `300` | Idle time before an on-demand model is unloaded. |
| `PRIVACY_MODE` | `functional` | `zero_data` keeps nothing about requests (and disables Files/Batches). |
| `SELF_UPDATE_POLICY` | `notify` | `off`, `notify` or `auto` (nodes on published images). |
| `LOG_LEVEL` | `INFO` | Backend log level. |

Model folders: models live under `AI_MODELS_ROOT/ocabra/models` (mounted as
`/data/models`); `MODELS_DIR` and `HF_CACHE_DIR` are fixed by compose. Setting
`OLLAMA_SHARED_MODELS_DIR=/data/ollama_models` (as `.env.example` does) also
offers the GGUFs Ollama already downloaded as llama.cpp models.

## First steps after install

1. Log in at `http://localhost:8484` with the admin credentials.
2. **Backends** page: install the backends you need (llama.cpp is the
   usual first one).
3. **Explore**: download models from Hugging Face or Ollama; or register files
   you already have.
4. Create an API key on the **API Keys** page and test:

   ```bash
   curl http://localhost:8484/v1/models -H "Authorization: Bearer YOUR_API_KEY"
   ```

`docs/API.md` lists every route; `/docs` (Swagger) has the request schemas of
the public API, and `/docs/changelog` what changed between versions.

## Updating

- **Stacks on published images** — `.env` sets
  `OCABRA_API_IMAGE=ghcr.io/successbyfailure/ocabra-api:X.Y.Z` and
  `OCABRA_FRONTEND_IMAGE=ghcr.io/successbyfailure/ocabra-frontend:X.Y.Z`. The
  node updates itself according to the update policy (**Settings**, "Actualizaciones de oCabra";
  `SELF_UPDATE_POLICY`): a one-shot `ocabra-updater` container fast-forwards
  the git checkout to the release tag, rewrites those two lines, pulls and
  recreates `api` and `frontend`, and rolls back if `/health` does not report
  the new version. A checkout with local changes is left alone. To move a
  stack to published images, add the two lines and run
  `docker compose pull api frontend && docker compose up -d api frontend`.
- **Stacks that build their own images** (no such variables) — the panel only
  announces new releases. Update with
  `git pull && docker compose build api frontend && docker compose up -d`.
  Build from a clean checkout: `docker compose build` copies the working tree,
  uncommitted changes included.

Installed backends are not touched by either; update them from the
**Backends** page.

## Federation and cloud nodes

Several oCabra servers can share their models (**Federation** page): a node
joins another with an invite, or by a join request that an admin accepts.
Peers that cannot be reached (no public URL) connect through an outbound
reverse tunnel. Rented GPUs (RunPod, from the cloud section of the **Federation** page) run the
`ocabra-node` image — API, Postgres, Redis and llama.cpp in one container, no
panel — which joins the hub on start with `FEDERATION_JOIN_URL` and
`FEDERATION_JOIN_INVITE`. Settings: the "Federación" section of `.env.example`.

## Troubleshooting

### Docker cannot access the GPU

- Linux: install the NVIDIA Container Toolkit, then
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- WSL2: update WSL and the Windows NVIDIA driver; enable Docker Desktop's WSL2
  backend and the integration for your distribution.
- `nvidia-smi` missing on the host: install the driver (R580+) and reboot.

### The api is unhealthy

```bash
docker compose logs api docker-agent
docker compose ps
```

Usual causes: Postgres not ready yet (the entrypoint retries for 30 s),
`DATABASE_URL` not matching `POSTGRES_*` (re-run `setup.sh`, which rewrites
it), or data folders the api user does not own (`Permission denied` in the
logs: see [File ownership](#file-ownership-non-root-api)).

### Port 8484 is taken

Change the first `caddy.ports` entry in `docker-compose.yml`
(`"YOUR_PORT:80"`); the other ports have variables (see
[Ports and network](#ports-and-network)).

### A backend fails to install

The install log is on the Backends page (`GET /ocabra/backends/{type}/logs`).
A source build needs disk space and time; an `oci` install that fails with an
authorization error means the repository is private and `BACKEND_OCI_TOKEN`
is missing.

### Models fail to load / out of VRAM

- The model editor (Models → edit) shows the memory plan per GPU before
  loading; use a smaller context preset or a more quantized file.
- For vLLM, lower `VLLM_GPU_MEMORY_UTILIZATION`.
- Unloaded idle models free VRAM after `IDLE_TIMEOUT_SECONDS`; PIN models stay
  loaded.

## Stopping and restarting

```bash
docker compose down        # stop (data, volumes and installed backends stay)
docker compose up -d       # start again
```
