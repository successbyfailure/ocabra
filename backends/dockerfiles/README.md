# oCabra — Backend OCI images

This directory contains one Dockerfile per inference backend. These images
are consumed by `BackendInstaller` (see `docs/tasks/modular-backends-plan.md`,
Fase 3 — Distribución OCI) rather than being executed as long-running
containers. Each final stage is built `FROM scratch` and contains only a
single `/backend/` tree so that `docker cp` into the live host filesystem is
a clean, self-contained extraction.

---

## 1. Naming convention

```
ghcr.io/ocabra/backend-<backend_type>:<tag>
```

- `<backend_type>` matches `BackendInterface.install_spec.oci_image` — same
  stable id used in `/ocabra/backends/{backend_type}/install`.
- `<tag>` encodes the release channel and hardware variant:

| Tag form                 | Meaning                                          |
|--------------------------|--------------------------------------------------|
| `latest-cuda12`          | Rolling latest build for CUDA 12.x hosts         |
| `latest-cuda11`          | Rolling latest build for CUDA 11.x hosts         |
| `latest-cpu`             | CPU-only variant (no NVIDIA runtime required)    |
| `latest-rocm`            | AMD ROCm variant (future)                        |
| `v<backend_ver>-cuda12`  | Pinned release, e.g. `v0.17.1-cuda12` for vllm   |
| `dev`                    | Local development build                          |

Hardware detection at install time (`detect_gpu_variant()`) picks the best
available tag from `install_spec.oci_tags`.

---

## 2. Internal `/backend/` layout

All images expose **exactly one** top-level directory `/backend/`:

```
/backend/
  ├── venv/                  Python venv (Python-based backends only)
  │   ├── bin/python
  │   └── lib/python3.11/site-packages/...
  ├── bin/                   Native binaries and/or helper tools
  │   ├── llama-server       (llama_cpp)
  │   ├── bitnet-server      (bitnet)
  │   └── ffmpeg             (bundled with TTS/audio images)
  ├── lib/                   Non-system shared libraries (native backends)
  │   └── libggml*.so*       (llama_cpp, bitnet)
  ├── workers/               oCabra worker scripts that run inside venv
  │   └── <backend>_worker.py
  ├── project/               Optional: cloned upstream project (e.g. ACE-Step)
  └── metadata.template.json Seed metadata (installer overwrites with resolved
                             digest/version/install_source/installed_at)
```

`BackendInstaller` is the only code that reads from this tree. When
`metadata.json` is present at `${settings.backends_dir}/<backend>/`, the
backend implementation reads `python_bin` / `extra_bins` from it and uses
those paths for `subprocess.exec`.

---

## 3. Building locally

Context must be the **repo root** (so `backend/workers/...` and
`backend/scripts/...` resolve). All commands are run from the repo root:

```bash
# Python-only backends (fast)
docker build -f backends/dockerfiles/Dockerfile.vllm      -t ghcr.io/ocabra/backend-vllm:dev      .
docker build -f backends/dockerfiles/Dockerfile.whisper   -t ghcr.io/ocabra/backend-whisper:dev   .
docker build -f backends/dockerfiles/Dockerfile.diffusers -t ghcr.io/ocabra/backend-diffusers:dev .
docker build -f backends/dockerfiles/Dockerfile.tts       -t ghcr.io/ocabra/backend-tts:dev       .
docker build -f backends/dockerfiles/Dockerfile.sglang    -t ghcr.io/ocabra/backend-sglang:dev    .
docker build -f backends/dockerfiles/Dockerfile.voxtral   -t ghcr.io/ocabra/backend-voxtral:dev   .
docker build -f backends/dockerfiles/Dockerfile.chatterbox -t ghcr.io/ocabra/backend-chatterbox:dev .
docker build -f backends/dockerfiles/Dockerfile.acestep   -t ghcr.io/ocabra/backend-acestep:dev   .

# Native builds (slow)
docker build -f backends/dockerfiles/Dockerfile.llama_cpp -t ghcr.io/ocabra/backend-llama-cpp:dev .
docker build -f backends/dockerfiles/Dockerfile.bitnet    -t ghcr.io/ocabra/backend-bitnet:dev    .

# Special case (see TODO in the Dockerfile)
docker build -f backends/dockerfiles/Dockerfile.tensorrt_llm \
             -t ghcr.io/ocabra/backend-tensorrt-llm:dev .
```

Switching variants is a single `--build-arg`:

```bash
# CPU-only whisper (no CUDA runtime needed)
docker build \
  -f backends/dockerfiles/Dockerfile.whisper \
  --build-arg BASE_IMAGE=python:3.11-slim \
  -t ghcr.io/ocabra/backend-whisper:dev-cpu .

# Pin a specific vLLM version
docker build \
  -f backends/dockerfiles/Dockerfile.vllm \
  --build-arg BACKEND_VERSION=0.17.1 \
  -t ghcr.io/ocabra/backend-vllm:v0.17.1-cuda12 .
```

---

## 4. How `BackendInstaller.install()` consumes these images

Full flow is documented in `docs/tasks/modular-backends-plan.md` §1-2. Short
version:

```python
# Pseudocode from core/backend_installer.py
image_ref = f"{spec.oci_image}:{spec.oci_tags[variant]}"
await docker("pull", image_ref)
cid = await docker("create", image_ref)
try:
    await docker("cp", f"{cid}:/backend/.", str(target_dir))
finally:
    await docker("rm", cid)

metadata = json.loads((target_dir / "metadata.template.json").read_text())
metadata.update({
    "installed_at": datetime.now(UTC).isoformat(),
    "install_source": "oci",
    "oci_image": spec.oci_image,
    "oci_tag": spec.oci_tags[variant],
    "oci_digest": digest,
    "size_mb": measure_dir_size_mb(target_dir),
})
(target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

Because the final layer is `FROM scratch`, `docker cp` extracts **only**
the `/backend/` tree without base-image noise (no `/usr`, no `/etc`, no
duplicate CUDA runtime — the host image already provides those).

---

## 5. Size / dependency summary

| Backend       | Est. size | Main deps                                                  | Native build | Notes |
|---------------|-----------|------------------------------------------------------------|--------------|-------|
| vllm          | ~9.5 GB   | vllm 0.17.1, torch 2.5                                     | no           | No worker script — spawned via `python -m vllm.entrypoints.openai.api_server` |
| whisper       | ~4.5 GB   | faster-whisper, NeMo ASR, pyannote, torch, librosa         | no           | Needs `ffmpeg` (bundled only in tts/voxtral/chatterbox images; whisper host requires it) |
| diffusers     | ~6 GB     | diffusers 0.31, accelerate, transformers, torch            | no           |       |
| tts           | ~5 GB     | qwen-tts, kokoro, transformers 5.x, torch                  | no           | ffmpeg bundled at `/backend/bin/ffmpeg` |
| sglang        | ~9 GB     | sglang 0.5.9 (+torch transitive)                           | no           |       |
| voxtral       | ~10 GB    | vllm 0.18, vllm-omni                                       | no           | ffmpeg bundled |
| chatterbox    | ~4.5 GB   | chatterbox-tts, torch, torchaudio                          | no           | ffmpeg bundled |
| acestep       | ~7 GB     | ACE-Step-1.5 upstream (via `uv sync`)                      | no           | Clones project into `/backend/project` |
| llama_cpp     | ~250 MB   | llama-server binary + libggml*.so                          | **yes**      | CUDA archs 61;70;75;80;86;89 |
| bitnet        | ~200 MB   | bitnet-server binary (reuses `backend/scripts/build_bitnet.sh`) | **yes**  | Defaults to CPU kernels; `BITNET_ENABLE_CUDA=true` to switch |
| tensorrt_llm  | ~18 GB    | tensorrt_llm wheel from NVIDIA pypi                        | no           | **Scaffold** — see TODO at top of its Dockerfile |

---

## 6. Multi-arch / variant matrix

All Dockerfiles accept `--build-arg BASE_IMAGE=...`. Canonical variants:

| Variant | BASE_IMAGE                                               |
|---------|----------------------------------------------------------|
| cuda12  | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (default) |
| cuda11  | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`          |
| cpu     | `python:3.11-slim` **or** `ubuntu:22.04`                 |
| rocm    | `rocm/dev-ubuntu-22.04:6.1`  (future)                    |

Native builds (`llama_cpp`, `bitnet`) require the matching `-devel` image
for the `builder` stage. CPU variants must drop `-DGGML_CUDA=ON` from the
cmake invocation (flip `ENABLE_CUDA=OFF` via `--build-arg`).

---

## 7. CI pipeline — TODO for the human

The intended GitHub Actions workflow (`.github/workflows/backend-images.yml`)
has not been written yet. Skeleton to implement:

```yaml
name: build-backend-images
on:
  push:
    branches: [main]
    paths:
      - 'backends/dockerfiles/**'
      - 'backend/workers/**'
      - 'backend/scripts/build_bitnet.sh'
      - '.github/workflows/backend-images.yml'
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-22.04
    strategy:
      fail-fast: false
      matrix:
        backend:
          - vllm
          - whisper
          - diffusers
          - tts
          - sglang
          - voxtral
          - chatterbox
          - acestep
          - llama_cpp
          - bitnet
          # - tensorrt_llm   # enable once the TODO in its Dockerfile is resolved
        variant:
          - cuda12
          # - cpu          # enable once per-backend CPU base is validated
        include:
          - variant: cuda12
            base_image: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
            base_image_devel: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Resolve base image
        id: base
        run: |
          case "${{ matrix.backend }}" in
            llama_cpp|bitnet|tensorrt_llm)
              echo "image=${{ matrix.base_image_devel }}" >> $GITHUB_OUTPUT ;;
            *)
              echo "image=${{ matrix.base_image }}" >> $GITHUB_OUTPUT ;;
          esac
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: backends/dockerfiles/Dockerfile.${{ matrix.backend }}
          build-args: |
            BASE_IMAGE=${{ steps.base.outputs.image }}
          push: true
          tags: |
            ghcr.io/ocabra/backend-${{ matrix.backend }}:latest-${{ matrix.variant }}
            ghcr.io/ocabra/backend-${{ matrix.backend }}:${{ github.sha }}-${{ matrix.variant }}
          cache-from: type=gha,scope=${{ matrix.backend }}-${{ matrix.variant }}
          cache-to:   type=gha,scope=${{ matrix.backend }}-${{ matrix.variant }},mode=max
```

Items still open for the CI pipeline:
- Disk pressure: ubuntu-22.04 runners only have ~14 GB free. The larger
  images (vllm/sglang/voxtral/tensorrt_llm) probably need `actions/runner`
  self-hosted or a disk-cleanup step.
- Signed attestations / SBOM via `cosign` — not required for v1.
- Digest pinning: after each push, write the resulting digest into
  `docs/tasks/modular-backends-digests.json` for reproducible installs.
