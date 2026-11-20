#!/usr/bin/env python3
"""Worker de generación de vídeo que conduce un ComfyUI sin interfaz.

Por qué ComfyUI y no ``ltx-pipelines`` o Diffusers: los pesos int8 «convrot»
de LTX-2.5 (21,5 GB el DiT + 15,4 GB el codificador de texto) solo los cargan
los kernels de ComfyUI (``comfy-kitchen``). El camino de PyTorch exige los
bf16 (68 GB) y convertirlos a fp8 al vuelo, que en una 3090 no acelera nada.
Medido en una RTX 3090 con este mismo workflow: 1280x704, 5 s con audio en
46 s en caliente y 89 s la primera vez (carga de pesos incluida).

El worker arranca su propio ComfyUI en un puerto local, le manda el workflow
en formato API con los parámetros de la petición y devuelve el MP4. No guarda
nada entre peticiones: el fichero se borra en cuanto se ha leído.

Contrato:
  POST /generate  JSON {prompt, negative_prompt?, width, height, seconds, fps?,
                  seed?, job_id?} -> video/mp4 con cabeceras x-ocabra-*
  GET  /progress  -> {"job_id", "percent", "stage"} de la generación en curso
  POST /free      -> descarga los pesos de la GPU sin parar el proceso
  GET  /health    -> {"ok": bool}
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Fracción del progreso que se atribuye a cada muestreador del workflow de
# LTX-2.5: la primera pasada (8 pasos a media resolución) es casi todo el
# trabajo; el refinado tras el escalado latente son 3 pasos; el resto es
# codificar el texto y decodificar vídeo y audio.
_STAGE_WEIGHTS = ((5.0, 70.0), (75.0, 15.0))
_POLL_S = 0.5


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, "%s %s", event, detail)


def _die_with_parent() -> None:
    """Que ComfyUI muera si muere el worker (PR_SET_PDEATHSIG).

    oCabra termina al worker, no a sus hijos: sin esto un ComfyUI huérfano se
    quedaría con 24 GB de la GPU sin que nadie lo contabilice.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class WorkflowTemplate:
    """Workflow en formato API con los nodos que rellena el worker."""

    def __init__(self, path: Path) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.params: dict[str, str] = doc["params"]
        self.prompt: dict[str, Any] = doc["prompt"]
        samplers = [
            node_id
            for node_id, node in self.prompt.items()
            if node["class_type"] == "SamplerCustomAdvanced"
        ]
        # Orden de ejecución: el muestreador cuyo latente sale de otro
        # muestreador va después.
        self.samplers = sorted(samplers, key=lambda n: self._depends_on_sampler(n, samplers))

    def _depends_on_sampler(self, node_id: str, samplers: list[str]) -> int:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            for value in self.prompt.get(current, {}).get("inputs", {}).values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    src = value[0]
                    if src in samplers and src != node_id:
                        return 1
                    if src not in seen:
                        seen.add(src)
                        stack.append(src)
        return 0

    def render(self, req: dict[str, Any], prefix: str) -> tuple[dict[str, Any], int]:
        prompt = json.loads(json.dumps(self.prompt))
        p = self.params

        def put(name: str, field: str, value: Any) -> None:
            node_id = p.get(name)
            if node_id and node_id in prompt and value is not None:
                prompt[node_id]["inputs"][field] = value

        seed = req.get("seed")
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**48)
        put("prompt", "value", str(req["prompt"]))
        if req.get("negative_prompt"):
            put("negative_prompt", "text", str(req["negative_prompt"]))
        put("width", "value", int(req["width"]))
        put("height", "value", int(req["height"]))
        put("seconds", "value", int(req["seconds"]))
        put("fps", "value", int(req.get("fps") or 24))
        put("seed", "noise_seed", int(seed))
        put("output", "filename_prefix", prefix)
        return prompt, int(seed)


class WorkerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_id: str = args.model_id
        self.model_path = Path(args.model_path)
        self.src_dir = Path(args.src_dir)
        self.comfy_port: int = args.comfy_port
        self.base = f"http://127.0.0.1:{args.comfy_port}"
        self.template = WorkflowTemplate(Path(args.workflow))
        self.work_dir = Path(tempfile.mkdtemp(prefix="ocabra_comfyui_video_"))
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()
        self.progress: dict[str, Any] = {"job_id": None, "percent": 0.0, "stage": "idle"}
        self.extra_args: list[str] = args.comfy_arg or []

    # ── ComfyUI ────────────────────────────────────────────────────────
    def start_comfyui(self) -> None:
        paths = self.work_dir / "extra_model_paths.yaml"
        # Misma estructura que el repo de LTX-2.5 en Hugging Face.
        paths.write_text(
            "ocabra_video:\n"
            f"  base_path: {self.model_path}\n"
            "  diffusion_models: diffusion_models\n"
            "  text_encoders: text_encoders\n"
            "  vae: vae\n"
            "  latent_upscale_models: latent_upscale_models\n"
            "  loras: loras\n"
            "  model_patches: model_patches\n",
            encoding="utf-8",
        )
        for sub in ("output", "input", "temp", "user"):
            (self.work_dir / sub).mkdir(exist_ok=True)
        cmd = [
            sys.executable,
            str(self.src_dir / "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(self.comfy_port),
            "--output-directory", str(self.work_dir / "output"),
            "--input-directory", str(self.work_dir / "input"),
            "--temp-directory", str(self.work_dir / "temp"),
            "--user-directory", str(self.work_dir / "user"),
            "--extra-model-paths-config", str(paths),
            "--disable-auto-launch",
            "--disable-all-custom-nodes",
            "--preview-method", "none",
            *self.extra_args,
        ]  # fmt: skip
        _log("comfyui_starting", port=self.comfy_port, cmd=" ".join(cmd))
        self.process = subprocess.Popen(  # noqa: S603 — argumentos fijos
            cmd, cwd=str(self.src_dir), preexec_fn=_die_with_parent
        )

    def stop_comfyui(self) -> None:
        proc, self.process = self.process, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    async def comfy_alive(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base}/system_stats")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def vram_used_mb(self) -> int:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                stats = (await client.get(f"{self.base}/system_stats")).json()
            dev = stats["devices"][0]
            return int((dev["vram_total"] - dev["vram_free"]) / 2**20)
        except Exception:  # noqa: BLE001 — solo informativo
            return 0

    # ── Generación ─────────────────────────────────────────────────────
    def _set_progress(self, job_id: str, percent: float, stage: str) -> None:
        self.progress = {"job_id": job_id, "percent": round(percent, 1), "stage": stage}

    async def generate(self, req: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        job_id = str(req.get("job_id") or uuid.uuid4().hex)
        prefix = f"ocabra/{job_id}"
        prompt, seed = self.template.render(req, prefix)
        client_id = uuid.uuid4().hex
        started = time.monotonic()
        self._set_progress(job_id, 1.0, "queued")
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self.base}/prompt", json={"prompt": prompt, "client_id": client_id}
            )
            if r.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"ComfyUI rejected the workflow: {r.text[:500]}"
                )
            prompt_id = r.json()["prompt_id"]
            progress_task = asyncio.create_task(self._follow_progress(job_id, client_id))
            try:
                history = await self._wait_history(client, prompt_id)
            finally:
                progress_task.cancel()

        status = history.get("status", {})
        if status.get("status_str") != "success":
            message, oom = "generation failed", False
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    message = (
                        f"{data.get('node_type')}: {data.get('exception_message', '')}".strip()
                    )
                    oom = "out of memory" in message.lower() or data.get(
                        "exception_type", ""
                    ).endswith("OutOfMemoryError")
            self._set_progress(job_id, 0.0, "failed")
            raise HTTPException(status_code=507 if oom else 500, detail=message[:1000])

        video = self._find_output(history)
        body = video.read_bytes()
        meta = video.with_suffix(".json")
        video.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        elapsed = time.monotonic() - started
        frames = int(req["seconds"]) * int(req.get("fps") or 24) + 1
        self._set_progress(job_id, 100.0, "completed")
        stats = {
            "width": req["width"],
            "height": req["height"],
            "seconds": req["seconds"],
            "fps": int(req.get("fps") or 24),
            "frames": frames,
            "seed": seed,
            "generation-s": round(elapsed, 2),
            "vram-used-mb": await self.vram_used_mb(),
        }
        _log("comfyui_video_done", job_id=job_id, **stats)
        return body, {f"x-ocabra-{k}": str(v) for k, v in stats.items()}

    async def _wait_history(self, client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
        while True:
            if self.process is None or self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="ComfyUI exited during generation")
            r = await client.get(f"{self.base}/history/{prompt_id}")
            data = r.json() if r.status_code == 200 else {}
            entry = data.get(prompt_id)
            if entry and entry.get("status", {}).get("completed") is not None:
                if entry["status"].get("status_str") in ("success", "error"):
                    return entry
            await asyncio.sleep(_POLL_S)

    def _find_output(self, history: dict[str, Any]) -> Path:
        for output in history.get("outputs", {}).values():
            for item in output.get("images", []) + output.get("videos", []):
                name = item.get("filename", "")
                if name.endswith(".mp4"):
                    path = self.work_dir / "output" / item.get("subfolder", "") / name
                    if path.exists():
                        return path
        raise HTTPException(status_code=500, detail="ComfyUI produced no video file")

    async def _follow_progress(self, job_id: str, client_id: str) -> None:
        """Progreso por el websocket de ComfyUI (eventos ``progress``)."""
        import aiohttp  # dependencia de ComfyUI, está en el venv del backend

        stages = {
            node: weights
            for node, weights in zip(self.template.samplers, _STAGE_WEIGHTS, strict=False)
        }
        url = f"ws://127.0.0.1:{self.comfy_port}/ws?clientId={client_id}"
        try:
            async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(msg.data)
                    data = event.get("data", {})
                    if event.get("type") == "progress" and data.get("node") in stages:
                        base, span = stages[data["node"]]
                        frac = data["value"] / max(1, data["max"])
                        self._set_progress(job_id, base + span * frac, "sampling")
                    elif event.get("type") == "executing" and data.get("node"):
                        node = data["node"]
                        if node not in stages and self.progress["percent"] >= 90:
                            self._set_progress(
                                job_id, max(self.progress["percent"], 92), "decoding"
                            )
                        elif node not in stages and self.progress["percent"] < 5:
                            self._set_progress(job_id, 3, "encoding_prompt")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — el progreso es opcional
            _log("progress_stream_lost", logging.WARNING, error=str(exc))


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI(title="oCabra ComfyUI video worker")

    @app.post("/generate")
    async def generate(req: dict[str, Any]) -> Response:
        for field in ("prompt", "width", "height", "seconds"):
            if req.get(field) in (None, ""):
                raise HTTPException(status_code=400, detail=f"'{field}' is required")
        if not await state.comfy_alive():
            raise HTTPException(status_code=503, detail="ComfyUI is not running")
        async with state.lock:
            body, headers = await state.generate(req)
        return Response(content=body, media_type="video/mp4", headers=headers)

    @app.get("/progress")
    async def progress() -> dict[str, Any]:
        return state.progress

    @app.post("/free")
    async def free() -> dict[str, bool]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{state.base}/free", json={"unload_models": True, "free_memory": True}
            )
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": await state.comfy_alive()}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oCabra ComfyUI video worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True, help="Carpeta de modelos de vídeo")
    parser.add_argument("--src-dir", required=True, help="Clon de ComfyUI")
    parser.add_argument("--workflow", required=True, help="Workflow en formato API (JSON)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--comfy-port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--comfy-arg",
        action="append",
        help="Opción extra para ComfyUI; repetir por opción (p. ej. --comfy-arg=--lowvram)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    args = parse_args()
    state = WorkerState(args)
    state.start_comfyui()

    def _shutdown(signum: int, _frame: Any) -> None:
        state.stop_comfyui()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="info")
    finally:
        state.stop_comfyui()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
