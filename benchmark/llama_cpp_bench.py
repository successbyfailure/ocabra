#!/usr/bin/env python3
"""Quick benchmark harness for the new oCabra llama.cpp loader.

Spawns ``llama-server`` for each configured model+config, hits the OpenAI
chat completions endpoint single-shot and in parallel, then measures
TTFT, decode tokens/s, aggregate throughput and observed VRAM. Writes
results to /tmp/llamacpp_bench_results.json.

Designed to run from inside the api container (has the binary + GPU
access). Loads each model from the Ollama-shared blob store.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

LLAMA_SERVER = "/data/backends/llama_cpp/bin/llama-server"
OLLAMA_BLOBS = Path("/data/ollama_models/blobs")
PORT = 18800
HOST = "127.0.0.1"


@dataclass
class ModelSpec:
    name: str
    blob_digest: str  # sha256 hex (no prefix)
    gpu_layers: int = 99
    ctx_size: int = 16384
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 8
    flash_attn: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    parallel_slots: int = 4
    cont_batching: bool = True
    cuda_visible_devices: str = "1"  # default 3090
    main_gpu: int | None = None
    tensor_split: str | None = None  # CSV like "1,2"
    n_cpu_moe: int | None = None


@dataclass
class BenchResult:
    model: str
    cuda_visible_devices: str
    ctx_size: int
    parallel_slots: int
    cache_type_kv: str
    load_time_s: float | None = None
    vram_after_load_mb: dict[int, int] = field(default_factory=dict)
    single_ttft_ms: float | None = None
    single_decode_tps: float | None = None
    parallel_aggregate_tps: float | None = None
    parallel_per_request_tps: float | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def _read_vram_mb() -> dict[int, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return {
            int(line.split(",")[0]): int(line.split(",")[1])
            for line in out.splitlines()
            if line.strip()
        }
    except Exception:
        return {}


def _start_server(spec: ModelSpec) -> subprocess.Popen:
    blob = OLLAMA_BLOBS / f"sha256-{spec.blob_digest}"
    if not blob.is_file():
        raise FileNotFoundError(f"blob missing: {blob}")
    cmd = [
        LLAMA_SERVER,
        "--model", str(blob),
        "--host", HOST,
        "--port", str(PORT),
        "--ctx-size", str(spec.ctx_size),
        "--batch-size", str(spec.batch_size),
        "--ubatch-size", str(spec.ubatch_size),
        "--n-gpu-layers", str(spec.gpu_layers),
        "--threads", str(spec.threads),
        "--alias", spec.name,
        "--parallel", str(spec.parallel_slots),
        "--cache-type-k", spec.cache_type_k,
        "--cache-type-v", spec.cache_type_v,
    ]
    if spec.flash_attn:
        cmd.extend(["--flash-attn", "on"])
    if spec.cont_batching:
        cmd.append("--cont-batching")
    if spec.main_gpu is not None:
        cmd.extend(["--main-gpu", str(spec.main_gpu)])
    if spec.tensor_split:
        cmd.extend(["--tensor-split", spec.tensor_split])
    if spec.n_cpu_moe is not None:
        cmd.extend(["--n-cpu-moe", str(spec.n_cpu_moe)])

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": spec.cuda_visible_devices,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # The freshly-built llama-server links against shared libs in its own
        # bin dir plus the system cuBLAS under /usr/local/cuda.
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/data/backends/llama_cpp/bin",
    }
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _wait_healthy(timeout_s: int = 240) -> tuple[bool, str | None]:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    with httpx.Client(timeout=3.0) as c:
        while time.time() < deadline:
            try:
                r = c.get(f"http://{HOST}:{PORT}/health")
                if r.status_code == 200:
                    return True, None
                last_err = f"status={r.status_code}"
            except Exception as exc:
                last_err = type(exc).__name__
            time.sleep(1)
    return False, last_err


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def _single_completion(prompt: str, max_tokens: int = 192) -> tuple[float, int]:
    """Returns (ttft_seconds, tokens_decoded). Streams to measure TTFT."""
    t0 = time.time()
    first_tok_at: float | None = None
    tokens = 0
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    with httpx.Client(timeout=300.0) as c:
        with c.stream(
            "POST", f"http://{HOST}:{PORT}/v1/chat/completions", json=body
        ) as resp:
            for chunk in resp.iter_lines():
                if not chunk or not chunk.startswith("data:"):
                    continue
                payload = chunk[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {})
                # Reasoning models (qwen3, deepseek-r1) emit ``reasoning_content``
                # before/instead of ``content``; both consume compute and count
                # as generated tokens for throughput purposes.
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if first_tok_at is None:
                        first_tok_at = time.time()
                    tokens += 1
    if first_tok_at is None:
        first_tok_at = time.time()
    return first_tok_at - t0, tokens


def _bench_single(spec: ModelSpec, result: BenchResult) -> None:
    prompts = [
        "Explain quicksort in 3 short paragraphs.",
        "Write a Python function that reverses a string and explain it.",
        "Summarize the philosophy of stoicism in 5 bullet points.",
    ]
    ttfts: list[float] = []
    tps: list[float] = []
    for p in prompts:
        t0 = time.time()
        ttft, tokens = _single_completion(p, max_tokens=192)
        elapsed = time.time() - t0
        decode_time = max(elapsed - ttft, 1e-6)
        if tokens > 0:
            tps.append(tokens / decode_time)
        ttfts.append(ttft)
    if ttfts:
        result.single_ttft_ms = round(sum(ttfts) / len(ttfts) * 1000, 1)
    if tps:
        result.single_decode_tps = round(sum(tps) / len(tps), 2)


def _bench_parallel(spec: ModelSpec, result: BenchResult, n: int = 4) -> None:
    prompt = (
        "Write a 200-word technical explanation of how flash attention "
        "differs from standard attention in transformer models, focusing "
        "on memory bandwidth."
    )
    t0 = time.time()
    total_tokens = 0
    per_request_tps_list: list[float] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(_single_completion, prompt, 256) for _ in range(n)]
        for f in as_completed(futs):
            try:
                ttft, tokens = f.result()
                total_tokens += tokens
                per_request_tps_list.append(
                    tokens / max(time.time() - t0 - ttft, 1e-6)
                )
            except Exception as exc:
                result.notes.append(f"parallel request failed: {exc}")
    elapsed = time.time() - t0
    if total_tokens > 0:
        result.parallel_aggregate_tps = round(total_tokens / elapsed, 2)
    if per_request_tps_list:
        result.parallel_per_request_tps = round(
            sum(per_request_tps_list) / len(per_request_tps_list), 2
        )


def run_bench(spec: ModelSpec) -> BenchResult:
    result = BenchResult(
        model=spec.name,
        cuda_visible_devices=spec.cuda_visible_devices,
        ctx_size=spec.ctx_size,
        parallel_slots=spec.parallel_slots,
        cache_type_kv=f"{spec.cache_type_k}/{spec.cache_type_v}",
    )
    vram_before = _read_vram_mb()
    proc = _start_server(spec)
    try:
        t0 = time.time()
        ok, err = _wait_healthy(timeout_s=300)
        result.load_time_s = round(time.time() - t0, 1)
        if not ok:
            result.error = f"server failed to become healthy: {err}"
            return result
        vram_after = _read_vram_mb()
        result.vram_after_load_mb = {
            idx: max(0, vram_after.get(idx, 0) - vram_before.get(idx, 0))
            for idx in vram_after
        }
        try:
            _bench_single(spec, result)
        except Exception as exc:
            result.notes.append(f"single bench failed: {exc!r}")
        try:
            _bench_parallel(spec, result)
        except Exception as exc:
            result.notes.append(f"parallel bench failed: {exc!r}")
    finally:
        _kill(proc)
        time.sleep(2)
    return result


def _resolve_blob_digest(model: str, tag: str) -> str:
    manifest_path = (
        Path("/data/ollama_models/manifests/registry.ollama.ai/library")
        / model
        / tag
    )
    payload = json.loads(manifest_path.read_text())
    for layer in payload["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return layer["digest"].split(":", 1)[1]
    raise RuntimeError(f"no model layer in {manifest_path}")


SPECS = [
    # qwen3:32b dense — fits 3090 alone with KV-quant + moderate context.
    # Best single-GPU coding model in the working set.
    ModelSpec(
        name="qwen3-32b",
        blob_digest=_resolve_blob_digest("qwen3", "32b"),
        ctx_size=16384,
        cuda_visible_devices="1",
        parallel_slots=4,
    ),
    # qwen3-coder:30b MoE (30B params, 3B active per token).
    # Drastically cheaper per token than dense 32B, ideal for agentic loops.
    # Multi-GPU split because raw weights still ~17 GB and we want headroom.
    ModelSpec(
        name="qwen3-coder-30b-moe",
        blob_digest=_resolve_blob_digest("qwen3-coder", "30b"),
        ctx_size=32768,
        cuda_visible_devices="0,1",
        tensor_split="1,2",
        parallel_slots=8,
    ),
    # qwen3:32b multi-GPU baseline for comparison with coder MoE.
    ModelSpec(
        name="qwen3-32b-split",
        blob_digest=_resolve_blob_digest("qwen3", "32b"),
        ctx_size=32768,
        cuda_visible_devices="0,1",
        tensor_split="1,2",
        parallel_slots=4,
    ),
    # mistral:7b — small/fast baseline reference for the harness.
    ModelSpec(
        name="mistral-7b",
        blob_digest=_resolve_blob_digest("mistral", "7b"),
        ctx_size=16384,
        cuda_visible_devices="0",
        parallel_slots=4,
    ),
]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    results: list[BenchResult] = []
    for spec in SPECS:
        if target and spec.name != target:
            continue
        print(f"\n=== {spec.name} (ctx={spec.ctx_size}, slots={spec.parallel_slots}, gpus={spec.cuda_visible_devices}, KV={spec.cache_type_k}/{spec.cache_type_v}) ===", flush=True)
        try:
            r = run_bench(spec)
        except Exception as exc:
            r = BenchResult(
                model=spec.name,
                cuda_visible_devices=spec.cuda_visible_devices,
                ctx_size=spec.ctx_size,
                parallel_slots=spec.parallel_slots,
                cache_type_kv=f"{spec.cache_type_k}/{spec.cache_type_v}",
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(r)
        print(json.dumps(asdict(r), indent=2), flush=True)
    Path("/tmp/llamacpp_bench_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
