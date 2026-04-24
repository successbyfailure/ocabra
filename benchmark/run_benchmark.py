#!/usr/bin/env python3
"""
oCabra Backend Benchmark — Qwen3-8B 4-bit across all backends.

Measures: load time, TTFT, output tok/s, inter-token latency, total latency,
          VRAM usage, and tokens generated.

Usage (from inside ocabra-api-1 container):
    python3 /app/benchmark/run_benchmark.py --backend vllm
    python3 /app/benchmark/run_benchmark.py --backend sglang
    python3 /app/benchmark/run_benchmark.py --backend llama_cpp
    python3 /app/benchmark/run_benchmark.py --backend ollama

Or from host via docker exec:
    docker exec ocabra-api-1 python3 /app/benchmark/run_benchmark.py --backend vllm
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_PORT = 18900  # dedicated port to avoid clashes with running models
GPU_ID = 1  # RTX 3090 (24 GB) — index 1

# Model paths (inside container)
MODELS = {
    "vllm": "/data/models/huggingface/Qwen--Qwen3-8B-AWQ",
    "sglang": "/data/models/huggingface/Qwen--Qwen3-8B-AWQ",
    "llama_cpp": "/data/hf_cache/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf",
    "ollama": "qwen3:8b",
    "tensorrt": "/data/models/huggingface/Qwen--Qwen3-8B-AWQ",
}

# TensorRT-LLM config (Docker-in-Docker)
TRTLLM_DOCKER_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:latest"
TRTLLM_HOST_MODELS = "/docker/ai-models/ocabra/models"
TRTLLM_HOST_HF_CACHE = "/docker/ai-models/ocabra/hf_cache"

MODEL_DISPLAY = "Qwen3-8B (4-bit)"

# Prompts — short, medium, long input to test different scenarios
PROMPTS = [
    {
        "name": "short",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ],
        "max_tokens": 64,
    },
    {
        "name": "medium",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Be concise but thorough.",
            },
            {
                "role": "user",
                "content": (
                    "Explain the differences between TCP and UDP protocols. "
                    "Cover reliability, ordering, speed, and typical use cases."
                ),
            },
        ],
        "max_tokens": 512,
    },
    {
        "name": "long_gen",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed technical tutorial about building a REST API "
                    "with Python FastAPI, including authentication, database models "
                    "with SQLAlchemy, error handling, and deployment with Docker. "
                    "Include code examples for each section."
                ),
            },
        ],
        "max_tokens": 1024,
    },
]

# Common generation params for fairness
GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    prompt_name: str = ""
    ttft_ms: float = 0.0          # time to first token
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    output_tok_per_sec: float = 0.0
    inter_token_latency_ms: float = 0.0  # average
    first_tokens: str = ""        # first ~50 chars of output for sanity check


@dataclass
class ConcurrentResult:
    """Aggregated result from N concurrent requests."""
    prompt_name: str = ""
    concurrency: int = 0
    # Per-request stats (lists, one entry per request)
    individual: list[PromptResult] = field(default_factory=list)
    # Aggregated
    wall_time_ms: float = 0.0       # wall clock from first launch to last finish
    total_tokens: int = 0
    system_tok_per_sec: float = 0.0  # total tokens / wall time (system throughput)
    avg_ttft_ms: float = 0.0
    avg_tok_per_sec: float = 0.0     # average per-request tok/s
    avg_itl_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    # Degradation vs single-request baseline
    ttft_degradation: float = 0.0    # multiplier (2.0 = 2× slower than single)
    tps_degradation: float = 0.0     # fraction (0.5 = half the single-request speed)


@dataclass
class BenchmarkResult:
    backend: str = ""
    model: str = MODEL_DISPLAY
    gpu: str = ""
    load_time_s: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_model_mb: float = 0.0
    concurrency: int = 1
    prompts: list[PromptResult] = field(default_factory=list)
    concurrent: list[ConcurrentResult] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_vram_used_mb(gpu_id: int = GPU_ID) -> float:
    """Get VRAM used on a specific GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(out.strip())
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Server lifecycle per backend
# ---------------------------------------------------------------------------

def find_free_port(start: int = BENCHMARK_PORT) -> int:
    """Find a free port starting from `start`."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")


def wait_for_health(url: str, timeout: int = 180, interval: float = 1.0) -> float:
    """Poll health endpoint. Returns seconds until healthy."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return time.monotonic() - t0
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(interval)
    raise TimeoutError(f"Health check failed after {timeout}s: {url}")


class BackendServer:
    """Manages a backend server process for benchmarking."""

    def __init__(self, backend: str, port: int):
        self.backend = backend
        self.port = port
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"
        self.health_url = f"{self.base_url}/health"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.model_name = "qwen3-8b-bench"

    def start(self) -> float:
        """Start the server, return load time in seconds."""
        model_path = MODELS[self.backend]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(GPU_ID),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }

        if self.backend == "vllm":
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--served-model-name", self.model_name,
                "--gpu-memory-utilization", "0.85",
                "--enforce-eager",
                "--max-model-len", "4096",
                "--dtype", "auto",
                "--quantization", "awq",
                "--tool-call-parser", "qwen3_json",
                "--reasoning-parser", "qwen3",
            ]

        elif self.backend == "sglang":
            cmd = [
                "/opt/sglang-venv/bin/python", "-m", "sglang.launch_server",
                "--model-path", model_path,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--served-model-name", self.model_name,
                "--mem-fraction-static", "0.85",
                "--context-length", "4096",
                "--quantization", "awq",
            ]

        elif self.backend == "llama_cpp":
            # Use CUDA-enabled build if available
            cuda_server = "/tmp/llama-cpp-cuda/build/bin/llama-server"
            server_bin = cuda_server if os.path.exists(cuda_server) else "llama-server"
            if server_bin == cuda_server:
                env["LD_LIBRARY_PATH"] = (
                    "/tmp/llama-cpp-cuda/build/bin:"
                    + env.get("LD_LIBRARY_PATH", "")
                )
            cmd = [
                server_bin,
                "--model", model_path,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--alias", self.model_name,
                "--ctx-size", "4096",
                "--n-gpu-layers", "99",
                "--batch-size", "512",
                "--ubatch-size", "128",
            ]
            # llama-server doesn't use CUDA_VISIBLE_DEVICES the same way
            # but --n-gpu-layers sends all to the visible GPU

        elif self.backend == "tensorrt":
            return self._start_tensorrt(env)

        elif self.backend == "ollama":
            # Ollama: no process to spawn, just load the model
            return self._start_ollama()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print(f"  CMD: {' '.join(cmd[:6])}...")
        log_path = f"/tmp/bench_{self.backend}.log"
        self._log_file = open(log_path, "w")
        print(f"  Log: {log_path}")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Wait for health
        load_time = wait_for_health(self.health_url, timeout=180)
        return load_time

    def _start_ollama(self) -> float:
        """Load model into Ollama and measure load time."""
        ollama_url = "http://ollama:11434"
        self.chat_url = f"{ollama_url}/v1/chat/completions"
        self.model_name = MODELS["ollama"]

        # First unload any existing model
        try:
            httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=30,
            )
            time.sleep(2)
        except Exception:
            pass

        # Load model by sending empty prompt
        t0 = time.monotonic()
        r = httpx.post(
            f"{ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_gpu": 99},
            },
            timeout=180,
        )
        load_time = time.monotonic() - t0
        if r.status_code != 200:
            raise RuntimeError(f"Ollama load failed: {r.status_code} {r.text[:200]}")
        return load_time

    def _start_tensorrt(self, env: dict) -> float:
        """Start TRT-LLM via Docker-in-Docker.

        Uses trtllm-serve which can accept a HF checkpoint directly
        (compiles to TRT engine on first run).
        """
        model_path = MODELS["tensorrt"]
        # Map container path → host path for Docker mount
        host_model_path = model_path.replace("/data/models", TRTLLM_HOST_MODELS)
        if model_path.startswith("/data/hf_cache"):
            host_model_path = model_path.replace("/data/hf_cache", TRTLLM_HOST_HF_CACHE)

        self._trt_container_name = f"bench-trtllm-{self.port}"
        hostname = os.environ.get("HOSTNAME", "")

        cmd = [
            "docker", "run", "--rm",
            f"--name={self._trt_container_name}",
            "--gpus", f'"device={GPU_ID}"',
            "--ipc=host",
            f"--network=container:{hostname}",
            "-v", f"{TRTLLM_HOST_MODELS}:{TRTLLM_HOST_MODELS}",
            "-v", f"{TRTLLM_HOST_HF_CACHE}:{TRTLLM_HOST_HF_CACHE}",
            TRTLLM_DOCKER_IMAGE,
            "trtllm-serve", "serve",
            host_model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--backend", "trt",
            "--max_batch_size", "4",
            "--max_seq_len", "4096",
        ]

        print(f"  CMD: docker run ... trtllm-serve serve {host_model_path}")
        log_path = f"/tmp/bench_{self.backend}.log"
        self._log_file = open(log_path, "w")
        print(f"  Log: {log_path}")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        # TRT-LLM takes longer to start (engine loading + warmup)
        load_time = wait_for_health(self.health_url, timeout=300)
        return load_time

    def stop(self):
        """Stop the server process."""
        if self.backend == "ollama":
            # Unload from Ollama
            try:
                httpx.post(
                    "http://ollama:11434/api/generate",
                    json={"model": MODELS["ollama"], "keep_alive": 0},
                    timeout=30,
                )
            except Exception:
                pass
            return

        if self.backend == "tensorrt":
            # Stop the Docker container
            container = getattr(self, "_trt_container_name", None)
            if container:
                try:
                    subprocess.run(
                        ["docker", "stop", container],
                        timeout=30, capture_output=True,
                    )
                except Exception:
                    try:
                        subprocess.run(
                            ["docker", "kill", container],
                            timeout=10, capture_output=True,
                        )
                    except Exception:
                        pass
            if self.process:
                self.process.wait(timeout=10)
                self.process = None
            return

        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=10)
                except Exception:
                    pass
            self.process = None


# ---------------------------------------------------------------------------
# Inference measurement
# ---------------------------------------------------------------------------

def run_prompt(
    chat_url: str, model_name: str, prompt: dict, backend: str = "vllm",
) -> PromptResult:
    """Run a single prompt with streaming, measuring all timing metrics."""

    # Ollama native API provides its own precise timing — use it.
    if backend == "ollama":
        return _run_prompt_ollama(model_name, prompt)

    return _run_prompt_openai(chat_url, model_name, prompt, backend)


def _run_prompt_ollama(model_name: str, prompt: dict) -> PromptResult:
    """Run prompt via Ollama native /api/chat with think:false and streaming."""
    result = PromptResult(prompt_name=prompt["name"])
    ollama_url = "http://ollama:11434/api/chat"

    payload = {
        "model": model_name,
        "messages": prompt["messages"],
        "stream": True,
        "think": False,
        "options": {
            "num_predict": prompt["max_tokens"],
            "temperature": GEN_PARAMS["temperature"],
            "top_p": GEN_PARAMS["top_p"],
        },
    }

    token_times: list[float] = []
    chunks_text: list[str] = []
    t_start = time.monotonic()
    t_first_token: float | None = None
    final_data: dict = {}

    try:
        with httpx.stream("POST", ollama_url, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})
                content = msg.get("content", "")

                if content:
                    now = time.monotonic()
                    if t_first_token is None:
                        t_first_token = now
                    token_times.append(now)
                    chunks_text.append(content)

                if chunk.get("done"):
                    final_data = chunk

    except Exception as e:
        result.prompt_name = f"{prompt['name']} [ERROR: {e}]"
        return result

    t_end = time.monotonic()

    total_text = "".join(chunks_text)
    n_tokens = len(token_times)

    result.tokens_generated = final_data.get("eval_count", n_tokens)
    result.total_latency_ms = (t_end - t_start) * 1000
    result.first_tokens = total_text[:80].replace("\n", "\\n")

    # Use Ollama's own precise timing if available
    prompt_eval_ns = final_data.get("prompt_eval_duration", 0)
    eval_count = final_data.get("eval_count", 0)
    eval_ns = final_data.get("eval_duration", 0)

    if prompt_eval_ns:
        result.ttft_ms = prompt_eval_ns / 1_000_000  # ns → ms

    if eval_count > 0 and eval_ns > 0:
        result.output_tok_per_sec = eval_count / (eval_ns / 1_000_000_000)
        result.inter_token_latency_ms = (eval_ns / eval_count) / 1_000_000

    return result


def _run_prompt_openai(
    chat_url: str, model_name: str, prompt: dict, backend: str,
) -> PromptResult:
    """Run prompt via OpenAI-compatible streaming API (vLLM, SGLang, llama.cpp)."""
    result = PromptResult(prompt_name=prompt["name"])

    # Build messages — for llama.cpp prepend /no_think to suppress thinking.
    messages = []
    for msg in prompt["messages"]:
        if backend == "llama_cpp" and msg["role"] == "user":
            messages.append({**msg, "content": "/no_think " + msg["content"]})
        else:
            messages.append(msg)

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": prompt["max_tokens"],
        "stream": True,
        **GEN_PARAMS,
    }

    # vLLM/SGLang thinking suppression
    if backend in ("vllm", "sglang", "tensorrt"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    token_times: list[float] = []
    chunks_text: list[str] = []
    t_start = time.monotonic()
    t_first_token: float | None = None

    try:
        with httpx.stream(
            "POST", chat_url, json=payload, timeout=120.0
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")

                if content or reasoning:
                    now = time.monotonic()
                    if t_first_token is None:
                        t_first_token = now
                    token_times.append(now)
                    chunks_text.append(content or reasoning)

    except Exception as e:
        result.prompt_name = f"{prompt['name']} [ERROR: {e}]"
        return result

    t_end = time.monotonic()

    total_text = "".join(chunks_text)
    n_tokens = len(token_times)

    result.tokens_generated = n_tokens
    result.total_latency_ms = (t_end - t_start) * 1000
    result.first_tokens = total_text[:80].replace("\n", "\\n")

    if t_first_token is not None:
        result.ttft_ms = (t_first_token - t_start) * 1000

    if n_tokens > 1 and t_first_token is not None:
        gen_duration = t_end - t_first_token
        result.output_tok_per_sec = (n_tokens - 1) / gen_duration if gen_duration > 0 else 0

        # Inter-token latency (average gap between consecutive tokens)
        gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        result.inter_token_latency_ms = (sum(gaps) / len(gaps)) * 1000 if gaps else 0

    return result


# ---------------------------------------------------------------------------
# Concurrent benchmark
# ---------------------------------------------------------------------------

def run_concurrent_prompt(
    chat_url: str,
    model_name: str,
    prompt: dict,
    backend: str,
    concurrency: int,
    baseline: PromptResult | None = None,
) -> ConcurrentResult:
    """Launch `concurrency` identical requests in parallel via threads."""
    cr = ConcurrentResult(prompt_name=prompt["name"], concurrency=concurrency)

    def _worker(idx: int) -> PromptResult:
        r = run_prompt(chat_url, model_name, prompt, backend)
        r.prompt_name = f"{prompt['name']}_{idx}"
        return r

    t_wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, i) for i in range(concurrency)]
        for fut in as_completed(futures):
            cr.individual.append(fut.result())
    t_wall_end = time.monotonic()

    cr.wall_time_ms = (t_wall_end - t_wall_start) * 1000
    cr.total_tokens = sum(r.tokens_generated for r in cr.individual)

    valid = [r for r in cr.individual if r.tokens_generated > 0]
    if valid:
        cr.avg_ttft_ms = sum(r.ttft_ms for r in valid) / len(valid)
        cr.avg_tok_per_sec = sum(r.output_tok_per_sec for r in valid) / len(valid)
        cr.avg_itl_ms = sum(r.inter_token_latency_ms for r in valid) / len(valid)
        cr.avg_total_latency_ms = sum(r.total_latency_ms for r in valid) / len(valid)

    if cr.wall_time_ms > 0:
        cr.system_tok_per_sec = cr.total_tokens / (cr.wall_time_ms / 1000)

    if baseline and baseline.tokens_generated > 0:
        if baseline.ttft_ms > 0:
            cr.ttft_degradation = cr.avg_ttft_ms / baseline.ttft_ms
        if baseline.output_tok_per_sec > 0:
            cr.tps_degradation = cr.avg_tok_per_sec / baseline.output_tok_per_sec

    return cr


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(backend: str, runs: int = 1, concurrency: int = 1) -> BenchmarkResult:
    result = BenchmarkResult(backend=backend)
    port = find_free_port()
    server = BackendServer(backend, port)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {backend.upper()}")
    print(f"  Model: {MODEL_DISPLAY}")
    print(f"  GPU: #{GPU_ID} (RTX 3090 24GB)")
    print(f"  Port: {port}")
    print(f"{'='*60}\n")

    # Get GPU info
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={GPU_ID}", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        result.gpu = out.strip()
    except Exception:
        result.gpu = "unknown"

    # --- Phase 1: Load model ---
    print("[1/3] Loading model...")
    result.vram_before_mb = get_vram_used_mb()
    try:
        t0 = time.monotonic()
        health_time = server.start()
        result.load_time_s = time.monotonic() - t0
        print(f"  ✓ Model loaded in {result.load_time_s:.1f}s (health OK at {health_time:.1f}s)")
    except Exception as e:
        result.error = f"Load failed: {e}"
        print(f"  ✗ {result.error}")
        server.stop()
        return result

    time.sleep(2)  # let VRAM stabilize
    result.vram_after_mb = get_vram_used_mb()
    result.vram_model_mb = result.vram_after_mb - result.vram_before_mb
    print(f"  VRAM: {result.vram_before_mb:.0f} → {result.vram_after_mb:.0f} MB "
          f"(model: {result.vram_model_mb:.0f} MB)")

    # --- Phase 2: Warmup ---
    print("\n[2/3] Warmup (1 request)...")
    warmup = run_prompt(server.chat_url, server.model_name, PROMPTS[0], backend)
    print(f"  Warmup done: {warmup.tokens_generated} tokens, "
          f"{warmup.output_tok_per_sec:.1f} tok/s")

    # --- Phase 3: Benchmark prompts ---
    print(f"\n[3/3] Running {len(PROMPTS)} prompts × {runs} run(s)...")

    for prompt in PROMPTS:
        best: PromptResult | None = None
        for run_i in range(runs):
            r = run_prompt(server.chat_url, server.model_name, prompt, backend)
            if best is None or r.output_tok_per_sec > best.output_tok_per_sec:
                best = r
            if runs > 1:
                print(f"    {prompt['name']} run {run_i+1}: "
                      f"{r.output_tok_per_sec:.1f} tok/s, TTFT={r.ttft_ms:.0f}ms")
        assert best is not None
        result.prompts.append(best)
        print(f"  ✓ {best.prompt_name:10s} | "
              f"TTFT={best.ttft_ms:7.0f}ms | "
              f"tok/s={best.output_tok_per_sec:6.1f} | "
              f"ITL={best.inter_token_latency_ms:6.1f}ms | "
              f"tokens={best.tokens_generated:4d} | "
              f"total={best.total_latency_ms:8.0f}ms")
        print(f"           → {best.first_tokens}")

    # --- Phase 4: Concurrent benchmark (if concurrency > 1) ---
    if concurrency > 1:
        result.concurrency = concurrency
        print(f"\n[4/4] Running {len(PROMPTS)} prompts × {concurrency} concurrent requests...")

        # Use single-request results as baseline for degradation calc
        baselines = {p.prompt_name: p for p in result.prompts}

        for prompt in PROMPTS:
            baseline = baselines.get(prompt["name"])
            cr = run_concurrent_prompt(
                server.chat_url, server.model_name, prompt, backend,
                concurrency, baseline,
            )
            result.concurrent.append(cr)
            print(f"  ✓ {cr.prompt_name:10s} | "
                  f"sys_tok/s={cr.system_tok_per_sec:6.1f} | "
                  f"avg_tok/s={cr.avg_tok_per_sec:6.1f} | "
                  f"avg_TTFT={cr.avg_ttft_ms:7.0f}ms | "
                  f"avg_ITL={cr.avg_itl_ms:6.1f}ms | "
                  f"wall={cr.wall_time_ms:8.0f}ms | "
                  f"total_tok={cr.total_tokens:5d}")
            if baseline and baseline.output_tok_per_sec > 0:
                print(f"           → TTFT ×{cr.ttft_degradation:.1f}  "
                      f"tok/s ×{cr.tps_degradation:.2f} vs single")

    # --- Cleanup ---
    print("\nStopping server...")
    server.stop()
    time.sleep(2)
    print("Done.\n")

    return result


def print_summary(result: BenchmarkResult):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {result.backend.upper()} — {result.model}")
    print(f"{'='*70}")
    print(f"  GPU:           {result.gpu}")
    print(f"  Load time:     {result.load_time_s:.1f}s")
    print(f"  VRAM (model):  {result.vram_model_mb:.0f} MB")
    print(f"  VRAM (total):  {result.vram_after_mb:.0f} MB")

    print(f"\n  --- Single request ---")
    print(f"  {'Prompt':<12s} {'TTFT':>8s} {'tok/s':>8s} {'ITL':>8s} "
          f"{'Tokens':>7s} {'Total':>9s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*9}")
    for p in result.prompts:
        print(f"  {p.prompt_name:<12s} {p.ttft_ms:>7.0f}ms {p.output_tok_per_sec:>7.1f} "
              f"{p.inter_token_latency_ms:>7.1f}ms {p.tokens_generated:>6d} "
              f"{p.total_latency_ms:>8.0f}ms")

    if result.concurrent:
        print(f"\n  --- Concurrent ({result.concurrency} requests) ---")
        print(f"  {'Prompt':<12s} {'sys_t/s':>8s} {'avg_t/s':>8s} {'avgTTFT':>8s} "
              f"{'avgITL':>7s} {'TotTok':>7s} {'Wall':>9s} {'TTFTx':>6s} {'TPSx':>6s}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*6} {'-'*6}")
        for cr in result.concurrent:
            print(f"  {cr.prompt_name:<12s} {cr.system_tok_per_sec:>7.1f} "
                  f"{cr.avg_tok_per_sec:>7.1f} {cr.avg_ttft_ms:>7.0f}ms "
                  f"{cr.avg_itl_ms:>6.1f}ms {cr.total_tokens:>6d} "
                  f"{cr.wall_time_ms:>8.0f}ms "
                  f"{cr.ttft_degradation:>5.1f}x {cr.tps_degradation:>5.2f}x")

    print()
    if result.error:
        print(f"  ⚠ Error: {result.error}")


def save_result(result: BenchmarkResult, output_dir: str = "/app/benchmark/results"):
    """Save result as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{result.backend}_{ts}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="oCabra backend benchmark")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["vllm", "sglang", "llama_cpp", "ollama", "tensorrt"],
        help="Backend to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per prompt (best of N, default=1)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=GPU_ID,
        help=f"GPU index to use (default={GPU_ID})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default=1, sequential only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port override (0=auto)",
    )
    args = parser.parse_args()

    # Update module-level config from args
    _update_config(args.gpu, args.port if args.port else BENCHMARK_PORT)

    result = run_benchmark(args.backend, runs=args.runs, concurrency=args.concurrency)
    print_summary(result)
    save_result(result)


def _update_config(gpu_id: int, port: int):
    global GPU_ID, BENCHMARK_PORT
    GPU_ID = gpu_id
    BENCHMARK_PORT = port


if __name__ == "__main__":
    main()
