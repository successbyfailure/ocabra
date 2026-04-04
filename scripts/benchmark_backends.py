#!/usr/bin/env python3
"""Benchmark: Ollama vs vLLM vs TensorRT-LLM

Measures throughput (tokens/s), latency (TTFT + total), and consistency
for each backend serving the same model (Qwen3-8B and Qwen3-32B).

Usage:
    python3 scripts/benchmark_backends.py --model 8b
    python3 scripts/benchmark_backends.py --model 32b
    python3 scripts/benchmark_backends.py --model 8b --model 32b --output results.json

Requirements:
    pip install aiohttp rich
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp


# ── Prompts ──────────────────────────────────────────────────────

PROMPTS = [
    "Explain the difference between TCP and UDP in two sentences.",
    "What is the capital of Australia?",
    "Write a Python function that checks if a number is prime.",
    "Translate 'Good morning, how are you?' to Spanish.",
    "What are the main benefits of containerization with Docker?",
    "Summarize the French Revolution in three sentences.",
    "What is the time complexity of quicksort in the average case?",
    "Give me a recipe for chocolate chip cookies in under 100 words.",
]

MAX_TOKENS = 200


# ── Model config ─────────────────────────────────────────────────

MODEL_CONFIG = {
    "8b": {
        "vllm":       "vllm/Qwen/Qwen3-8B",
        "trtllm":     "tensorrt_llm/Qwen3-8B-fp16",
        "ollama":     "qwen3:8b",
        "ocabra_url": "http://localhost:8484",
        "ollama_url": "http://localhost:7869",
    },
    "32b": {
        "vllm":       "vllm/Qwen/Qwen3-32B-AWQ",
        "trtllm":     "tensorrt_llm/Qwen3-32B-AWQ-tp2-fp16",
        "ollama":     "qwen3:32b",
        "ocabra_url": "http://localhost:8484",
        "ollama_url": "http://localhost:7869",
    },
}


# ── Result dataclasses ───────────────────────────────────────────

@dataclass
class RequestResult:
    backend: str
    model_size: str
    prompt_idx: int
    ttft_ms: float       # time to first token
    total_ms: float      # total generation time
    output_tokens: int
    tokens_per_sec: float
    error: str | None = None


@dataclass
class BenchmarkSummary:
    backend: str
    model_size: str
    n_success: int
    n_errors: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    tps_mean: float
    tps_p50: float
    tps_p95: float
    total_p50_ms: float


# ── Backend clients ──────────────────────────────────────────────

async def _ocabra_stream(session: aiohttp.ClientSession, base_url: str, model_id: str, prompt: str) -> AsyncIterator[str]:
    """Stream tokens via oCabra /v1/chat/completions (OpenAI SSE)."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                pass


async def _ollama_stream(session: aiohttp.ClientSession, base_url: str, model_name: str, prompt: str) -> AsyncIterator[str]:
    """Stream tokens via Ollama /api/chat (NDJSON).

    Yields both `content` and `thinking` tokens so thinking models (Qwen3, etc.)
    are measured correctly.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": MAX_TOKENS},
        "stream": True,
    }
    async with session.post(f"{base_url}/api/chat", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            line = line.decode().strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                # Yield content AND thinking tokens (Qwen3 puts thinking in "thinking" key)
                text = msg.get("content", "") or msg.get("thinking", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                pass


async def run_one(
    session: aiohttp.ClientSession,
    backend: str,
    model_size: str,
    prompt_idx: int,
    cfg: dict,
) -> RequestResult:
    prompt = PROMPTS[prompt_idx]
    t_start = time.perf_counter()
    t_first: float | None = None
    token_count = 0
    error: str | None = None

    try:
        if backend == "ollama":
            stream = _ollama_stream(session, cfg["ollama_url"], cfg["ollama"], prompt)
        else:
            model_id = cfg[backend]
            stream = _ocabra_stream(session, cfg["ocabra_url"], model_id, prompt)

        async for token in stream:
            if t_first is None:
                t_first = time.perf_counter()
            token_count += len(token.split())  # approximate word tokens

    except Exception as exc:
        error = str(exc)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    ttft_ms = ((t_first - t_start) * 1000) if t_first else total_ms
    tps = (token_count / (t_end - t_start)) if token_count > 0 else 0.0

    return RequestResult(
        backend=backend,
        model_size=model_size,
        prompt_idx=prompt_idx,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        output_tokens=token_count,
        tokens_per_sec=tps,
        error=error,
    )


async def run_backend(backend: str, model_size: str, cfg: dict, concurrency: int = 1) -> list[RequestResult]:
    """Run all prompts sequentially (concurrency=1) or in parallel."""
    results: list[RequestResult] = []
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if concurrency == 1:
            for i in range(len(PROMPTS)):
                r = await run_one(session, backend, model_size, i, cfg)
                status = f"✓ {r.tokens_per_sec:.1f} tps" if not r.error else f"✗ {r.error[:60]}"
                print(f"  [{backend:10}] prompt {i+1}/{len(PROMPTS)} → {status}")
                results.append(r)
        else:
            tasks = [run_one(session, backend, model_size, i, cfg) for i in range(len(PROMPTS))]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)
    return results


def summarize(results: list[RequestResult]) -> BenchmarkSummary:
    ok = [r for r in results if not r.error]
    err = [r for r in results if r.error]
    if not ok:
        return BenchmarkSummary(
            backend=results[0].backend, model_size=results[0].model_size,
            n_success=0, n_errors=len(err),
            ttft_p50_ms=0, ttft_p95_ms=0,
            tps_mean=0, tps_p50=0, tps_p95=0, total_p50_ms=0,
        )
    ttfts = sorted(r.ttft_ms for r in ok)
    tpss = sorted(r.tokens_per_sec for r in ok)
    totals = sorted(r.total_ms for r in ok)

    def p(lst, pct): return lst[min(int(len(lst) * pct / 100), len(lst) - 1)]

    return BenchmarkSummary(
        backend=results[0].backend,
        model_size=results[0].model_size,
        n_success=len(ok),
        n_errors=len(err),
        ttft_p50_ms=p(ttfts, 50),
        ttft_p95_ms=p(ttfts, 95),
        tps_mean=statistics.mean(r.tokens_per_sec for r in ok),
        tps_p50=p(tpss, 50),
        tps_p95=p(tpss, 95),
        total_p50_ms=p(totals, 50),
    )


def print_summary(summaries: list[BenchmarkSummary]) -> None:
    print("\n" + "=" * 72)
    print(f"{'BACKEND':<12} {'SIZE':<5} {'OK':>4} {'ERR':>4} {'TTFT p50':>9} {'TTFT p95':>9} {'TPS mean':>9} {'TPS p50':>8} {'Total p50':>10}")
    print("-" * 72)
    for s in summaries:
        print(
            f"{s.backend:<12} {s.model_size:<5} {s.n_success:>4} {s.n_errors:>4} "
            f"{s.ttft_p50_ms:>8.0f}ms {s.ttft_p95_ms:>8.0f}ms "
            f"{s.tps_mean:>8.1f}  {s.tps_p50:>7.1f} "
            f"{s.total_p50_ms:>9.0f}ms"
        )
    print("=" * 72)


async def _ensure_loaded(base_url: str, model_id: str) -> None:
    """Load model via oCabra if not already loaded."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url}/ocabra/models") as resp:
            models = await resp.json()
        state = next((m for m in models if m["model_id"] == model_id), None)
        if state is None:
            print(f"  Model {model_id!r} not registered — skipping.")
            return
        if state["status"] == "loaded":
            print(f"  {model_id} already loaded.")
            return
        print(f"  Loading {model_id} ...")
        enc_id = model_id.replace("/", "%2F")
        async with session.post(f"{base_url}/ocabra/models/{enc_id}/load") as resp:
            result = await resp.json()
            print(f"  → status: {result.get('status', result.get('detail', '?'))}")


async def main(model_sizes: list[str], backends: list[str], output: str | None) -> None:
    all_results: list[RequestResult] = []
    all_summaries: list[BenchmarkSummary] = []

    for size in model_sizes:
        cfg = MODEL_CONFIG[size]
        print(f"\n{'─'*60}")
        print(f"  Benchmark — Qwen3-{size.upper()} | {len(PROMPTS)} prompts | max_tokens={MAX_TOKENS}")
        print(f"{'─'*60}")

        for backend in backends:
            if backend != "ollama":
                model_id = cfg[backend]
                print(f"\n[{backend}] Ensuring {model_id} is loaded...")
                await _ensure_loaded(cfg["ocabra_url"], model_id)

            print(f"\n[{backend}] Running {len(PROMPTS)} prompts...")
            results = await run_backend(backend, size, cfg)
            all_results.extend(results)
            summary = summarize(results)
            all_summaries.append(summary)

        print_summary([s for s in all_summaries if s.model_size == size])

    print("\n\n=== OVERALL SUMMARY ===")
    print_summary(all_summaries)

    if output:
        data = {
            "results": [
                {
                    "backend": r.backend, "model_size": r.model_size,
                    "prompt_idx": r.prompt_idx, "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms, "output_tokens": r.output_tokens,
                    "tokens_per_sec": r.tokens_per_sec, "error": r.error,
                }
                for r in all_results
            ],
            "summaries": [
                {k: getattr(s, k) for k in BenchmarkSummary.__dataclass_fields__}
                for s in all_summaries
            ],
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ollama vs vLLM vs TRT-LLM")
    parser.add_argument("--model", dest="models", action="append", choices=["8b", "32b"], default=None)
    parser.add_argument("--backend", dest="backends", action="append", choices=["vllm", "trtllm", "ollama"], default=None)
    parser.add_argument("--output", default=None, help="Save JSON results to file")
    args = parser.parse_args()

    models = args.models or ["8b", "32b"]
    backends = args.backends or ["vllm", "trtllm", "ollama"]

    asyncio.run(main(models, backends, args.output))
