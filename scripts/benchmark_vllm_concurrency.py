#!/usr/bin/env python3
"""Benchmark a local vLLM checkpoint through its OpenAI-compatible server.

This script is intended to run inside the oCabra API container, where the
vLLM runtime and GPU devices are available. It starts an isolated worker,
warms it up, measures concurrent batches, and always unloads the worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx
from ocabra.backends.vllm_backend import VLLMBackend

DEFAULT_PROMPT = (
    "Explain in concise technical terms why continuous batching improves "
    "LLM serving throughput. Give exactly three points."
)
CODE_PROMPT = (
    "Implement a production-quality Python async function that retries an "
    "HTTP request with exponential backoff, jitter, type hints, cancellation "
    "safety, and concise tests. Explain the key decisions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--attention-backend")
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--speculative-method")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--code-prompt", action="store_true")
    parser.add_argument("--concurrency", default="1,4,8,12")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


async def request_batch(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    prompt: str,
    concurrency: int,
    max_tokens: int,
) -> dict[str, float | int]:
    async def request_one(index: int) -> tuple[int, float]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Request {index}: {prompt}",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
        }
        started = time.perf_counter()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        return int(body["usage"]["completion_tokens"]), time.perf_counter() - started

    started = time.perf_counter()
    responses = await asyncio.gather(*(request_one(i) for i in range(concurrency)))
    wall_seconds = time.perf_counter() - started
    output_tokens = sum(tokens for tokens, _ in responses)
    return {
        "concurrency": concurrency,
        "wall_seconds": wall_seconds,
        "output_tokens": output_tokens,
        "aggregate_tokens_per_second": output_tokens / wall_seconds,
        "request_seconds_max": max(duration for _, duration in responses),
    }


async def main() -> None:
    args = parse_args()
    gpu_indices = [int(value) for value in args.gpus.split(",")]
    concurrency = [int(value) for value in args.concurrency.split(",")]
    extra_config: dict[str, Any] = {
        "tensor_parallel_size": len(gpu_indices),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "disable_log_requests": True,
        "enforce_eager": False,
        "language_model_only": args.language_model_only,
        "use_flashinfer_sampler": False,
    }
    if args.pipeline_parallel_size:
        extra_config["pipeline_parallel_size"] = args.pipeline_parallel_size
        extra_config["tensor_parallel_size"] = 1
    if args.tool_call_parser:
        extra_config["tool_call_parser"] = args.tool_call_parser
    if args.kv_cache_dtype:
        extra_config["kv_cache_dtype"] = args.kv_cache_dtype
    if args.attention_backend:
        extra_config["attention_backend"] = args.attention_backend
    if args.speculative_method:
        extra_config["speculative_config"] = {
            "method": args.speculative_method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.cpu_offload_gb:
        extra_config["extra_args"] = ["--cpu-offload-gb", str(args.cpu_offload_gb)]

    backend = VLLMBackend()
    started = time.perf_counter()
    try:
        worker = await backend.load(
            args.model,
            gpu_indices,
            port=args.port,
            extra_config=extra_config,
        )
        load_seconds = time.perf_counter() - started
        url = f"http://127.0.0.1:{worker.port}/v1/chat/completions"
        prompt = CODE_PROMPT if args.code_prompt else DEFAULT_PROMPT
        timeout = httpx.Timeout(600)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await request_batch(
                client,
                url=url,
                model=args.model,
                prompt=prompt,
                concurrency=1,
                max_tokens=16,
            )
            results: dict[str, Any] = {}
            for count in concurrency:
                samples = [
                    await request_batch(
                        client,
                        url=url,
                        model=args.model,
                        prompt=DEFAULT_PROMPT,
                        concurrency=count,
                        max_tokens=64,
                    )
                    for _ in range(args.repetitions)
                ]
                results[str(count)] = {
                    "wall_seconds_median": statistics.median(
                        float(sample["wall_seconds"]) for sample in samples
                    ),
                    "aggregate_tokens_per_second_median": statistics.median(
                        float(sample["aggregate_tokens_per_second"])
                        for sample in samples
                    ),
                    "samples": samples,
                }

            long_samples = [
                await request_batch(
                    client,
                    url=url,
                    model=args.model,
                    prompt=prompt,
                    concurrency=1,
                    max_tokens=256,
                )
                for _ in range(args.repetitions)
            ]
            results["long"] = {
                "wall_seconds_median": statistics.median(
                    float(sample["wall_seconds"]) for sample in long_samples
                ),
                "aggregate_tokens_per_second_median": statistics.median(
                    float(sample["aggregate_tokens_per_second"])
                    for sample in long_samples
                ),
                "samples": long_samples,
            }
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "gpu_indices": gpu_indices,
                        "config": extra_config,
                        "load_seconds": load_seconds,
                        "vram_estimate_mb": worker.vram_used_mb,
                        "results": results,
                    },
                    indent=2,
                )
            )
    finally:
        await backend.unload(args.model)


if __name__ == "__main__":
    asyncio.run(main())
