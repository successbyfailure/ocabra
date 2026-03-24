#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _request(method: str, url: str, body: dict | None = None, timeout: int = 60) -> tuple[int, dict | str | None]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: dict | str = json.loads(raw)
        except Exception:
            payload = raw
        return exc.code, payload


def _find_engine_dir(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if list(child.rglob("*.engine")):
            return child
    return None


def _guess_model_id_from_dir(engine_dir: Path) -> str:
    name = engine_dir.name
    if "--" in name:
        return name.replace("--", "/")
    return name


def _as_model_path(model_id: str) -> str:
    return urllib.parse.quote(model_id, safe="/:._-")


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorRT-LLM smoke test for oCabra")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="oCabra API base URL")
    parser.add_argument("--model-id", default="", help="Canonical model id (e.g. tensorrt_llm/org/model)")
    parser.add_argument("--engine-dir", default="", help="TensorRT engine dir with *.engine files")
    parser.add_argument("--tokenizer-path", default="", help="Optional tokenizer path")
    parser.add_argument("--engine-dir-api", default="", help="Engine dir path as seen by API container")
    parser.add_argument("--tokenizer-path-api", default="", help="Tokenizer path as seen by API container")
    parser.add_argument("--launch-mode", default="module", choices=["binary", "module", "docker"])
    parser.add_argument("--python-bin", default="/usr/bin/python3")
    parser.add_argument("--serve-module", default="tensorrt_llm.commands.serve")
    parser.add_argument("--serve-bin", default="/usr/local/bin/trtllm-serve")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--keep-config", action="store_true", help="Do not delete model config at the end")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    status, payload = _request("GET", f"{base}/health", timeout=20)
    if status != 200:
        print(f"[FAIL] API not healthy: status={status} payload={payload}")
        return 1
    print("[OK] API health")

    model_id = args.model_id.strip()
    engine_dir = Path(args.engine_dir).expanduser() if args.engine_dir else None
    engine_dir_api = args.engine_dir_api.strip()
    tokenizer_path_api = args.tokenizer_path_api.strip()

    if engine_dir is None and not engine_dir_api:
        guesses = [
            Path("/docker/ai-models/ocabra/models/tensorrt_llm"),
            Path("/docker/ai-models/ocabra/models/tensorrt"),
            Path("/data/models/tensorrt_llm"),
        ]
        for guess in guesses:
            found = _find_engine_dir(guess)
            if found is not None:
                engine_dir = found
                print(f"[INFO] Detected engine dir: {engine_dir}")
                break

    if engine_dir is None or not engine_dir.exists():
        if not engine_dir_api:
            print("[FAIL] No TensorRT engine directory found.")
            print("       Provide --engine-dir /path/to/engine or --engine-dir-api /path/in/api.")
            return 2
    else:
        if not list(engine_dir.rglob("*.engine")):
            print(f"[FAIL] Engine dir has no *.engine files: {engine_dir}")
            return 2

    if not model_id:
        if engine_dir is None:
            print("[FAIL] --model-id is required when --engine-dir is not locally available")
            return 2
        guessed = _guess_model_id_from_dir(engine_dir)
        if not guessed.startswith("tensorrt_llm/"):
            guessed = f"tensorrt_llm/{guessed}"
        model_id = guessed
        print(f"[INFO] Using model_id={model_id}")

    extra_config: dict[str, object] = {
        "engine_dir": (engine_dir_api or str(engine_dir)),
        "launch_mode": args.launch_mode,
        "python_bin": args.python_bin,
        "serve_module": args.serve_module,
        "serve_bin": args.serve_bin,
        "max_batch_size": args.max_batch_size,
        "context_length": args.context_length,
    }
    tokenizer_path_value = tokenizer_path_api or args.tokenizer_path.strip()
    if tokenizer_path_value:
        extra_config["tokenizer_path"] = tokenizer_path_value

    register_body = {
        "model_id": model_id,
        "backend_type": "tensorrt_llm",
        "display_name": model_id.split("/", 1)[-1],
        "load_policy": "on_demand",
        "auto_reload": False,
        "extra_config": extra_config,
    }

    status, payload = _request("POST", f"{base}/ocabra/models", register_body, timeout=60)
    if status not in (200, 201, 400):
        print(f"[FAIL] Register failed: status={status} payload={payload}")
        return 3
    if status in (200, 201):
        print("[OK] Model registered")
    else:
        print(f"[INFO] Model already existed (status={status})")

    model_path = _as_model_path(model_id)

    status, payload = _request("POST", f"{base}/ocabra/models/{model_path}/load", timeout=600)
    if status != 200:
        print(f"[FAIL] Load failed: status={status} payload={payload}")
        return 4
    print("[OK] Model loaded")

    infer_body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    status, payload = _request("POST", f"{base}/v1/chat/completions", infer_body, timeout=300)
    if status != 200:
        print(f"[FAIL] Inference failed: status={status} payload={payload}")
        return 5
    print("[OK] Inference response received")

    status, payload = _request("POST", f"{base}/ocabra/models/{model_path}/unload", timeout=120)
    if status != 200:
        print(f"[FAIL] Unload failed: status={status} payload={payload}")
        return 6
    print("[OK] Model unloaded")

    if not args.keep_config:
        status, payload = _request("DELETE", f"{base}/ocabra/models/{model_path}", timeout=120)
        if status != 200:
            print(f"[WARN] Delete config failed: status={status} payload={payload}")
        else:
            print("[OK] Model config removed")

    print("[PASS] TensorRT-LLM smoke test completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
