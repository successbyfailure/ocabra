# Plan: Nemotron-3-Nano en vLLM para asistente-agente concurrente (3090)

Fecha: 2026-07-26

## Objetivo
Servir **Nemotron-3-Nano-30B-A3B-AWQ** por vLLM en la RTX 3090 como modelo
asistente-agente que atiende **el máximo de peticiones concurrentes con 64K de
contexto por petición**, aprovechando el KV barato de su arquitectura híbrida
Mamba. Limpiar los modelos cuantizados que no vamos a usar.

## Contexto (verificado esta sesión)
- vLLM 0.26.0 + torch 2.11.0 + transformers 5.14.1; el entorno 0.19.1 se
  conserva como rollback.
- Nemotron AWQ carga nativo (compressed-tensors → Marlin MoE), KV pool ~153K tok,
  2.84× concurrencia a 256K → a **64K la concurrencia sube a ~11×**.
- gemma-4-12B se descartó durante la evaluación original de vLLM 0.19.1
  (`gemma4_unified`, crash Marlin). Sigue disponible en Ollama; requiere una
  revalidación separada antes de asumir soporte en 0.26.0.
- Qwen3-Coder-30B-A3B-AWQ: transformer estándar, KV pesado (~59K a fp16). No es
  el elegido para concurrencia.

## Pasos
1. ✅ **Reconfigurar `nemotron-nano-vllm`** en oCabra:
   - `max_model_len = 65536` (64K por petición → más concurrencia).
   - `gpu_memory_utilization = 0.92`.
   - Reasoning parser de vLLM si está soportado (para que el razonamiento vaya a
     `reasoning_content` y no ensucie `content`).
2. ✅ **Verificar** carga en la 3090 y concurrencia:
   - Proceso efectivo: `--max-model-len 65536 --max-num-seqs 12`.
   - Carga validada en GPU 1 con 20.418 MiB estimados por oCabra.
   - `/v1/chat/completions` separa `reasoning` de `content`.
   - Ráfaga real: 12/12 peticiones concurrentes cortas HTTP 200 en 1,86 s.
   - `max_num_seqs=12` limita las secuencias activas, pero no implica
     `12 × 64K`: con el presupuesto KV observado caben aproximadamente dos
     peticiones que ocupen el contexto completo de 64K; las demás esperan en cola.
3. ✅ **Limpiar modelos no usados** (~liberar disco en `/data/hf_cache/hub` y la
   caché efímera del contenedor):
   - `cyankiwi/gemma-4-12B-it-AWQ-INT4`
   - `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
   - `Vishva007/gemma-4-E4B-it-W4A16-AutoRound-GPTQ`
   - `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`
   - `google/gemma-4-12B-it-qat-w4a16-ct`
   - `stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ` (en `/root/.cache`)
   - **CONSERVAR**: `stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ` (el que usamos).

## Notas
- La 3090 con vLLM no convive con gemma4:26b (Ollama) ni con otro vLLM a la vez;
  Nemotron es `on_demand` y desaloja lo que haya al cargarse.
- El timeout global de inactividad es 123 s. Para evitar cargas en frío, cambiar
  Nemotron a política `warm` o ampliar su timeout; `warm` conserva unos 22,5 GiB
  de VRAM mientras el worker siga cargado.
- gemma-4 en vLLM queda pendiente de una futura actualización de vLLM con soporte
  nativo `Gemma4Unified`.

## Benchmark

Comparativa reproducible de carga, velocidad individual y lotes concurrentes:
[`docs/benchmarks/nemotron_vllm_vs_ollama.md`](../benchmarks/nemotron_vllm_vs_ollama.md).

Comparativa antes/después de vLLM 0.26.0, con una y dos GPU:
[`docs/benchmarks/nemotron_vllm_upgrade_026.md`](../benchmarks/nemotron_vllm_upgrade_026.md).
