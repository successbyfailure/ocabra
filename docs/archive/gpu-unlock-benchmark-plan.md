# Plan de prueba — GPU unlock (Zagitta) + patch P2P

**Estado: ejecutado 2026-10-24.** Resultados completos, scripts reutilizables
y el estado actual del sistema viven en `/home/makespace/gpu-bench/README.md`
(fuera de este repo, a propósito — es un proyecto de benchmarking de host,
no código de oCabra). Este fichero queda como referencia histórica del plan
original.

Resumen: Zagitta instalado y activo (mejora GEMM sintético, ~0% en LLM real
cuantizado). Patch P2P bloqueado (infraestructura lista — ReBAR, IOMMU,
ACS — pero el driver no reporta P2P disponible ni tras 3 intentos; probable
causa en firmware GSP cerrado). P2P en pausa hasta nueva versión/documentación.

## Hardware real (confirmado 2026-10-24)

```
GPU0  NVIDIA GeForce RTX 3060  12GB  bus 00000000:01:00.0
GPU1  NVIDIA GeForce RTX 3090  24GB  bus 00000000:21:00.0
```

`nvidia-smi topo -m`: conexión `NODE` entre ambas — cada una cuelga de un
puente PCIe distinto dentro del mismo nodo NUMA, no comparten switch PCIe.
No hay NVLink.

Repasado `vllm_backend.py` / `runtime_preset_capacity.py`: ningún preset de
vLLM reparte un modelo entre 3060+3090 vía TP (son heterogéneas, vLLM
espera GPUs equivalentes). **Corrección**: eso no significa que no haya
multi-GPU real hoy — `gemma4:26b` (PIN de producción) y `RavenX 30B` SÍ
corren con `gpu_indices=[0,1]` vía llama.cpp/Ollama, con
`tensor_split=evenly` repartiendo capas proporcional a VRAM entre ambas
tarjetas ([[project_ollama_kv_context]], [[project_scheduler_eviction]]
en memoria). Ese es el patrón que le importa a la Parte B, no el TP de
vLLM.

## Parte A — Zagitta `gpu-unlock`

Repo: https://github.com/Zagitta/gpu-unlock
Desbloquea el ratio de emisión de instrucciones FP16/BF16 FMA→FP32 en
GA102/103/104/106 (incluye nuestra 3090 y 3060) escribiendo los speed
selectors antes de que el driver nvidia haga bind. Según el repo, no
mejora decode (memory-bound) pero sí prefill, diffusion y FlashAttention.

### A.1 Baseline (antes de instalar nada)

1. Fijar relojes para quitar varianza de boost:
   `sudo nvidia-smi -i <idx> -lgc <min>,<max>` (usar el rango de boost
   estable de cada tarjeta) y `-pl` al power limit actual.
2. Dejar calentar la GPU ~2 min con carga sintética antes de medir (el
   throttling térmico falsea la primera muestra).
3. Microbenchmark GEMM puro (igual metodología que el propio repo, para
   poder comparar con sus cifras reportadas):
   - Script PyTorch con `torch.matmul` en FP16, BF16 y TF32 sobre matrices
     cuadradas grandes (8192³ si cabe en VRAM de la 3060; si no, escalar),
     usando CUDA events para timing, 5 muestras, reportar mediana en TFLOPS.
   - Correr en 3090 y 3060 por separado.
4. Benchmark de aplicación real con vLLM, separando los dos regímenes que
   el repo distingue:
   - **Prefill-heavy**: prompt largo (4k-8k tokens), generación corta (ej.
     32 tokens). Usar `vllm/benchmarks/benchmark_throughput.py` o
     `benchmark_serving.py` del propio vLLM con ese perfil.
   - **Decode-heavy**: prompt corto, generación larga (ej. 512-1024
     tokens). Mismo script, perfil invertido.
   - Métrica: tokens/s de prefill y tokens/s de decode por separado, no
     mezclados — si se promedian se pierde la señal que el repo predice
     (decode no debería moverse, prefill sí).
5. Si usamos el backend de Diffusers en producción, medir también
   it/s o s/step en un pipeline SDXL/similar con steps fijos y mismo seed.
6. Check de correctitud: generar con seed fijo antes/después y diferenciar
   logits o output — el repo reporta salidas bit-idénticas; si no lo son,
   abortar.
7. Registrar potencia y temperatura durante cada corrida
   (`nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv -l 1`)
   para descartar que la mejora venga de un power limit distinto, no del
   unlock.

### A.2 Instalación y medición post-unlock

1. Ventana de mantenimiento (toca initramfs + cmdline del bootloader,
   requiere reboot).
2. `make install` desde el repo, añadir `gpu_unlock.targets=all` al
   cmdline, reboot.
3. Verificar con `journalctl -k -b | grep gpu_unlock` y
   `lspci -nnk -s <bus>` que el unlock corrió en ambas tarjetas.
4. Repetir exactamente los pasos A.1.3–A.1.7 con los mismos scripts,
   mismos seeds, mismo power limit/clocks fijados.
5. Comparar: GEMM TFLOPS, tokens/s prefill, tokens/s decode, it/s
   diffusion, correctitud de salida.

### A.3 Rollback

1. Quitar `gpu_unlock.targets=` del cmdline, reboot.
2. Confirmar que `lspci -nnk` ya no muestra señales del unlock y que la
   carga de modelos funciona con normalidad.
3. Desinstalar el paquete si no se adopta.

## Parte B — Patch P2P (PCIe peer-to-peer)

**Corrección respecto a la primera versión de este plan**: sí hay un
camino de código real que se beneficiaría — `gemma4:26b` (producción,
PIN) y `RavenX 30B` corren hoy repartidos entre 3060+3090 vía llama.cpp
(`split_mode=layer`, `tensor_split=evenly`). En cada forward pass la
activación cruza la frontera entre GPU0/GPU1 una vez. llama.cpp intenta
`cudaDeviceEnablePeerAccess` para ese salto; en GeForce normalmente falla
por la misma restricción artificial que bloquea el patch, y cae a un
staging por RAM del host. El patch convertiría ese salto en una copia
GPU→GPU directa.

Diferencia importante con el caso "TP de vLLM" que asumí originalmente:
aquí el dato que cruza es solo el hidden state de la capa límite (pocos
KB), no un allreduce de NCCL completo — el beneficio esperado es de
**latencia por token** en decode, no de ancho de banda. Esto es
complementario al unlock de Zagitta (que no mejora decode): si el patch
P2P funciona, decode en modelos que spannean ambas GPUs podría mejorar
donde Zagitta no llega.

Aún no tenemos un repo/patch concreto identificado para esto (lo que
describí es el mecanismo general sobre `open-gpu-kernel-modules`, no un
proyecto puntero). Antes de instalar nada en la Parte B hay que decidir
qué implementación usar — pendiente de confirmar un repo.

### B.1 Prerrequisito — ¿siquiera aplica la topología?

1. `nvidia-smi topo -p2p r` (o `-p2p p`) para ver si el driver ya reporta
   P2P como posible/imposible entre bus 01:00.0 y 21:00.0 en el estado
   actual.
2. Revisar ACS (Access Control Services) en la BIOS/kernel — si ACS está
   activo en el puente PCIe, redirige el tráfico P2P por el IOMMU y anula
   la ganancia aunque el patch esté instalado. Comprobar con
   `lspci -vvv` buscando `ACSCtl` en los bridges intermedios.

### B.2 Microbenchmark (el estándar para esto, no hay que inventarlo)

1. `p2pBandwidthLatencyTest` de los CUDA samples de NVIDIA — mide ancho de
   banda y latencia uni/bidireccional con P2P habilitado vs deshabilitado
   explícitamente. Correr antes y después del patch.
2. `nccl-tests` (`all_reduce_perf`, `sendrecv_perf`) entre ambas GPUs para
   ver bus bandwidth real en el patrón de comunicación que usaría vLLM/NCCL
   en TP.

### B.3 Benchmark de aplicación — caso real, no hipotético

Usar `gemma4:26b` (o RavenX 30B) tal cual corre hoy en producción,
spanning 3060+3090:
- Medir **latencia por token en decode** (ms/token, no tokens/s agregado,
  para que se note el coste del salto entre GPUs) con un prompt corto y
  generación larga, antes y después del patch.
- Control sin reinstalar nada: forzar el fallback de host-staging en
  llama.cpp si expone una env var equivalente a `GGML_CUDA_NO_PEER_COPY`
  / similar (verificar en la versión de llama.cpp que empaqueta Ollama;
  si no existe, el control real es simplemente antes/después del patch).
- Repetir con prefill largo para confirmar que ahí el efecto es menor
  (la frontera se cruza menos veces relativo al trabajo total de cómputo).

## Orden recomendado de ejecución

1. Baseline A.1 (no requiere reboot, se puede hacer ya).
2. B.1 (prerrequisito de topología, tampoco requiere reboot) — si P2P ya
   sale "no factible" por topología/ACS, hay que saberlo antes de instalar
   nada.
3. Ventana de mantenimiento para instalar Zagitta gpu-unlock (Parte A.2) —
   beneficio confirmado en prefill/diffusion con el hardware que ya hay.
4. Identificar y aplicar el patch P2P, medir con B.2 (micro) y B.3
   (`gemma4:26b` real) — ya no es "para cuando cambie el hardware", es
   sobre la carga de producción actual. Pendiente: confirmar qué
   repo/implementación de P2P usar.
