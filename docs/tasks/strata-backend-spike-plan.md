# Plan de evaluación e integración de Strata

**Estado:** spike preparado; pendiente ejecutar la descarga y benchmark en hardware libre
**Backend propuesto:** `strata` (experimental y opt-in)
**Upstream fijado para el spike:** Strata `v0.2.13`, Terminal/Control `0.1.59.1`
**Hardware objetivo:** RTX 3060 12 GB (GPU 0) + RTX 3090 24 GB (GPU 1), 94 GiB RAM

## Decisión que debe resolver el spike

Strata no compite como motor general con vLLM, llama.cpp o SGLang. Su valor es
servir unas pocas familias MoE grandes repartiendo el trabajo entre GPU, RAM,
CPU y SSD. Solo se integrará si aporta una ventaja material y repetible en este
host frente al backend llama.cpp ya operativo.

La primera prueba usa **Qwen3.6-35B-A3B UD-Q4_K_XL** porque necesita unos 23 GB
de modelo, cabe con margen en disco y RAM y permite aislar el valor del motor sin
la descarga de 66–76 GB de Qwen3.8. DeepSeek V4.1 queda fuera: requiere al menos
239 GB de descarga y el almacén de modelos dispone actualmente de unos 147 GB.

## Alcance del spike preparado

El harness está en `benchmark/strata_spike.py` y separa explícitamente:

1. `preflight`: hardware, RAM, disco, puerto, versión y procesos GPU; no cambia estado.
2. `plan`: ejecuta `strata launch --plan`; no descarga ni arranca el motor.
3. `run --execute`: descarga lo que falte, arranca, mide, archiva y detiene Strata.

La instalación reproducible del Terminal está en
`benchmark/install_strata_spike.sh`. Descarga el instalador de la release
fijada y verifica el SHA-256 antes de ejecutarlo. Todo queda aislado en
`/docker/ai-models/ocabra/models/huggingface/strata-spike`; no modifica el PATH
global ni instala paquetes del sistema. Se usa esta subcarpeta porque el dataset
ZFS `/docker/ai-models` tiene espacio suficiente y concede escritura ahi; la
raiz `/docker/ai-models/ocabra` no concede escritura al usuario operativo.

El runner prueba dos perfiles con iguales modelo, contexto, visión y
concurrencia:

| Perfil | Roles | Objetivo |
|---|---|---|
| `3090` | GPU 1 main, GPU 0 off | Baseline de una sola tarjeta |
| `split` | GPU 1 main, GPU 0 helper | Medir el valor real de la caché/capas adicionales |

Se fuerza contexto 32K, visión desactivada, un slot y layer split normal. Así se
evita mezclar el resultado con el coste del encoder visual, batching o el split
engine experimental.

## Procedimiento de prueba

### 1. Instalar el Terminal fijado

```bash
chmod +x benchmark/install_strata_spike.sh benchmark/strata_spike.py
benchmark/install_strata_spike.sh
```

El instalador solo ocupa aproximadamente 100 MB. El motor y el modelo no se
descargan en este paso.

En este host el paso ya esta ejecutado y deja `strata 0.2.13.1` (Control
`0.1.59.1`) instalado en el prefijo aislado.

### 2. Preflight y plan sin cambios

```bash
python3 benchmark/strata_spike.py preflight
python3 benchmark/strata_spike.py plan --profile 3090
python3 benchmark/strata_spike.py plan --profile split
```

Revisar especialmente el modelo, descarga, rutas y selección de GPU que devuelve
el plan de Strata. No continuar si el plan elige otra familia o requiere usar swap
para mantener el modelo.

Resultado del preflight del 2026-11-09: hardware, driver, RAM, disco y puerto son
compatibles; quedan como bloqueantes tres procesos GPU de oCabra (Whisper y un
llama-server repartido entre ambas tarjetas). Los dos planes upstream validan
Qwen3.6 UD-Q4_K_XL, contexto 32K y los roles pedidos. Ambos dicen que el engine
falta. Tambien dicen `0 GB` y "model already on this computer" aunque el directorio
aislado no contiene ningun run config; se considera una estimacion no fiable del
plan y la ejecucion debe asumir que puede descargar los ~23 GB completos.

### 3. Liberar las GPU

Descargar desde oCabra los modelos residentes y confirmar que no quedan procesos
de cómputo. El runner se niega a ejecutar mientras detecte procesos GPU. La opción
`--allow-busy-gpus` solo existe para una decisión deliberada del operador; no debe
usarse en la comparación de aceptación.

### 4. Ejecutar ambos brazos

```bash
python3 benchmark/strata_spike.py run --profile 3090 --execute
python3 benchmark/strata_spike.py run --profile split --execute
```

Cada ejecución conserva `run.json` bajo
`benchmark/results/strata-spike/<fecha>_<perfil>/` con:

- revisión y estado dirty de oCabra;
- versión de Strata y hardware;
- roles y comando efectivos, sin API key;
- tiempo de carga total;
- `/health` y `/v1/status`;
- VRAM por GPU cargado y después del stop;
- TTFT, latencia, tokens/s y muestra de respuesta para dos prompts;
- cualquier error de parada o proceso residual.

## Criterios de aceptación

El spike pasa solo si se cumplen todos:

1. Carga, dos generaciones streaming, una tool call forzada y descarga completan
   sin intervención.
2. Tras el stop no quedan procesos Strata ni VRAM materialmente retenida.
3. No hay crecimiento sostenido de swap durante las generaciones.
4. El texto y tool calling básico son utilizables; no basta con velocidad.
5. Frente a llama.cpp, Strata aporta al menos una de estas ventajas:
   - 20 % más throughput de salida con TTFT no peor en más de 25 %;
   - contexto/capacidad que llama.cpp no puede servir estable;
   - 25 % menos RAM+VRAM para rendimiento equivalente.
6. El perfil split solo se recomendará si mejora al menos 10 % sobre la 3090
   sola o permite una configuración de calidad/contexto que no cabe en ella.

Una regresión pequeña puede aceptarse únicamente si habilita Qwen3.8 o DeepSeek
con calidad útil que ningún backend actual puede ofrecer en el host.

## Integración posterior al spike

Si pasa, el cambio de producto se divide en unidades revisables:

### A. Contrato y artefacto

- Añadir `strata` a los backend types y a `docs/CONTRACTS.md`.
- Añadir artefacto `strata_config`, cuyo `path` apunta al `strata-*.json`
  generado por el preparador oficial.
- Mantener separado preparar/descargar de cargar el worker.

### B. `StrataBackend`

- Implementar `BackendInterface` en `backend/ocabra/backends/strata_backend.py`.
- oCabra lanza `serve/server.py` en localhost y posee el grupo de procesos.
- API key interna aleatoria; nunca exponer directamente el puerto del worker.
- Readiness mediante `/health` con `loaded=true`.
- Forward de `/v1/chat/completions`, incluido SSE.
- Capabilities iniciales: chat, tools, reasoning, streaming y visión opcional.
- No anunciar completions legacy, embeddings, pooling, rerank, audio ni imagen.
- Desactivar idle unload/autoload de Strata: ModelManager es la única autoridad.
- Terminar y reaprovechar procesos huérfanos igual que los demás backends nativos.

### C. Descubrimiento y configuración

- Descubrir run configs válidos, no cualquier GGUF compatible por nombre.
- Exponer contexto, roles GPU, slots y visión dentro de `extra_config.strata`.
- Estimar VRAM como reserva operativa y publicar por separado la RAM esperada;
  `vram_used_mb` no representa por sí solo el coste de Strata.
- No automatizar `curl | sh`: la instalación modular debe fijar release, digest
  y artefactos verificables, idealmente vía OCI cuando esa fase esté terminada.

### D. Validación de producto

- Tests unitarios de comandos, health, forwarding, SSE y cleanup cancelado.
- E2E load/chat/unload en 3090 y split.
- Benchmark contra `llamacpp_mks` con el mismo modelo, contexto y prompts.
- Prueba de tools, visión opcional, timeout, cancelación y presión de VRAM.
- Añadir changelog y subir versión solo cuando el backend sea visible en API/UI.

## Límites deliberados

- El spike no registra todavía `strata` en `main.py` ni altera ModelManager.
- No descarga un modelo sin `run --execute` explícito.
- No modifica `docker-compose.yml`.
- No prueba DeepSeek por falta de espacio.
- No evalúa batching hasta validar primero estabilidad y rendimiento single-slot.

Estos límites permiten decidir con datos antes de ampliar contratos compartidos
o introducir otro gestor de procesos en producción.
