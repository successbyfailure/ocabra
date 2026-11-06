# Plan: modernización de la federación

**Fecha:** 2026-11-06
**Estado:** 🔵 pendiente de abordar — documentado, sin trabajo iniciado
**Plan original:** [`federation-plan.md`](federation-plan.md) (2026-04-12)

---

## Contexto

La federación se diseñó en abril de 2026. Desde entonces oCabra ha incorporado
piezas que el routing federado **no conoce**: presets de runtime y paralelismo
efectivo, control de admisión por slots, presupuesto de KV, y la escalera de
contextos por preset.

Criterio del usuario (2026-11-06):

> *"Cuando diseñamos la federación no teníamos paralelismo ni muchas mejoras de
> ahora, probablemente hay que actualizar ese código. Puede que en un futuro
> cercano metamos en la federación un equipo de memoria unificada y ahí
> tendremos nuevos casos de uso y ver qué estrategias usamos para gestionar las
> peticiones."*

Las **estrategias** de gestión de peticiones se deciden más adelante y con el
hardware nuevo delante. Este documento solo deja el terreno mapeado.

### Por qué se ejecuta en remoto (no es una carencia)

Decisión deliberada del usuario: para muchos casos **sale más rápido tirar del
peer que desalojar un modelo caliente** de las tarjetas locales para atender esa
petición. El coste real que se evita no es el throughput del modelo remoto, es la
evicción. **No proponer migrar el reranker a local.**

---

## Ya resuelto — no volver sobre esto

| Versión | Commit | Qué |
|---|---|---|
| v0.31.1 | `3d39ade` | La admisión exceptúa del escalado por slots a los endpoints de encoder, decidiendo por `request_kind` en vez de por `ModelState.capabilities` (que solo se rellena tras una carga **local**, así que un modelo remote-only quedaba limitado a 4 concurrentes). |
| v0.31.2 | `e1c93ed` | Las peticiones destinadas a un peer no heredan el cap local por slots (`remote_bound`). Cierra el mismo problema para modelos **generativos** alojados solo en remoto — el caso de uso directo de una máquina de memoria unificada. |

Detalle en las memorias `project_federated_encoder_inflight_cap` y
`project_federation_routing_stale`.

---

## Hueco 1 — el "sesgo local" no es un sesgo, es preferencia absoluta

`backend/ocabra/core/federation.py:511` (`select_target`). El docstring dice
*"local gets a −5 bias (prefer local to avoid latency)"*, pero el cálculo es:

```python
peer_score  = active_requests * 10 + gpu_util   # nunca < 0
local_score = -5.0
if local_score <= peer_score:
    return "local"
```

`-5 <= peer_score` es cierto **siempre**. Es decir: cuando el modelo está
cargado en local, local gana incondicionalmente. El scoring de peers solo llega
a decidir algo entre varios peers (`find_best_peer:488`) o cuando el modelo no
está cargado aquí.

Hay que decidir si eso es lo que se quiere (y entonces simplificar el código y
el docstring, que hoy engañan) o si el sesgo debe ser ponderado de verdad.

## Hueco 2 — no existe el concepto de coste de evicción

La estrategia descrita arriba —preferir el peer antes que desalojar un modelo
caliente— **no está implementada en ningún sitio**. Hoy funciona *por
construcción*: el reranker simplemente nunca se carga en local, así que
`resolve_federated` cae en su paso 2 (`local_available=False`) y va al peer. El
día que ese modelo se cargue aquí por cualquier motivo, el router se lo queda en
local sin preguntarse qué tendría que desalojar.

Para expresar la estrategia de verdad hace falta que la decisión compare *coste
total* (incluida la evicción y el cold start que provoca) en vez de *carga
instantánea*.

## Hueco 3 — el heartbeat no publica capacidad real

`HeartbeatModel.parallelism` **existe en el esquema pero llega siempre `null`**:
se lee de `extra_config._runtime.parallelism`
(`backend/ocabra/api/internal/federation.py:119`), o sea del sistema de presets
de runtime, que no rellenan los encoders ni vLLM. Misma raíz que los dos bugs ya
arreglados.

El score (`active_requests * 10 + gpu_util`) además ignora:

- el paralelismo efectivo del worker remoto,
- la VRAM libre del peer,
- la profundidad de cola / capacidad de admisión del peer,
- y `active_requests` es **del peer entero**, no por modelo.

Y es una foto del último heartbeat (`federation_heartbeat_interval`,
`backend/ocabra/config.py:756`), con la staleness que eso implica.

## Hueco 4 — supuestos que rompe un equipo de memoria unificada

Para Apple Silicon / Strix Halo / DGX Spark y similares:

- `HeartbeatGpu` exige `index` + `total_vram_mb` / `free_vram_mb` **por
  tarjeta**. Allí hay un pool compartido con CPU y sistema: "VRAM libre" no es
  una partición fija y el techo útil es configurable, no físico.
- `gpu_utilization_avg_pct` vale menos como proxy de carga: el cuello es ancho
  de banda de memoria, no ocupación del núcleo.
- Lo más relevante: la ventaja de esa máquina es que **cabe un modelo grande sin
  desalojar nada** — justo el criterio que se quiere maximizar. El scoring actual
  no tiene forma de expresar *"este peer sostiene este modelo sin evicción"*.

## Hueco 5 — endpoints sin federación

| Endpoint | Federación | Nota |
|---|---|---|
| `/v1/chat/completions` | ✅ | Incluye SSE vía `proxy_stream`, con fallback a local en error de red o rechazo del peer; **sin** fallback en streaming, a propósito |
| `/v1/completions` | ✅ | Streaming incluido |
| `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags` | ✅ | |
| `/v1/audio/transcriptions`, `/v1/audio/speech` | ✅ | |
| `/v1/images/generations`, `/v1/images/edits` | ✅ | |
| `/v1/embeddings` | ✅ | |
| `/v1/pooling`, `/v1/score`, `/v1/rerank`, `/v1/classify` | ✅ | |
| **`/v1/batches`** | ❌ | Cero ganchos (`api/openai/batches.py`). Un batch sobre un modelo remote-only probablemente falle al intentar cargarlo en local — **sin verificar**. Hueco real: Batches llegó después del plan de federación |
| **`/v1/files`** | ❌ | Cero ganchos. Relevante solo en la medida en que Batches lo necesite |
| **`/v1/realtime`** | ❌ | **No es un olvido**: el plan original lo dejó explícitamente fuera del v1 — *"Realtime API WebSocket federado (requiere relay de WebSocket, complejidad alta)"*. Es la pieza con más peso si se quiere voz contra el equipo nuevo |

---

## Decisiones pendientes del usuario

1. ¿El sesgo local debe ser absoluto (y se simplifica el código) o ponderado?
2. ¿Qué publica el heartbeat como "capacidad"? Paralelismo efectivo y huecos
   libres parecen el mínimo, pero para memoria unificada hace falta algo que no
   sea VRAM libre por tarjeta.
3. ¿Entra el relay de WebSocket para realtime federado, o la voz se queda
   local-only?
4. ¿Batches federado merece la pena, o se asume que los batches son locales?

## Cómo verificar cuando se aborde

No hay modelo remote-only **no-encoder** en la federación hoy (mapache solo
expone `vllm/BAAI/bge-reranker-v2-m3` y `ollama/qwen3-embedding:0.6b`, ambos
encoders), así que la vía `remote_bound` ya arreglada **no es observable
end-to-end en producción** y está cubierta solo por tests. El primer modelo
generativo remoto que se añada es la oportunidad de validarla de verdad.

Lecciones de método del ciclo anterior, por si se repite el tipo de medición:

- Medir **en caliente**: la primera tanda tras un cold start de vLLM daba 25× por
  la captura de CUDA graphs y llevó a una conclusión falsa.
- **Comprobar códigos de estado** en pruebas de concurrencia: unos 429
  instantáneos se colaron como "finalizaciones en 0.02s".
- No usar `/ocabra/models` como prueba de lo que hay en memoria:
  `_resolve_capabilities_payload` (`api/internal/models.py:1734`) recalcula las
  capabilities al vuelo y enmascara justo este tipo de bug.
