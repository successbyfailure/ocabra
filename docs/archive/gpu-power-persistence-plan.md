# Plan: persistir power caps de GPU

**Fecha:** 2026-06-05
**Estado:** Fase 1 entregada y verificada en hardware
**Objetivo:** Que la configuración de límite de potencia (TDP) y modo de
persistencia de las GPUs sobreviva a reinicios del contenedor `api`, del
host y del driver NVIDIA.

**Contexto:** El `PATCH /ocabra/gpus/{i}/power` ya existía y aplicaba el
cap en caliente vía `hw-monitor`, pero la decisión no se almacenaba en
ningún sitio. Cada reinicio devolvía las GPUs a `default_w` (170 W para
la RTX 3060, 370 W para la RTX 3090 en este host), exponiendo a la
máquina a picos de potencia que el usuario había evitado a propósito.

---

## Fase 1 — Persistencia + reapply automático · ✅ entregada

### F1-1: Tabla `gpu_power_settings` · ✅

Migración `0020_gpu_power_settings.py`:

| Columna | Tipo | Notas |
|--|--|--|
| `gpu_uuid` | varchar(80) PK | NVML UUID, estable entre reboots / hardware reorder |
| `power_limit_w` | int nullable | watios; `NULL` = sin cap configurado |
| `persistence_mode` | bool nullable | NVML persistence mode |
| `last_known_index` | int nullable | solo debug/display |
| `last_known_name` | varchar(128) nullable | solo debug/display |
| `updated_at` | timestamp | server_default `now()` |

**Decisión clave**: la clave es el UUID, no el índice. Si añades / quitas
una GPU del slot PCIe, el índice cambia pero el UUID no — el cap se
sigue aplicando a la tarjeta correcta.

### F1-2: Módulo `ocabra/db/gpu_power_settings.py` · ✅

Helpers:

- `read_gpu_uuid(index)` / `read_gpu_name(index)` — wrappers NVML
  tolerantes a errores.
- `upsert_setting(session, gpu_uuid=..., power_limit_w=..., ...)` —
  usa `INSERT ... ON CONFLICT DO UPDATE` de PostgreSQL. Los kwargs
  `None` significan "no toques esta columna" para evitar pisar valores
  no relacionados.
- `reapply_persisted_settings(session_factory, *, wait_for_hw_monitor=True)` —
  función idempotente que (a) inicializa NVML, (b) construye el mapa
  `uuid → índice actual`, (c) espera el heartbeat de hw-monitor hasta
  30 s, (d) por cada fila publica `gpu:set_power_limit` (canal Redis
  pubsub donde hw-monitor escucha) y opcionalmente fija el persistence
  mode vía NVML directa. Devuelve `{"applied", "skipped_no_match", "errors"}`.

### F1-3: Endpoint `PATCH /ocabra/gpus/{index}/power` · ✅

Tras aplicar el cap en caliente, hace upsert en la tabla. Casos:

- `power_limit_w` ausente del body → solo persistence_mode se actualiza.
- `power_limit_w = 0` (sentinel reset-to-default): si tampoco hay
  persistence_mode, se borra la fila entera. Si lo hay, se guarda
  `power_limit_w = NULL` para que el próximo arranque no reaplique nada
  pero conserve la preferencia de persistence.
- Caso normal: upsert con el valor.

`GET /ocabra/gpus/{index}/power-limits` añade `saved_w` y
`saved_persistence_mode` para que el UI pinte el badge.

### F1-4: Hook en `lifespan` · ✅

En `main.py`, tras `gpu_manager_ready`, se lanza `_reapply_gpu_power_caps`
como tarea de background:

```python
gpu_power_reapply_task = asyncio.create_task(
    _reapply_gpu_power_caps(),
    name="gpu-power-reapply",
)
```

No bloquea el resto del startup (la tarea espera el heartbeat de
hw-monitor por su cuenta). En el log queda:

```
gpu_power_setting_reapplied gpu_index=0 gpu_uuid=GPU-cc2065... power_limit_w=140
gpu_power_setting_reapplied gpu_index=1 gpu_uuid=GPU-73e2f3... power_limit_w=280
gpu_power_reapply_summary  applied=2 errors=0 skipped_no_match=0
```

### F1-5: UI — badge "Guardado" · ✅

`GPUSettings.tsx` muestra un pill verde **"Guardado"** cuando el valor
actual coincide con `saved_w`, y un pill ámbar **"Guardado: 280W"**
cuando hay un valor guardado distinto del actual (por ejemplo, alguien
movió el slider sin pulsar "Aplicar"). El texto de ayuda menciona
explícitamente la persistencia automática.

### F1-6: Tests · ✅

`tests/test_gpu_power_settings.py` (5 tests, sin DB ni NVML reales):

- `test_reapply_skips_when_no_pynvml` — entorno sin NVML.
- `test_reapply_no_rows_returns_zero` — tabla vacía.
- `test_reapply_publishes_for_matching_uuid` — flujo feliz: lee UUID
  vivo, publica al canal `gpu:set_power_limit` con el índice correcto,
  fija persistence_mode vía NVML.
- `test_reapply_counts_unknown_uuid_as_skipped` — UUID guardado que ya
  no existe en el hardware actual.
- `test_reapply_waits_for_hw_monitor` — sin heartbeat, marca error y no
  publica.

### F1-7: Verificación en hardware · ✅

Inserto manualmente dos filas (140 W para la RTX 3060, 280 W para la
RTX 3090). Reinicio `docker compose restart api`. En el log:

```
gpu_power_reapply_summary applied=2 errors=0 skipped_no_match=0
```

En hw-monitor:

```
power_limit_set gpu=0 limit_w=140.0
power_limit_set gpu=1 limit_w=280.0
```

`nvmlDeviceGetPowerManagementLimit` confirma:

```
GPU 0: limit=140W
GPU 1: limit=280W
```

Luego limpio las filas de prueba y reaplico los defaults vía Redis para
no dejar las GPUs limitadas (170 W / 370 W).

---

## Avances

- Migración + tabla en BD, indexada por UUID estable.
- `PATCH` persiste automáticamente; `GET` expone el valor guardado.
- Reapply en arranque, no bloqueante, robusto a NVML/hw-monitor caídos.
- Badge visual claro en el UI; el usuario sabe a simple vista si el cap
  está "pinned" o solo en caliente.
- Verificación end-to-end con hardware real.

---

## Deudas / cuestiones pendientes

### D-1: Modo de persistencia no funciona sin privilegios extra en `api`

`nvmlDeviceSetPersistenceMode` se llama directa desde el contenedor
`api`, que no es `privileged`. En la mayoría de hosts NVIDIA esto falla
con "Insufficient Permissions" y se loguea como `gpu_power_reapply_persistence_mode_failed`.
El cap de potencia sigue funcionando porque va por hw-monitor.

**Sugerencia:** mover persistence-mode también a Redis con un nuevo
canal (`gpu:set_persistence_mode`) que hw-monitor consuma.

### D-2: No hay validación de rango contra `min_w` / `max_w` al persistir

Si el usuario hace `POST` con `power_limit_w=10000` y NVML lo rechaza,
fallamos antes de la persistencia (correcto). Pero si NVML lo acepta y
luego la GPU se cambia por una con `max_w` menor, el reapply intentará
fijar un valor fuera de rango y hw-monitor solo loguea el error.

**Sugerencia:** en el reapply, clamp al `[min_w, max_w]` actual antes
de publicar.

### D-3: La fila no se borra automáticamente si una GPU desaparece

Si extraes una tarjeta, su fila se queda huérfana en la BD para
siempre. El reapply la cuenta como `skipped_no_match` pero no la limpia.

**Sugerencia:** opción A — purgar filas con `updated_at` > N meses
sin match. Opción B — endpoint admin `DELETE /ocabra/gpus/saved/{uuid}`.

### D-4: No hay endpoint para listar los saved settings

Para auditar qué hay guardado sin entrar en la BD, falta un
`GET /ocabra/gpus/saved`. El UI actualmente solo muestra el saved del
GPU activo via `/power-limits`.

### D-5: El `reapply` no se reintenta si hw-monitor llega tarde

Esperamos hasta 30 s. Si hw-monitor tarda más (caso raro pero posible
tras un host reboot), la primera tanda queda sin aplicar. La fila
sigue en BD y se aplicará al siguiente reinicio, pero hasta entonces
las GPUs van a `default_w`.

**Sugerencia:** loop de reintento con backoff hasta ~10 min.

---

## Referencias

- Commit: `bca9310` — *feat(gpu): persist power caps + persistence mode
  across restarts*
- Canal Redis: `gpu:set_power_limit` (consumido por `hw-monitor`)
- NVML docs:
  https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
