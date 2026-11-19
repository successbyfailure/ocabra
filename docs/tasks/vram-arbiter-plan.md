# Árbitro de VRAM (REF-03, primera versión) — cierra lo que quedaba de BUG-23

Código: `backend/ocabra/core/vram_arbiter.py`, uso en `core/model_manager.py`.
Tests: `backend/tests/core/test_vram_arbiter.py`.

## Problema

Desde v0.113.0 una carga toma la puerta global (`_backend_load_gate`) desde la
admisión hasta el arranque del backend, y los bucles de fondo ceden si está
ocupada. Seguían fuera de toda coordinación:

- `unload()` manual (y cualquier unload fuera de una carga): cambiaba la VRAM
  mientras una carga medía las tarjetas para admitirse o esperaba a que se
  liberase lo que acababa de expulsar (`_wait_for_vram_released` se da por
  satisfecho con la memoria que libere *cualquiera*).
- `ensure_vram_free` (servicios: TRELLIS, Hunyuan…): su cascada de expulsión
  corría en paralelo con la de una carga, y la reserva del servicio se tomaba
  después, en el endpoint, con un hueco en el que una carga podía quedarse la
  memoria liberada.
- `make_room_for_vision`: encogía el contexto del modelo y cargaba el proyector
  con solo el lock del propio modelo; una carga podía admitirse viendo el hueco
  que el proyector iba a ocupar.
- El encogimiento de `make_room_for_whisper`: igual, sobre un vecino.
- El vigilante de VRAM: comprobaba la puerta y luego expulsaba (TOCTOU).

## Diseño

**Un único mutex: la puerta de carga de siempre**, envuelta en `VramArbiter`.
Reutilizarla es lo más seguro: todas las reglas de reentrada y de rendición
que ya se probaron en producción (v0.113.0) siguen valiendo tal cual, y no
aparece un segundo lock con el que ordenar. `_acquire_load_gate`,
`_try_load_gate`, `_holds_load_gate`, `_load_gate_busy` delegan en él.

Lo que añade a un `asyncio.Lock`:

- Reentrada para la tarea dueña (como antes).
- Espera acotada abandonable (`abort_if`, sondeada cada 0,25 s) y opción de que
  agotar la espera devuelva `False` en vez de `LoadGateTimeoutError`.
- **Registro de esperas pasivas del dueño** (`waiting_on(key)`): mientras el
  dueño espera a que se drenen las peticiones de un modelo (los tres bucles de
  drenado) o la gracia de un servicio (`pressure_evict`), lo anota. `key` es
  el id del modelo o `service:<id>`. Solo cuenta lo que registra el dueño.

### Qué se serializa (quién toma el árbitro)

| Operación | Cómo | Si no lo consigue |
|---|---|---|
| Carga (`_load_model`) | espera hasta `max_drain_timeout_s` | `LoadGateTimeoutError` (503 reintentable), igual que antes |
| `unload()` fuera de una carga | espera hasta `pressure_eviction_drain_timeout_s`; se rinde si el dueño está en una espera pasiva o el modelo deja de ser descargable | sigue **sin** árbitro (comportamiento anterior; un atasco nunca impide recuperar un worker a mano) |
| `unload(reason="idle")` | igual, pero se rinde si hay una carga en curso | se omite; el modelo cae en una vuelta posterior |
| `unload()` desde una tarea que tiene el lock de algún modelo y no el árbitro | no espera | sigue sin árbitro, como antes |
| `ensure_vram_free` | espera hasta `pressure_eviction_drain_timeout_s`; se rinde si el dueño está expulsando *ese* servicio | `InsufficientVRAMError` (el contrato del endpoint para «no se pudo liberar todo») |
| `make_room_for_vision` (encoger + cargar el proyector) | espera hasta `pressure_eviction_drain_timeout_s`; se rinde si el dueño está drenando *ese* modelo | `VisionNoRoomError` (503 `vision_no_room`, el router reintenta en otro destino) |
| `make_room_for_whisper` (encoger vecino / expulsar) | igual que visión, con el worker de whisper | no hace hueco (best effort, como cuando no había nada que encoger) |
| `evict_for_whisper` | solo si está libre (sin cambios) | `False` |
| Vigilante de VRAM | solo si está libre, alrededor de la expulsión | se salta la vuelta |
| `grow_context_back`, `_regrow_if_room` | ya lo tomaban | — |

### Reservas

El libro mayor son los locks de VRAM del `GPUManager` (`lock_vram`), que la
admisión ya descuenta (`get_free_vram` resta siempre los `service:*`). La regla:
**una operación que libera VRAM para alguien que la ocupará después de soltar
el árbitro escribe esa reserva antes de soltarlo**. Hoy aplica a
`ensure_vram_free`: reserva `service:<id>` dentro del árbitro, tanto si libera
todo como en el fallo parcial (el endpoint lo hacía después y lo sigue
haciendo; es idempotente). El proyector de visión se carga dentro del árbitro,
así que no necesita reserva: cuando se suelta, la memoria ya está ocupada y
medida. Una carga escribe su lock al terminar, dentro de la puerta, como antes.

## Orden de locks

1. **Carga**: lock del modelo M → árbitro (orden establecido).
2. **Todo lo demás**: árbitro → lock del modelo V (expulsiones, unload, resize
   dentro de visión/whisper). Al pedir el lock de V con el árbitro tomado se
   aplica la regla de siempre: `unload()` se rinde si V pasa a
   LOADING/UNLOADING bajo otro dueño.
3. Nunca se espera el árbitro teniendo el lock de un modelo salvo en (1): un
   `unload()` en esa situación sigue sin árbitro.

## Por qué no puede interbloquearse

Un ciclo necesitaría que el dueño del árbitro esperase algo que solo libera una
tarea que espera el árbitro. Lo que el dueño puede esperar:

- **El lock de un modelo V.** Lo tiene una carga de V (V está LOADING → el
  unload del dueño se rinde), un `resize_context` sin árbitro (su drenado está
  acotado por `llama_cpp_live_context_drain_timeout_s` y no pide el árbitro), o
  un unload que siguió sin árbitro (no pide el árbitro después). Nadie espera
  el árbitro teniendo un lock de modelo salvo una carga, y ese caso es el de
  LOADING.
- **El drenado de las peticiones de un modelo M.** Está registrado
  (`waiting_on(M)`). Las tareas del camino de petición que esperan el árbitro
  con una petición en vuelo en M (visión en M, whisper en M) se rinden al
  verlo; un `unload()` se rinde ante cualquier espera pasiva. Además el drenado
  tiene presupuesto (BUG-24).
- **La gracia de un servicio S** (`pressure_evict`, ≤ 30 s). Registrada
  (`waiting_on("service:S")`): un `ensure_vram_free` de S se rinde.
- **Tiempos fijos** (`_wait_for_vram_released`, `_wait_for_free_vram`,
  arranque del backend): acotados, no dependen de otras tareas.

Toda espera del árbitro por un no-dueño está acotada (como mucho
`pressure_eviction_drain_timeout_s`, o `max_drain_timeout_s` para cargas). En
el peor caso hay retraso acotado, nunca un cuelgue. La reentrada es por tarea:
si el dueño delegase en una tarea hija (`create_task`, `wait_for` en 3.11) esa
hija *no* es dueña; hoy ninguna expulsión dentro de la puerta lo hace — no lo
introduzcas sin pasar la propiedad.

## Qué nota un cliente

- Una petición con la primera imagen a un worker sin proyector, o una
  transcripción que necesita hueco, puede esperar a que termine una carga en
  curso (hasta `pressure_eviction_drain_timeout_s`) en vez de competir con
  ella; si esa carga estaba drenando el mismo worker, la imagen recibe al
  momento un 503 `vision_no_room` reintentable (el router prueba otro destino).
- Un servicio que llama a `ensure-vram` durante una carga espera a que
  termine; si la carga lo estaba expulsando, recibe `ok: false` al momento.
- Un `unload` manual durante la admisión de una carga espera a que acabe.

## Fuera de alcance (pendiente de REF-03 completo)

- El crecimiento de whisper (diarizador, peticiones extra) ocurre dentro de la
  petición, después de soltar el árbitro, sin reserva: una carga inmediata
  puede quedarse el hueco que `make_room_for_whisper` hizo.
- `resize_context` llamado desde la API (`api/internal/models.py`) o desde
  `_return_to_initial` no toma el árbitro (crecer el contexto a mano sigue
  fuera; `_return_to_initial` solo encoge).
- `grow_context_back` espera el árbitro sin regla de rendición (camino de
  petición); con el dueño drenando ese worker esperaría hasta los tiempos.
- `services.py` sigue reservando tras un `ensure_vram_free` abortado porque la
  carga expulsaba al servicio (mismo resultado que antes).
- `ensure_vram_free` puede elegir como candidato al propio servicio que lo pide
  (`get_pressure_eviction_candidates` no lo excluye); preexistente.
- Absorber en el árbitro la admisión, las tres listas de candidatos, las dos
  cascadas de expulsión y las cuatro esperas de "VRAM liberada" (REF-03).
