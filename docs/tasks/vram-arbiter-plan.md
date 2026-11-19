# Árbitro de VRAM (REF-03) — cierra lo que quedaba de BUG-23

Código: `backend/ocabra/core/vram_arbiter.py`, uso en `core/model_manager.py`.
Tests: `backend/tests/core/test_vram_arbiter.py` (primera versión),
`test_vram_characterization.py` + `vram_world.py` (caracterización),
`test_vram_arbiter_closures.py`, `test_vram_waits.py` (segunda fase).

Primera versión: commit 7aa8d4d (v0.115.0). Segunda fase (rama
`ref/vram-arbiter-full`): sección «Segunda fase» al final.

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

## Fuera de alcance de la primera versión

(Lo cerrado en la segunda fase va tachado; ver abajo.)

- ~~El crecimiento de whisper sin reserva.~~ Reserva `reserve:whisper:<id>`.
- ~~`resize_context` desde la API / `_return_to_initial` sin árbitro.~~
- ~~`grow_context_back` sin regla de rendición.~~
- `services.py` sigue reservando tras un `ensure_vram_free` abortado porque la
  carga expulsaba al servicio (fichero de la API, fuera de este trabajo).
- ~~`ensure_vram_free` puede elegir como candidato al propio servicio.~~
- Absorber en el árbitro la admisión, las dos cascadas y las esperas: hecho
  en parte (ver «Lo que queda»).

## Segunda fase (REF-03)

### Red de seguridad: caracterización

`tests/core/vram_world.py` monta un mundo falso con las dos tarjetas de
producción (0 = 3060 de 12 GB, 1 = 3090 de 24 GB): el `GPUManager` y el
`GPUScheduler` **reales** leyendo de un mundo que ocupa y libera memoria de
verdad (la publicación de hw-monitor llega con retraso configurable y NVML
«en vivo» se puede apagar), backend que reserva/libera en ese mundo y Redis
falso. `Faults` cuenta los `await` de esos colaboradores en la tarea vigilada
y hace fallar el N-ésimo con una excepción o una cancelación.

`test_vram_characterization.py` fija: GPU elegida (sin preferencia,
preferida, desbordada, partida, preset por tarjeta); orden del planificador
(on-demand > warm > pin blando, LRU dentro de cada política; nunca pin duro
ni sesión realtime; ocupados drenados y saltados; preferida imposible →
la otra); cascada de colocación forzada (LRU puro entre warm/on-demand antes
del pin blando); plan imposible sin expulsar nada; reglas de pares sin
expulsión; vigilante; `ensure_vram_free`; whisper y visión; y un fallo y una
cancelación en **cada** await de una carga que expulsa (≈30 puntos): nada
queda a medias (árbitro libre, ningún LOADING/UNLOADING, reservas = GPUs de
los LOADED, ningún proceso huérfano fuera de los ERROR) y la siguiente carga
funciona.

Bugs reales que destapó (cada uno con test que falla con el código
anterior):

1. El respaldo por `worker_pool` de `_get_pressure_eviction_candidates`
   devolvía **pines duros** cuando la tarjeta no tenía otro candidato
   LOADED (y en un modelo que prefería esa tarjeta, antes de mirar la otra).
   Latente: hoy no hay pines duros en producción.
2. Cancelar una carga mientras descargaba una víctima (o cancelar un
   unload) dejaba la víctima en **UNLOADING para siempre** (unload() la
   ignora, las peticiones esperan, solo un reinicio la recupera). Ahora:
   antes de pedir la parada vuelve a su estado; después, ERROR (que unload()
   y el respaldo de expulsión terminan).
3. La expulsión por presión del planificador (y la de servicios) tomaba la
   referencia de VRAM libre **después** del unload: con la liberación ya
   publicada esperaba los 12 s enteros a una segunda subida.
4. Con la cifra sondeada atrasada, el reintento tras una expulsión veía la
   tarjeta llena y **expulsaba también al siguiente** candidato.
5. El vigilante expulsaba con la cifra leída antes de tomar el árbitro.

### Qué se movió y a dónde

- `vram_arbiter.py` (planificador, funciones puras sobre una instantánea):
  `rank_lru_candidates`, `rank_pressure_candidates`, `is_hard_pin`;
  `wait_for_reading` (la única forma de esperar VRAM liberada).
  `ModelManager._get_eviction_candidates` / `_get_pressure_eviction_candidates`
  delegan con la misma firma.
- `ModelManager` (ejecutor, bajo el árbitro): `_drain_timeout_for` +
  `_wait_drained` sustituyen los tres bloques de drenado copiados
  (último recurso, planificador, adelanto de batch); `free_card()` dentro de
  `_admit_gpu_load` sustituye la cascada por tarjeta copiada entre el camino
  de un fragmento de tensor-split y el de una GPU.
- `GPUManager.refresh_from_live(gpu)`: lleva la cifra sondeada al valor de
  NVML ahora; `is_reservation()` / `RESERVATION_PREFIXES`.

### Esperas de «VRAM liberada»

Las cuatro esperas con tiempo fijo (`_wait_for_vram_released` 12 s, la de
después de expulsar en la admisión 12 s, `_wait_for_free_vram` 15 s con
asentamiento de 3 s, `_wait_for_shrink_published` 6 s) usan
`wait_for_reading`: terminan **en cuanto la lectura muestra la memoria**, al
dejar de moverse (cuando lo hay) o en el mismo tope de antes. Cada lectura
refresca antes la GPU desde NVML (`refresh_from_live`), igual que las
referencias tomadas antes de un unload y el comienzo de la admisión y de
`ensure_vram_free`: se decide con la tarjeta de ahora, no con la última
publicación de hw-monitor (~2 s). Sin NVML en el proceso todo sigue como
antes (sigue la cifra sondeada).

No se usa «el PID ha desaparecido de NVML» como señal: al descargar se
olvida el alias PID de NVML → worker (`forget_worker_pid`, nodos tipo
Mapache) y el PID «desaparecería» sin que el proceso hubiese salido. La
memoria libre en vivo no tiene ese problema.

La pausa de 3 s cuando no hay lectura de GPU ninguna se queda (no hay nada
que mirar). Una expulsión de servicio que no libera nada sigue esperando
sus 12 s a una subida de 1 GiB que no llega (como antes).

### Reservas

Libro mayor: los locks de VRAM del `GPUManager`. Dos clases de clave son
**reservas** (memoria comprometida antes de asignarse; todas las cifras de
libre las restan): `service:<id>` (generación de un servicio) y, nuevo,
`reserve:whisper:<id>` (crecimiento de whisper dentro de la petición:
diarizador +3,4 GB, +350 MB por petición extra). Solo `service:` deja la GPU
fuera del planificador.

`make_room_for_whisper` escribe la reserva siempre que hay crecimiento
previsto: dentro del árbitro cuando lo tomó (encoger/expulsar), o sin él si
la tarjeta ya tenía el hueco. Se retira cuando la tarjeta muestra el
crecimiento (uso en vivo ≥ uso al reservar + reserva − 256 MB), cuando
terminan las peticiones de ese whisper, cuando se descarga o a los 120 s;
varias peticiones comparten una (la mayor). `_fresh_free_vram` (decisiones
de whisper y visión) resta las reservas ajenas.

### Orden de locks (actualizado)

1. **Carga**: lock del modelo M → árbitro.
2. **Todo lo demás** (expulsiones, unload, ensure_vram_free, visión,
   whisper, **resize_context**, **grow_context_back**, vigilante): árbitro →
   lock del modelo V.
3. Nunca se espera el árbitro teniendo el lock de un modelo salvo en (1): un
   `unload()` o un `resize_context()` en esa situación siguen sin árbitro.

### Por qué sigue sin interbloquearse

El argumento de la primera versión vale tal cual; lo nuevo:

- `resize_context` con el árbitro pide el lock de M. Si M es LOADING o
  UNLOADING (una carga/descarga lo tiene, y la carga puede estar esperando
  el árbitro) se rinde al momento (`abort_if`) con RuntimeError. Si M es
  LOADED, quien tiene su lock no espera el árbitro (regla 3) y lo suelta.
- El drenado de `resize_context` es una espera pasiva registrada
  (`waiting_on(M)`): visión/whisper/grow de una petición en M se rinden.
- `grow_context_back` usa la regla del camino de petición
  (`_arbiter_for_request`): se rinde si el dueño espera a M y espera como
  mucho `pressure_eviction_drain_timeout_s`.
- `_return_to_initial` (`background=True`) no espera mientras haya una carga
  en curso; espera acotada en cualquier caso.
- La reserva de whisper no espera nada (escribe un lock y lanza un vigilante
  que solo lee).

### Qué nota un cliente

- Las cargas que expulsan tardan menos: la espera tras cada expulsión acaba
  cuando NVML muestra la memoria (antes, hasta 12 s de más si la liberación
  ya estaba publicada), y una expulsión basta donde antes la cifra atrasada
  provocaba una segunda.
- Una transcripción con diarización reserva su crecimiento: un modelo que
  se carga a la vez va a la otra tarjeta en vez de dejar al diarizador sin
  memoria (CUDA OOM).
- `POST /ocabra/models/{id}/context` espera a que termine una carga en curso
  (hasta `pressure_eviction_drain_timeout_s`) y, si no puede, 409 con
  «retry shortly».
- Una petición que necesita devolver el contexto a un worker encogido
  mientras una carga lo drena recibe al momento el 503
  `context_temporarily_reduced` reintentable (antes podía esperar hasta
  `max_drain_timeout_s`).
- Un servicio (`ensure-vram`) ya no puede ser expulsado por su propia
  petición.

### Lo que queda de REF-03 y por qué

- **Las dos cascadas siguen siendo dos.** La del planificador
  (servicios → encoger en todas → expulsar barato+encoger → presión con
  verificación previa, orden por política) y la de colocación forzada
  (abaratar → LRU educado sin PIN → último recurso) eligen víctimas
  distintas para el mismo estado (lo fijan
  `test_pressure_evicts_on_demand_before_warm_before_soft_pin` frente a
  `test_forced_gpu_evicts_by_lru_among_warm_and_on_demand_before_any_pin`).
  Unificarlas cambia a quién se expulsa en producción: es una decisión de
  política, no un refactor. Se comparte ya el drenado, la cascada por
  tarjeta y las esperas.
- **`_admit_gpu_load`, `_assign_gpus_for_load`, `_evict_*`, `_try_shrink_*`,
  `make_room_for_*`, `ensure_vram_free` siguen siendo métodos de
  `ModelManager`** (el ejecutor). Moverlos a `vram_arbiter.py` es mecánico
  pero toca ~1.500 líneas entrelazadas con la máquina de estados (estados,
  claims, eventos, `_load_model`) y con la llamada a `adopt_new_processes` /
  `_calibrate_after_load`; va con REF-05 paso 2 (`_load_model` en etapas
  `resolve_placement` / `admit`), con esta caracterización como red.
- La reserva de whisper del camino rápido (la tarjeta ya tenía hueco) se
  escribe sin el árbitro: una carga que ya midió dentro del árbitro puede
  no verla. Tomar el árbitro ahí haría esperar a toda transcripción detrás
  de las cargas.
- La siguiente publicación de hw-monitor puede reescribir el refresco en
  vivo con una muestra anterior; solo puede mostrar menos libre (conservador).
- `services.py` reserva tras un abort (ver arriba).
