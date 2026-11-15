# Plan: federación con túnel inverso, solicitudes de unión y nodos RunPod

**Fecha:** 2026-11-15
**Estado:** 🟡 en curso — ronda 1
**Relacionado:** [`federation-plan.md`](federation-plan.md),
[`federation-modernizacion-plan.md`](federation-modernizacion-plan.md)

---

## Objetivo

Ampliar temporalmente la capacidad o la redundancia de oCabra con nodos
remotos: el de **otra persona en otra ubicación** (hoy `mapache`) y **pods de
RunPod** levantados bajo demanda. Para que esto sea práctico:

1. **Basta con que uno de los dos lados tenga URL alcanzable.** El nodo remoto
   abre una conexión saliente (`wss://`) hacia el que sí la tiene, y por ella
   le llegan las peticiones. Un pod de RunPod no necesita exponer ningún puerto.
2. **Unirse se pide, no se configura a mano.** Desde el nodo B se pide unirse
   dando la URL de A; en A aparece la solicitud para aceptarla o denegarla.
3. **RunPod se gestiona desde la UI**: elegir GPU, lanzar, ver coste, apagar.

Criterio del usuario (2026-11-15):

> *"con que una lo tenga debería de valer; ej. el nodo remoto es el que inicia
> la conexión con nuestro oCabra por https y no necesita tener url abierta"* ·
> *"que se pueda mandar solicitud de unirse a la federación: desde el segundo
> nodo pido federar con la url del primero y en el primero sale la solicitud
> para aceptar/denegar"*

---

## Diseño

### Túnel = un transporte de httpx

Hoy `FederationManager` habla con cada peer con un `httpx.AsyncClient`
(heartbeat, `proxy_request`, `proxy_stream`, `proxy_multipart`, carga/descarga
remota). El túnel se implementa como un `httpx.AsyncBaseTransport`: para un peer
en modo `tunnel`, su cliente usa ese transporte en vez de la red. **El
enrutado, el fallback a local, el heartbeat y las operaciones remotas no
cambian**; solo cambia por dónde viajan los bytes.

- El nodo B (agente) abre `wss://A/ocabra/federation/tunnel` y se presenta con
  un `HELLO` que lleva su token de túnel.
- Por el único WebSocket se multiplexan N peticiones a la vez. Tramas binarias
  `[tipo:1][stream_id:4][payload]`:
  `REQ_HEAD` (JSON: method, path, headers) → `REQ_BODY`* → `REQ_END`;
  `RESP_HEAD` (JSON: status, headers) → `RESP_BODY`* → `RESP_END` | `ERROR`;
  `CANCEL` en cualquier sentido. El cuerpo va troceado (64 KiB) así que el
  streaming SSE y las subidas multipart (audio) funcionan tal cual.
- B ejecuta cada petición contra **su propia API local**
  (`http://127.0.0.1:8000`) sustituyendo el `Authorization` por una API key
  local propia del enlace. **El ACL lo aplica B con su sistema de auth normal**:
  lo que B ofrece se decide con los grupos y el rol de esa key.
- Si el WebSocket se cae, las peticiones en vuelo fallan con error de red →
  entra el fallback a local que ya existe (no-streaming). El peer pasa a
  offline al instante, no tras 3 heartbeats.
- La API corre con `--workers 1`, así que el registro de túneles vive en
  memoria del proceso.

### Unirse: invitaciones y solicitudes

Dos caminos, el mismo resultado (un peer `tunnel` en A y un "enlace" en B):

| Camino | Quién | Aprobación |
|---|---|---|
| **Solicitud** | B pide unirse con la URL de A | manual en A (aceptar/denegar) |
| **Invitación** | A genera un código y se lo pasa a B (o lo inyecta en un pod) | automática si la invitación lo dice |

Flujo de la solicitud:

1. B → `POST A/ocabra/federation/join-requests` (sin auth) con su `node_id`,
   nombre, versión, GPUs, un mensaje opcional y **qué ofrece** (inferencia /
   gestión completa). A responde `request_id` + `poll_secret` y un
   **código de verificación** corto (p. ej. `4F-92-AC`) que ambos lados muestran
   para comprobarlo por otro canal.
2. En A aparece en *Solicitudes pendientes*. El admin acepta (eligiendo el
   nombre del peer) o deniega.
3. B consulta `GET A/ocabra/federation/join-requests/{id}` con su
   `poll_secret`; al aceptarse recibe **una vez** el token del túnel y conecta.

Protección del endpoint público: desactivado por defecto
(`federation_accept_join_requests`), límite de pendientes, caducidad de 24 h,
límite por IP; las invitaciones funcionan aunque las solicitudes estén
desactivadas. A guarda solo el hash del token.

### Direccionalidad

Un enlace hace que **A use los modelos de B**. Si B quiere usar los de A, es
otra relación (B añade a A como peer HTTP normal con una key que A le dé, ya
que A es alcanzable) — se deja para una ronda posterior como "federación
mutua" con un clic.

### RunPod

- Cliente de la REST API v2 (la GraphQL se retira a principios de 2027): tipos
  de GPU con precio, crear / consultar / parar / terminar pod.
- API key de RunPod cifrada en BD; tabla `cloud_nodes` (proveedor, pod_id,
  peer, GPU, $/h, creado, TTL, estado, coste acumulado).
- Lanzar = crear invitación de un uso + crear el pod con
  `FEDERATION_JOIN_URL` y `FEDERATION_JOIN_INVITE` en el entorno (ya
  implementado: el nodo se une solo al arrancar, ver `_bootstrap_join`). El pod
  arranca, se une y aparece como peer.
- Salvaguardas: TTL máximo, apagado por inactividad, conciliación con la API de
  RunPod al arrancar (pods huérfanos), autodestrucción del nodo si pierde el
  túnel X minutos.
- Necesita una imagen `ocabra-node` (api + Postgres + Redis en un solo
  contenedor, con llama.cpp) publicada en un registro.

### UI

Federación pasa a ser una página propia con: **Nodos** (tarjetas con tipo de
conexión —HTTP directo / túnel / RunPod—, dueño/notas, estado, GPUs, modelos
cargados, tráfico), **Solicitudes** (con contador), **Invitar**, **Unirse a
otro nodo** (lado B: URL, estado de la solicitud, estado del túnel) y **Nube**
(RunPod).

---

## Decisiones de la ronda 1 (2026-11-15)

- **Dos modos de federación** (criterio del usuario): *"Soy parte del
  enjambre"* = mutuo y con control completo, y *"Mi server mis reglas"* = cada
  lado elige qué ofrece y qué permite. **Solo el enjambre está hecho**; el otro
  se diseñará viendo cómo se usa el primero. En la UI aparece como
  "Próximamente".
- **El túnel es simétrico.** Una sola conexión; el que marca usa ids de stream
  pares y el que acepta impares, y cada lado puede pedirle cosas al otro. Así
  la federación mutua no necesita intercambiar URLs ni API keys.
- **Identidad de servicio por enlace.** Al aceptarse, cada lado crea un
  usuario `federation:<nodo>` (rol `system_admin`, contraseña aleatoria nunca
  mostrada) cuya key ejecuta lo que pide el otro. El acceso real lo limita la
  lista de rutas del túnel (`/v1`, `/api`, heartbeat y, en modo completo,
  `/ocabra/models`, `/ocabra/downloads`, `/ocabra/gpus`): nunca usuarios,
  config ni keys. Borrar el peer borra el usuario.
- **Anti-bucles.** Lo que llega de otro nodo lleva `x-ocabra-federation-hop`;
  mientras se atiende, `get_remote_models()` devuelve vacío y la petición no
  sale a un tercero.
- **`node_id` estable**: si no está configurado se genera y se persiste.
- Los peers HTTP de antes (`mapache` hoy) siguen funcionando igual, en modo
  `manual`.

## Fases de la ronda 1

| # | Qué | Estado |
|---|---|---|
| 1 | Túnel simétrico: protocolo, transporte httpx, dialer con reconexión | ✅ 0.82.0 |
| 2 | Solicitudes de unión + invitaciones + identidad de servicio + anti-bucles | ✅ 0.82.0 |
| 3 | Página de Federación propia (sale de Settings) | ✅ 0.82.0 |
| 4 | Prueba real con `mapache` (el usuario actualiza su oCabra desde git) | ⬜ |
| 5 | Imagen `ocabra-node` | ⬜ |
| 6 | RunPod: cliente, `cloud_nodes`, salvaguardas, UI | ⬜ |
| 7 | Prueba real con un pod (3090 Community ≈ $0.22/h) | ⬜ |

## Verificación (0.82.0)

- 30+ tests del túnel en memoria y con uvicorn + `websockets` reales
  (multipart, SSE, cancelación, reconexión, ambos sentidos).
- 5 tests del flujo completo con dos `FederationManager` sobre Postgres
  (`tests/test_federation_join.py`): solicitud → aceptar → túnel mutuo,
  invitación, denegación, expulsión (el otro queda `rejected`), guardas del
  endpoint público. Se ejecutan dentro del contenedor (`requires_db`).
- Suite completa: host 2457 ok; contenedor con Postgres, mismos 11 fallos que
  HEAD (dependen del entorno: GPUs reales, scripts fuera de `backend/`).
- La URL pública deja pasar el upgrade a WebSocket (openresty → Caddy).

## Deuda / pendiente

- **Contrapresión**: un stream lento bloquea el socket entero cuando su cola
  (256 tramas) se llena. Aceptable para v1; si molesta, ventana por stream.
- **Despedida**: si B se sale, A no se entera hasta que el túnel cae (lo ve
  offline, no "se fue"). Falta una trama BYE.
- **El heartbeat lista todos los modelos cargados**, aunque en "mi server mis
  reglas" habrá que filtrarlos por lo que se ofrece.
- **Una petición no salta a un tercer nodo** (anti-bucles). Si algún día
  interesa un enjambre de >2 nodos con reenvío, habrá que pasar a TTL de saltos.
- Sesiones de `/v1/realtime`, `/v1/batches` y `/v1/files` siguen sin federarse
  (ya estaba en el plan de modernización).

## Pendiente de confirmar

- Registro para `ocabra-node` (GHCR / Docker Hub, público o privado).
