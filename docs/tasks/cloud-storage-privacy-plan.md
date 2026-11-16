# Plan: discos en la nube, modos de privacidad y «Mi servidor, mis reglas»

**Fecha:** 2026-11-16 · **Estado:** 🟡 en curso

Tres piezas, en este orden: discos de RunPod (para que un pod nuevo arranque
con modelos y backends ya puestos), los dos modos de privacidad (pensando sobre
todo en nodos ajenos como los pods) y el segundo modo de federación.

---

## 1. Discos de almacenamiento en RunPod

### Lo que ofrece la API (REST v2, comprobado en `api.runpod.io/v2/openapi.json`)

| Tipo | Tamaño / precio | Restricción | Uso aquí |
|---|---|---|---|
| **Network volume** | 10–4096 GB, solo crece. $0,07/GB·mes (primer TB), $0,05 después. Se cobra hasta que se borra. | **Ata el pod a su centro de datos.** Solo Secure Cloud. Hasta 4 por pod. | El disco de caché de modelos |
| **Global volume** | Sin tamaño, se cobra por uso | Sin restricción de centro de datos; 1 por pod; cupo por cuenta | Pendiente: precio no documentado |

- `POST/GET /v2/network-volumes`, `GET/PATCH/DELETE /v2/network-volumes/{id}`.
- En el pod: `mounts.network: [{volumeId, path}]` (ruta explícita) +
  `dataCenterIds: [<el del disco>]`. Se puede montar/desmontar después
  (`UpdatePodMounts`), pero aquí se monta al crear.
- `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY` da, por centro, qué
  GPUs hay y con qué stock, y si admite discos (`networkVolumeTypes`).

### Diseño

- **El disco solo guarda lo reproducible:** modelos (`/data/models`,
  `/data/hf_cache`, `/data/ollama_models`) y backends (`/data/backends`).
  **Nunca** la base de datos, Redis, ficheros de batches ni salidas: eso sigue
  en el disco efímero del contenedor (y en modo 0 data, en RAM). Así un disco
  compartido entre pods no arrastra datos de usuarios de un pod al siguiente.
- Se monta en `/persist` y el entrypoint del nodo enlaza esas carpetas allí. Los
  backends instalados por OCI tienen rutas absolutas `/data/backends/<tipo>`,
  así que se enlaza el directorio entero y se copia una vez el llama.cpp que
  trae la imagen.
- **Un disco, un pod a la vez** (la documentación no aclara el montaje
  simultáneo y escribir a la vez desde dos nodos corrompería descargas).
- Al lanzar con disco: Secure Cloud forzado, `dataCenterIds` = el del disco y el
  catálogo de GPUs filtrado por lo que hay en ese centro.
- Coste: tamaño × $0,07/mes, prorrateado en el gasto mensual y el tope.
- UI: sección «Discos» en Nube: crear (nombre, tamaño, centro de datos — con
  las GPUs disponibles en cada uno para elegir bien), ampliar, borrar, ver qué
  pod lo usa; en «Lanzar», elegir disco.

---

## 2. Modos de privacidad

Dos modos por nodo (`privacy_mode`): **óptimo** (el de siempre) y **0 data**.
El nodo publica su modo en el heartbeat; el hub lo enseña y puede exigirlo.

### Óptimo — lo que hay, pero que de verdad se limpie

Se puede escribir lo imprescindible de forma temporal y se borra en cuanto no
hace falta. **Fallo encontrado al revisarlo:** los ficheros de batches (entrada,
salida y errores) **no se borraban nunca**: 1.964 ficheros desde el 26/07 en
`/data/openai_files`. Arreglo: caducidad de ficheros (`openai_files_ttl_hours`,
por defecto 7 días tras terminar el batch; los subidos sin batch, 7 días), con
limpieza periódica de fichero + fila. Las imágenes por URL ya caducan (1 h).

### 0 data — el nodo no guarda nada del contenido ni de quién lo pidió

Pensado para nodos que no controlas del todo (un pod en una máquina ajena).
Inventario de esquinas, de lo más obvio a lo menos:

| Esquina | En 0 data |
|---|---|
| **Estadísticas por petición** (`request_stats`: usuario, clave, IP, user agent, texto de error, modelo pedido) | No se escriben filas. Solo contadores agregados en memoria (Prometheus), sin identidades. |
| **Texto de error** que puede citar la entrada (validaciones, `log_tail` de workers) | No se guarda; en logs se recorta. |
| **Batches y `/v1/files`** (entrada y salida en disco) | Desactivados (`403` claro). Un batch exige guardar entrada y salida. |
| **Imágenes por URL** (`response_format=url` escribe a disco) | Se fuerza `b64_json`. |
| **Vídeo, música, 3D, audio** que escriben a disco para servir por URL | Lo mismo: respuesta en línea; los temporales en `tmpfs` y borrados al terminar. |
| **Subidas temporales** (transcripción, edición de imagen) | En memoria o `tmpfs`, nunca en el disco persistente ni en el disco de modelos. |
| **Logs** (`body_preview` de respuestas de peers, prompts en logs de depuración, `log_tail`) | Sin contenido en logs; nivel de llama.cpp sin volcado de peticiones. |
| **Caché de prompt / KV** (llama.cpp guarda el prompt anterior en el slot; `--slot-save-path` lo escribiría a disco) | Nunca a disco; opcional «sin caché entre peticiones» (más privado, más lento) — decisión del usuario. |
| **Base de datos y Redis del pod** | En `tmpfs`: al apagar no queda nada; Redis ya va sin persistencia. |
| **Swap y volcados de memoria** | `ulimit -c 0`; sin swap en el contenedor. |
| **Disco de modelos (network volume)** | Solo modelos y backends, nunca nada de usuarios (sección 1). |
| **Langfuse / trazas externas** | Prohibidas (hoy desactivadas aquí). |
| **Feed en vivo y panel remoto** | Sin nombre de usuario ni clave: «petición» y poco más. |
| **Agentes y MCP** (ejecuciones, resultados de herramientas) | Sin persistir ejecuciones. |
| **Federación** | El nodo 0 data no registra quién le manda nada; el hub ve que el nodo es 0 data y puede **exigirlo** para ciertas claves o grupos («esta clave solo va a nodos 0 data»). |
| **Proveedor** (RunPod) | Lo que oCabra no puede impedir: el anfitrión podría leer memoria o disco. 0 data = oCabra no retiene nada, **no** confidencialidad frente al anfitrión. Por eso, en pods 0 data: Secure Cloud por defecto (centros certificados) y aviso explícito en la UI. |
| **Tokens y secretos en el pod** (`HF_TOKEN`, invitación) | Variables de entorno que ve el proveedor: no pasar `HF_TOKEN` salvo que haga falta; la invitación es de un solo uso. |
| **Cabecera de respuesta** | `X-Ocabra-Privacy: zero-data` para que el cliente sepa dónde ha caído. |

---

## 3. Federación «Mi servidor, mis reglas»

Opciones pedidas por el usuario, más las que salen al pensarlo:

| Opción | Qué hace |
|---|---|
| **Yo primero** | Las peticiones locales tienen prioridad sobre las remotas (cola y slots). |
| **Cargar modelos o solo lo caliente** | Si un peer puede provocar cargas aquí o solo usar lo ya cargado. |
| **Límite de peticiones por hora** | Por peer. |
| **Límite de vatios al mes** | Energía consumida sirviendo peticiones de peers (ya medimos energía por petición). |
| **Permitir descargar modelos** | Que el peer pueda pedir descargas aquí. |
| **Permitir administrar opciones** | Panel remoto con escritura o solo lectura. |
| *Añadidas:* modelos ofrecidos | Lista blanca de modelos que los peers ven. |
| *Añadidas:* franjas horarias | Solo de noche, etc. |
| *Añadidas:* expulsiones | Si una carga pedida por un peer puede expulsar un modelo local en uso (nunca, solo ociosos, sí). |
| *Añadidas:* privacidad exigida | Aceptar del peer solo si viene como 0 data, o servir siempre en 0 data. |

Se diseñará con detalle después de 1 y 2.
