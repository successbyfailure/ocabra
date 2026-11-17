# Federación: catálogo sumado y estrategias de reparto por modelo

Decidido con el usuario el 2026-11-17. Las estrategias se van añadiendo **por
orden de necesidad y casos de uso**, no todas de golpe.

## Principios

1. **Los nodos suman sus modelos.** Cada nodo ve sus modelos y los que ofrece
   cada vecino, calientes o en frío. Un modelo que solo existe en otro nodo es
   de primera clase: aparece en los listados, entra en los grupos de acceso y
   puede ser destino de un router.
2. **El reparto es del modelo, no del peldaño.** Dónde se sirve un modelo no
   depende de su escalera de contexto. El antiguo «Ejecución» por peldaño
   (`execution_policy`) se retiró (migración 0046). Sus 5 modelos en «Solo
   local» pasaron a `local_only`, y «Remota» no tenía usuarios.
3. **Los routers apuntan a modelos, no a nodos.** Dónde corre cada destino lo
   decide la estrategia de ese destino.

## Hecho (0.97.0)

- **Catálogo.** El heartbeat anuncia también los modelos en frío
  (`configured`/`unloaded`/`loading`) con sus `capabilities`, salvo que las
  reglas del vecino digan «solo lo caliente» (`allow_loads=false`).
  - `FederationManager.get_remote_catalog()` devuelve todo lo ofrecido;
    `get_remote_models()` sigue siendo solo lo cargado.
  - `remote_model_for(name)` resuelve un id o un perfil anunciado.
- **Modelos solo remotos.** `resolve_federated` añade el paso 2b: si no lo
  tenemos y un vecino lo ofrece en frío, va al menos cargado.
- **Routers.** Un destino que no es un perfil local, pero que algún vecino en
  línea ofrece, es candidato (con filtro por las capacidades anunciadas). La
  validación al guardar lo acepta (`ProfileRegistry.remote_target_lookup`).
  Mientras su nodo no está en línea, el router lo salta.
- **Estrategias** (`core/federation_policy.py`, en `server_config`
  `federation_model_policies`):
  - `auto`: lo de antes (lo caliente gana, sesgo local −5).
  - `local_only`: nunca sale.
  - `prefer`: nodos en orden (nombre o id de vecino, o `local`). El primero en
    línea que lo ofrezca lo sirve, aunque esté en frío. `fallback=false` da
    503 `preferred_nodes_unavailable` con `Retry-After`.
  - Una petición que llega de un vecino (hop) ignora nuestra estrategia: se
    sirve aquí.
  - `X-Ocabra-Node` sigue ganando a todo.
- **API:** `GET /ocabra/federation/strategies[?model_id=]` y
  `PUT /ocabra/federation/strategies` (rol `model_manager`).
- **UI:** sección «Reparto» en el editor del modelo; Routers ofrece los modelos
  remotos como destino y muestra «remoto · cargado / en frío» y en qué nodo.
- **Arreglos de paso:**
  - Chat y completions en streaming mandaban al vecino el nombre del router.
  - El reintento por id canónico usaba el primer destino del router, no el
    elegido.
  - `/v1/models` y `/api/tags` listaban modelos remotos sin mirar los grupos.

## Siguientes estrategias (cuando haya caso de uso)

- **Afinidad por API key**: la misma clave vuelve al mismo nodo mientras esté
  sano, para aprovechar la caché del prompt. Necesita una tabla
  clave→nodo con caducidad y una regla para cuando ese nodo se satura.
- **Coste de expulsión**: hoy `auto` no sabe que cargar aquí echa a otro modelo
  (el ping-pong nemotron↔gemma4 del 16/11). Una estrategia que prefiera el
  vecino cuando aquí haría falta expulsar.
- **Reparto por carga entre varios nodos que lo tienen caliente** (round-robin
  o por menor cola), más allá del sesgo local fijo.

## Esquinas apuntadas

- Un modelo solo remoto cuyo vecino falla a mitad: el paso local da 404 «no
  existe» en vez de 503. Habría que distinguir «existe en la federación pero
  no está disponible».
- `prefer` con un vecino en frío y `allow_loads` del vecino desactivado a la
  vez: el heartbeat ya no lo anuncia, así que salta al siguiente nodo. Bien,
  pero el editor lo muestra «no lo ofrece» sin decir por qué.
- Pods: su `peer_id` cambia en cada lanzamiento. La estrategia guarda nombres
  (o ids), así que un pod relanzado con otro nombre no casa. Conviene nombres
  estables para los pods que se usen en `prefer`.
- Estrategias para modelos solo remotos (elegir entre dos pods): la API lo
  admite (clave = id canónico), pero el editor no abre modelos sin config
  local.
- Grupos: un modelo solo remoto necesita estar en el grupo del usuario, salvo
  que se acceda a través de un router.
