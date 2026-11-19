# Plan: Sistema de Autenticación, Usuarios y Grupos

**Fecha:** 2026-04-04
**Estado:** PENDIENTE DE IMPLEMENTACIÓN
**Prioridad:** CRÍTICA — bloqueante para exposición pública

---

## Resumen

oCabra pasa a tener su propio sistema de auth, reemplazando completamente la capa que
antes hacía LiteLLM. Tres roles jerárquicos, API keys por usuario para OpenAI/Ollama,
grupos que controlan qué modelos ve cada usuario, y dashboard protegido con sesión JWT.

---

## Roles

Jerárquicos: cada rol incluye los permisos del anterior.

| Capacidad | `user` | `model_manager` | `system_admin` |
|-----------|--------|-----------------|----------------|
| Login al dashboard | ✓ | ✓ | ✓ |
| Cargar / descargar modelos manualmente | ✓ | ✓ | ✓ |
| Ver stats propias (tokens, requests, coste energético) | ✓ | ✓ | ✓ |
| Ver stats globales de GPU (agregadas) | ✓ | ✓ | ✓ |
| Ver stats globales de requests (todos los usuarios) | — | ✓ | ✓ |
| Buscar y descargar modelos de internet (HF / Ollama) | — | ✓ | ✓ |
| Borrar modelos del disco | — | ✓ | ✓ |
| Compilar engines TRT-LLM | — | ✓ | ✓ |
| Gestionar usuarios (crear, editar, desactivar) | — | — | ✓ |
| Gestionar grupos (crear, editar, asignar modelos) | — | — | ✓ |
| Resetear contraseña de otro usuario | — | — | ✓ |
| Ver y revocar API keys de otros usuarios | — | — | ✓ |
| Cambiar configuración del servidor | — | — | ✓ |
| Cambiar propia contraseña | ✓ | ✓ | ✓ |
| Gestionar propias API keys | ✓ | ✓ | ✓ |

---

## Autenticación

### Dashboard — sesión JWT

- Cookie HTTP-only `ocabra_session` firmada con `settings.jwt_secret`.
- Payload: `{ sub: user_id, role, iat, exp }`.
- Duración por defecto: **24 horas** (configurable en Settings por admin).
- Con "Recordarme": **30 días**.
- Refresh automático si la sesión tiene menos de 1h de vida al hacer una petición.
- Logout: borrar cookie + añadir JWT a lista de revocados en Redis (TTL = exp restante).

### APIs OpenAI / Ollama — API keys

- Header: `Authorization: Bearer sk-ocabra-<32 chars aleatorios>`.
- La key se muestra **una sola vez** al crearla; en BD solo se guarda el hash SHA-256.
- Se guarda un prefijo de 10 chars para mostrar en la lista (`sk-ocabra-aBc3…`).
- Identifican al usuario; se aplican sus restricciones de grupo.
- Caducidad: configurable por el usuario (default 6 meses, o "sin caducidad").
- Un usuario puede tener múltiples keys con nombre/descripción libre.
- Los admins pueden ver las keys (prefijo + metadatos, nunca el valor) y revocarlas.

### Modo sin API key (legacy / LAN)

- Server settings `require_api_key_openai` y `require_api_key_ollama` (ambos `true` por defecto).
- Si está desactivado, las peticiones sin `Authorization` se tratan como usuario `anonymous`.
- `anonymous` pertenece implícitamente al grupo `default` y **solo** puede usar esos modelos.
- Si llega una key válida, se aplican los grupos del usuario (aunque el modo esté desactivado).

### Endpoints públicos (sin auth)

- `POST /ocabra/auth/login`
- `POST /ocabra/auth/logout`
- `GET /health`, `GET /ready`
- `GET /metrics` (Prometheus scrapea sin auth)

---

## Usuarios

### Primer admin

Al arrancar con BD vacía, oCabra crea automáticamente el primer admin con:

```
OCABRA_ADMIN_USER=ocabra   (default: ocabra)
OCABRA_ADMIN_PASS=ocabra   (default: ocabra)
```

Si el usuario ya existe, no hace nada (idempotente). No se fuerza cambio de contraseña.

### Contraseñas

- Hash con **bcrypt** (coste 12).
- Los admins pueden resetear la contraseña de cualquier usuario (sin conocer la actual).
- Los usuarios pueden cambiar su propia contraseña (requieren la actual).

---

## Grupos y modelos

### Grupo `default`

- Creado automáticamente al inicializar la BD.
- Todos los usuarios son miembros implícitos y no pueden ser removidos de él.
- Los admins gestionan qué modelos contiene desde la UI (Settings → Groups).

### Asignación de modelos

- Al registrar/configurar un modelo, el modal muestra un selector de grupos
  con `default` pre-marcado.
- Un modelo puede estar en varios grupos.
- Si se elimina un modelo del sistema, se elimina de todos los grupos automáticamente.

### Visibilidad de modelos

- `GET /v1/models` y `GET /api/tags` filtran por la unión de grupos del usuario.
- Llamar a un modelo al que no se tiene acceso → `404 Model not found`
  (no `403`, para no revelar existencia del modelo).
- Los admins ven todos los modelos independientemente de grupos.

### Membresía

- Un usuario puede pertenecer a múltiples grupos; accede a la **unión** de sus modelos.
- El grupo `default` no se puede eliminar.

---

## Esquema de base de datos

### Tabla `users`

```sql
id            UUID PK default gen_random_uuid()
username      TEXT UNIQUE NOT NULL
email         TEXT UNIQUE          -- opcional
hashed_password TEXT NOT NULL
role          TEXT NOT NULL        -- 'user' | 'model_manager' | 'system_admin'
is_active     BOOLEAN DEFAULT true
created_at    TIMESTAMPTZ DEFAULT now()
updated_at    TIMESTAMPTZ DEFAULT now()
```

### Tabla `api_keys`

```sql
id            UUID PK default gen_random_uuid()
user_id       UUID FK users(id) ON DELETE CASCADE
name          TEXT NOT NULL        -- label del usuario
key_hash      TEXT UNIQUE NOT NULL -- SHA-256 del valor real
key_prefix    TEXT NOT NULL        -- "sk-ocabra-aBc3" para mostrar en UI
expires_at    TIMESTAMPTZ          -- NULL = sin caducidad
last_used_at  TIMESTAMPTZ
is_revoked    BOOLEAN DEFAULT false
created_at    TIMESTAMPTZ DEFAULT now()
```

### Tabla `groups`

```sql
id            UUID PK default gen_random_uuid()
name          TEXT UNIQUE NOT NULL
description   TEXT DEFAULT ''
is_default    BOOLEAN DEFAULT false  -- solo uno puede ser true
created_at    TIMESTAMPTZ DEFAULT now()
```

### Tabla `user_groups` (many-to-many)

```sql
user_id       UUID FK users(id) ON DELETE CASCADE
group_id      UUID FK groups(id) ON DELETE CASCADE
PRIMARY KEY (user_id, group_id)
```

### Tabla `group_models` (many-to-many)

```sql
group_id      UUID FK groups(id) ON DELETE CASCADE
model_id      TEXT NOT NULL   -- canonical: "vllm/Qwen/Qwen3-8B"
PRIMARY KEY (group_id, model_id)
```

### Cambio en `request_stats`

Añadir columna `user_id UUID FK users(id) ON DELETE SET NULL` para stats por usuario.

---

## Endpoints nuevos

### Auth (público)

```
POST  /ocabra/auth/login           { username, password } → cookie + { user }
POST  /ocabra/auth/logout          → borrar cookie
GET   /ocabra/auth/me              → { user } con rol y grupos
POST  /ocabra/auth/refresh         → renovar cookie si próxima a expirar
PUT   /ocabra/auth/password        { current_password, new_password }
```

### Usuarios (system_admin)

```
GET    /ocabra/users               lista de usuarios
POST   /ocabra/users               { username, password, role, email? }
GET    /ocabra/users/{id}
PATCH  /ocabra/users/{id}          { role?, is_active?, email? }
DELETE /ocabra/users/{id}
POST   /ocabra/users/{id}/reset-password   { new_password }
GET    /ocabra/users/{id}/keys     ver keys de otro usuario (admin)
DELETE /ocabra/users/{id}/keys/{key_id}   revocar key de otro (admin)
```

### API keys (propias — cualquier rol)

```
GET    /ocabra/auth/keys           lista mis keys (prefijo, nombre, exp, last_used)
POST   /ocabra/auth/keys           { name, expires_in_days? } → { key: "sk-ocabra-..." }
DELETE /ocabra/auth/keys/{key_id}  revocar mi key
```

### Grupos (system_admin)

```
GET    /ocabra/groups
POST   /ocabra/groups              { name, description? }
PATCH  /ocabra/groups/{id}
DELETE /ocabra/groups/{id}         (no permitido si is_default=true)

GET    /ocabra/groups/{id}/models
POST   /ocabra/groups/{id}/models  { model_id }
DELETE /ocabra/groups/{id}/models/{model_id}

GET    /ocabra/groups/{id}/members
POST   /ocabra/groups/{id}/members { user_id }
DELETE /ocabra/groups/{id}/members/{user_id}   (no permitido para default)
```

---

## Cambios en endpoints existentes

### `/ocabra/*` — protección por rol

Cada endpoint recibe `Depends(require_role(...))`:

| Endpoints | Rol mínimo |
|-----------|-----------|
| `GET /ocabra/gpus`, `GET /ocabra/stats/*` (globales GPU) | `user` |
| `POST .../load`, `POST .../unload` | `user` |
| `GET /ocabra/stats/requests` (todos los usuarios) | `model_manager` |
| `GET /ocabra/registry/*`, `POST /ocabra/downloads` | `model_manager` |
| `DELETE /ocabra/models/*` | `model_manager` |
| `POST /ocabra/trtllm/compile` | `model_manager` |
| `GET/PATCH /ocabra/config` | `system_admin` |
| `GET/POST /ocabra/users` | `system_admin` |
| `GET/POST /ocabra/groups` | `system_admin` |

### `/v1/*` y `/api/*` — filtrado por grupos

- `GET /v1/models` y `GET /api/tags`: solo devuelven modelos del usuario.
- `POST /v1/chat/completions` etc.: si el `model` no está en los grupos del usuario → `404`.
- `user_id` del token se inyecta en `request_stats` al persistir.

---

## Archivos a crear / modificar

### Backend — nuevos

| Archivo | Descripción |
|---------|-------------|
| `backend/ocabra/db/auth.py` | ORM: `User`, `ApiKey`, `Group`, `UserGroup`, `GroupModel` |
| `backend/ocabra/api/internal/auth.py` | Endpoints login/logout/me/refresh/password/keys |
| `backend/ocabra/api/internal/users.py` | CRUD usuarios + reset password |
| `backend/ocabra/api/internal/groups.py` | CRUD grupos + modelos + miembros |
| `backend/ocabra/core/auth_manager.py` | JWT gen/val, bcrypt, seed primer admin, key hashing |
| `backend/ocabra/api/_deps_auth.py` | `get_current_user()`, `require_role()`, `get_user_groups()` |
| `backend/alembic/versions/0006_auth_users_groups.py` | Migración |

### Backend — modificados

| Archivo | Cambio |
|---------|--------|
| `backend/ocabra/main.py` | Incluir routers auth/users/groups; seed primer admin en lifespan |
| `backend/ocabra/config.py` | `jwt_secret`, `jwt_ttl_hours`, `jwt_remember_days`, `require_api_key_openai`, `require_api_key_ollama`, `ocabra_admin_user`, `ocabra_admin_pass` |
| `backend/ocabra/api/internal/*.py` (todos) | Añadir `Depends(require_role(...))` |
| `backend/ocabra/api/openai/_deps.py` | Auth por API key + filtro por grupos |
| `backend/ocabra/api/ollama/_shared.py` | Auth por API key + filtro por grupos |
| `backend/ocabra/stats/collector.py` | Inyectar `user_id` en `request_stats` |
| `backend/ocabra/db/stats.py` | Añadir `user_id` a `RequestStat` |
| `docker-compose.yml` | Variables `OCABRA_ADMIN_USER/PASS`, `JWT_SECRET`, `REQUIRE_API_KEY_*` |
| `.env.example` | Documentar variables nuevas |
| `backend/pyproject.toml` | `passlib[bcrypt]`, `PyJWT` |

### Frontend — nuevos

| Archivo | Descripción |
|---------|-------------|
| `frontend/src/pages/Login.tsx` | Formulario login |
| `frontend/src/pages/Users.tsx` | Gestión de usuarios (admin) |
| `frontend/src/pages/Groups.tsx` | Gestión de grupos y modelos (admin) |
| `frontend/src/stores/authStore.ts` | Zustand: usuario actual, rol, estado auth |
| `frontend/src/hooks/useAuth.ts` | Hook: `useCurrentUser()`, `useRequireRole()` |
| `frontend/src/components/auth/ApiKeyManager.tsx` | Crear/listar/revocar propias keys |
| `frontend/src/components/auth/ProtectedRoute.tsx` | Wrapper de rutas autenticadas |

### Frontend — modificados

| Archivo | Cambio |
|---------|--------|
| `frontend/src/App.tsx` | Rutas protegidas + `/login` + `/users` + `/groups` |
| `frontend/src/api/client.ts` | Auth headers, endpoints auth/users/groups/keys |
| `frontend/src/types/index.ts` | Tipos `User`, `ApiKey`, `Group` |
| `frontend/src/pages/Settings.tsx` | Sección API keys + links a Users/Groups para admin |
| `frontend/src/pages/Stats.tsx` | Filtro own vs global según rol |
| `frontend/src/pages/Explore.tsx` | Ocultar si rol = `user` |
| `frontend/src/components/models/ModelConfigModal.tsx` | Selector de grupos al registrar |
| `frontend/src/components/layout/Sidebar.tsx` | Mostrar usuario actual + logout; condicionar nav por rol |

---

## Fases de implementación

### Fase 1 — Fundación backend (bloquea todo)

1. Migración Alembic `0006_auth_users_groups`
2. `db/auth.py` — modelos SQLAlchemy
3. `core/auth_manager.py` — bcrypt, JWT, key hashing, seed primer admin
4. `api/_deps_auth.py` — `get_current_user()`, `require_role()`, anonymous context
5. `api/internal/auth.py` — login, logout, me, refresh, password change, API keys propias
6. Wiring en `main.py` + seed en lifespan + variables de config

### Fase 2 — Protección de endpoints existentes

7. `Depends(require_role(...))` en todos los routers `/ocabra/*`
8. API key auth en `openai/_deps.py` y `ollama/_shared.py`
9. Filtrado de modelos por grupos en `/v1/models`, `/api/tags` y en inferencia
10. Inyección de `user_id` en `request_stats`

### Fase 3 — Gestión de usuarios y grupos

11. `api/internal/users.py` — CRUD + reset password + ver/revocar keys ajenas
12. `api/internal/groups.py` — CRUD grupos + modelos + miembros
13. Selector de grupos en el modal de configuración de modelo

### Fase 4 — Frontend

14. `Login.tsx` + `authStore.ts` + `useAuth.ts` + `ProtectedRoute.tsx`
15. Sidebar: usuario actual + logout + navegación por rol
16. `Settings.tsx`: sección API keys (`ApiKeyManager`)
17. `Users.tsx` (admin)
18. `Groups.tsx` (admin)
19. `Stats.tsx`: vista propia (tokens/coste del usuario) + GPU global
20. `Explore.tsx`: ocultación por rol; `ModelConfigModal`: selector de grupos

### Fase 5 — Tests

21. `test_auth.py`: login correcto/incorrecto, JWT caducado, 401/403 por rol
22. `test_api_keys.py`: crear, usar, revocar, caducidad
23. `test_groups.py`: asignación de modelos, filtrado en `/v1/models`, 404 en acceso no autorizado
24. `test_anonymous.py`: modo sin key, acceso solo a default group

---

## Notas de implementación

### JWT secret

`jwt_secret` debe generarse automáticamente al arrancar si no está en `.env`:

```python
# En config.py
jwt_secret: str = Field(default_factory=lambda: secrets.token_hex(32))
```

Si se reinicia sin la variable fijada, todas las sesiones activas se invalidan.
**Recomendación al usuario:** fijar `JWT_SECRET` en `.env` para persistir sesiones entre reinicios.

### Hashing de API keys

```
valor_real = "sk-ocabra-" + secrets.token_urlsafe(24)   # mostrar solo al crear
hash_bd    = hashlib.sha256(valor_real.encode()).hexdigest()
prefijo    = valor_real[:18] + "…"                       # "sk-ocabra-aBc3Xy…"
```

### Modelo `anonymous`

No existe en BD. Es un contexto sintético creado en `get_current_user()` cuando
`require_api_key=false` y no llega `Authorization`:

```python
ANONYMOUS_CONTEXT = UserContext(
    user_id=None,
    role="user",
    groups=["default"],  # solo default group
    is_anonymous=True,
)
```

### Grupo `default` en `/v1/models` y `/api/tags`

El filtrado ocurre en la capa de deps, no en cada endpoint individualmente.
`get_current_user()` devuelve el set de `model_ids` accesibles; el listado filtra por ese set.

### WebSocket

El WS en `/ocabra/ws` recibe el JWT via cookie (mismo origen, se envía automáticamente).
El handshake valida la cookie antes de aceptar la conexión.
