# Plan: distribución y actualización de backends · `llama_cpp_mks`

**Fecha de apertura:** 2026-11-06
**Estado:** 🔵 diseño — sin trabajo iniciado
**Relacionado:** [`modular-backends-plan.md`](modular-backends-plan.md) (Bloque 15),
[`llama-cpp-elastic-kv-plan.md`](llama-cpp-elastic-kv-plan.md)

---

## Decisión de partida

**El llama.cpp custom es un backend distinto del upstream**, no una variante del
mismo. Decisión del usuario (2026-11-06), tomada conociendo el coste: implica un
`backend_type` nuevo (`llama_cpp_mks`) y auditar los sitios que hoy comparan
contra `"llama_cpp"` literal. A cambio permite tener los dos instalados a la vez,
comparar A/B y hacer rollback por modelo.

## Estado real de la distribución (verificado 2026-11-06)

| Pieza | Estado |
|---|---|
| Instalación **desde fuente** (venv + pip, apt, `git_repo`/`git_ref`, `post_install_script`, `extra_bins` → `metadata.json`) | ✅ funciona |
| Descubrimiento de lo ya instalado, detección "built-in" (imagen gorda), desinstalación, locks anti-concurrencia | ✅ funciona |
| Distribución por **OCI** | ❌ declarada y muerta. Todos los backends traen `oci_image="ghcr.io/ocabra/backend-*"` + `oci_tags`, pero `BackendInstaller.install(method="oci")` lanza `NotImplementedError` (`core/backend_installer.py:306`) y la API responde 501 (`api/internal/backends.py:174`). `ghcr.io/v2/ocabra/backend-llama-cpp/tags/list` → 401 |
| **CI** que construya esas imágenes | ❌ no existe ninguna configuración de CI en el repo |
| Flujo de **actualización** de un backend | ❌ no hay endpoint. Los endpoints son listar, ver, instalar, desinstalar y logs. Actualizar = desinstalar + reinstalar |
| Reporte de **versión declarada vs instalada** | ❌ no existe. Ver el incidente de abajo |

### Evidencia de por qué hace falta "declarada vs instalada"

El 2026-11-06 se descubrió que `vllm_backend.install_spec` declaraba
`vllm==0.29.0` mientras el venv tenía **0.32.0**: se actualizó fuera de banda y
el pin nunca se subió. Una reinstalación habría **degradado** el backend dos
versiones menores en silencio. Se reconcilió en v0.31.3 (`43d24ca`), pero nada
impide que vuelva a pasar con cualquier otro backend: la UI no compara lo
declarado con lo instalado.

## Paso 0 — bloqueante: el trabajo custom no está en ningún repositorio

Antes de cualquier diseño, hay un riesgo de pérdida total. La personalización de
llama.cpp (KV elástica: `cache_reuse`, `n_cache_reuse`, `has_active_slot`,
endpoints `get_slots`/`post_slots`) son **226 inserciones y 50 borrados en 5
ficheros de `tools/server/`** que existen **solo como cambios sin commitear**:

- `/docker/llama.cpp-ocabra` — rama `ocabra/elastic-kv`, 7 ficheros sucios
  (incluye los tests). `git log master..ocabra/elastic-kv` está **vacío**: la
  rama no tiene ni un commit.
- `/data/backends/llama_cpp/ocabra-src` — 5 ficheros sucios. **De aquí se
  construye el binario que corre en producción** (`LLAMA_CPP_CUSTOM_SERVER_BIN`).

Verificado que los 5 ficheros de servidor son idénticos entre ambos árboles (434
líneas de diff; solo difiere la longitud de los hashes abreviados). No hay stash,
no hay `.patch` en el repo de oCabra, el remoto `upstream` está marcado
`no_push` y no hay fork configurado al que empujar.

**Acción:** commitear en `ocabra/elastic-kv` y empujar a un fork propio
(`llama.cpp-mks`). Es gratis, elimina el riesgo y además es el **prerequisito
técnico** de todo lo demás: en cuanto exista ese fork, `git_repo` + `git_ref` del
spec distribuyen el backend por la vía de fuente que ya funciona, sin maquinaria
nueva.

Efecto secundario deseable: `LLAMA_CPP_CUSTOM_SERVER_BIN` desaparece.
`_resolve_server_bin()` lo consulta como prioridad 1 precisamente porque hoy la
instalación no puede producir el binario custom; con el fork instalado, la
prioridad 2 (`metadata.json` → `extra_bins.server`) basta.

## Trabajo para `llama_cpp_mks`

`LlamaCppMksBackend` debería **heredar de `LlamaCppBackend`** y sobrescribir solo
`install_spec` (con `git_repo` = fork mks, `git_ref` = rama) y el nombre del
tipo. Duplicar el backend entero no tiene sentido: el comportamiento de runtime
es el mismo salvo las capacidades extra del fork.

Los sitios que comparan contra `"llama_cpp"` literal caen en cinco familias.
Revisadas en detalle 8 de ~20 ficheros; **el inventario completo es parte del
trabajo**, no está cerrado aquí.

**1. Conjuntos y registros — mecánico, añadir el tipo nuevo**
- `core/model_ref.py:3` `KNOWN_BACKEND_TYPES`
- `core/duration_estimator.py:71` (p50/p95 por backend) y `:160` (set de backends)
- `core/profile_registry.py:715` (backend → categoría) y `:752` (tupla con `bitnet`)
- `core/runtime_preset_capacity.py:27` `_PROPORTIONAL_MULTI_GPU_BACKENDS`

**2. Predicados de comportamiento — la familia arriesgada.** Deben pasar a un
predicado de familia (`backend_type in LLAMA_CPP_FAMILY`), porque el fork
necesita **el mismo** comportamiento:
- `core/worker_pool.py:541` admisión adaptativa por prefill
- `core/runtime_preset_capacity.py:126`
- `api/openai/_deps.py:1410`, `:1437`, `:1642`, `:1688` (presupuesto KV de
  fairness y resolución de opciones)

**3. Espacio de nombres de `extra_config`.** `core/model_manager_helpers.py:171`
lee `extra.get("llama_cpp")`. **Decisión pendiente:** ¿los modelos del fork
siguen usando la clave `llama_cpp` (menos churn, specs compartidos) o una propia?

**4. Asignación del scanner.** `registry/local_scanner.py:288` y `:473` asignan
`backend_type="llama_cpp"` a los GGUF descubiertos. **Decisión pendiente:** ¿a
qué backend van por defecto los GGUF nuevos?

**5. Allowlists del frontend.** `frontend/src/lib/modelContext.ts` y las
categorías de perfil. Es exactamente la clase de bug que ya ha morder dos veces:
ver las memorias `project_generation_tab_crash` (selectores TTS siempre vacíos
por allowlist desactualizada) y `project_llama_cpp_acl_gap` (5 modelos migrados
sin entrada en `group_models`).

### Migración de modelos entre los dos backends

El backend **forma parte del id canónico**: `model_id = "<backend_type>/<modelo>"`
(`core/model_ref.py`). Así que mover un modelo del upstream al fork es un
**rename**, no un flag: `llama_cpp/gemma4-26b` → `llama_cpp_mks/gemma4-26b`.
Afecta a `model_configs.model_id`, `model_profiles.base_model_id`, los alias de
perfil, las entradas de `group_models` y el historial de `request_stats`.

**La palanca limpia es la capa de perfiles.** Los clientes piden un perfil
(`gemma4:26b`), nunca el id canónico, así que cambiar el `base_model_id` de un
perfil de un backend al otro es **invisible para el cliente** siempre que se
actualicen alias y `group_models`. Eso da además el A/B que motiva la decisión:
dos perfiles apuntando a los dos backends, con un router delante.

## Lo que falta de distribución general

1. **Flujo de actualización.** Endpoint que, para un backend instalado desde
   fuente, reinstale con un pin o `git_ref` nuevo, mostrando antes la versión
   instalada frente a la declarada. Para `llama_cpp_mks` esto es además el
   mecanismo de despliegue del fork: cambiar `git_ref` y actualizar.
2. **Decidir el futuro de OCI.** O se completa la Fase 3 **con CI que construya
   y publique las imágenes**, o se elimina `oci_image`/`oci_tags` de los specs y
   se asume que la distribución es desde fuente. Lo que no debe quedarse es el
   estado actual: metadatos que prometen algo que devuelve 501.
3. **Superficie de versiones.** Exponer por backend: versión declarada, versión
   instalada y, si se puede, la disponible upstream. El incidente de vLLM habría
   sido visible de inmediato.

## Decisiones pendientes del usuario

1. ¿Clave de `extra_config` compartida (`llama_cpp`) o propia para el fork?
2. ¿A qué backend se asignan por defecto los GGUF que descubre el scanner?
3. ¿Se completa OCI con CI, o se elimina y la distribución es desde fuente?
4. ¿La actualización de un backend es manual (cambiar el pin y reinstalar) o hay
   comprobación de novedades upstream?
