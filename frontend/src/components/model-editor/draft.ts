import type { EvictionSchedule } from "@/types"
import type { DeploymentDraft, DraftChange, Json, JsonObject, RawModel, SchemaField } from "./types"

/** Sección del extra_config en la que lee cada backend (llamacpp_mks comparte
 *  la de llama.cpp). Debe coincidir con config_section_for del backend. */
export function sectionFor(backend: string): string {
  return backend === "llama_cpp" || backend === "llamacpp_mks" ? "llama_cpp" : backend
}

export function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function clone<T>(value: T): T {
  return value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T)
}

export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b || a === null || b === null) return false
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
    return a.every((v, i) => deepEqual(v, b[i]))
  }
  if (typeof a === "object") {
    const ka = Object.keys(a as object)
    const kb = Object.keys(b as object)
    if (ka.length !== kb.length) return false
    return ka.every((k) => deepEqual((a as JsonObject)[k], (b as JsonObject)[k]))
  }
  return false
}

export function getPath(obj: JsonObject | null | undefined, path: string[]): Json | undefined {
  let node: Json | undefined = obj ?? undefined
  for (const part of path) {
    if (!isObject(node) || !(part in node)) return undefined
    node = node[part]
  }
  return node
}

/** Copia de ``obj`` con ``value`` en ``path``. ``undefined`` quita la clave
 *  (y las secciones que se queden vacías). */
export function setPath(obj: JsonObject, path: string[], value: Json | undefined): JsonObject {
  const [head, ...rest] = path
  const next: JsonObject = { ...obj }
  if (rest.length === 0) {
    if (value === undefined) delete next[head]
    else next[head] = value
    return next
  }
  const child = isObject(next[head]) ? (next[head] as JsonObject) : {}
  const updated = setPath(child, rest, value)
  if (Object.keys(updated).length === 0 && value === undefined) delete next[head]
  else next[head] = updated
  return next
}

/** JSON Merge Patch (RFC 7386) que lleva de ``base`` a ``draft``: las claves
 *  que desaparecen van como null; los objetos se comparan por dentro, de modo
 *  que tocar un campo de ``speculative`` no reescribe sus hermanos. */
export function mergePatch(base: JsonObject, draft: JsonObject): JsonObject | null {
  const out: JsonObject = {}
  for (const key of Object.keys(base)) {
    if (!(key in draft)) out[key] = null
  }
  for (const [key, value] of Object.entries(draft)) {
    const before = base[key]
    if (isObject(before) && isObject(value)) {
      const sub = mergePatch(before, value)
      if (sub) out[key] = sub
    } else if (!deepEqual(before, value)) {
      out[key] = value
    }
  }
  return Object.keys(out).length ? out : null
}

/** Claves internas que el sistema reescribe y el editor no debe mandar. */
const INTERNAL = new Set(["_runtime", "registry_metadata"])

export function draftFromModel(model: RawModel): DeploymentDraft {
  const extra = clone(model.extra_config ?? {}) as JsonObject
  return {
    loadPolicy: model.load_policy,
    pinPermanent: Boolean(model.pin_permanent),
    preferredGpu: model.preferred_gpu ?? null,
    extraConfig: extra,
    schedules: clone(model.schedules ?? []) as EvictionSchedule[],
  }
}

export interface DraftPatch {
  loadPolicy?: DeploymentDraft["loadPolicy"]
  pinPermanent?: boolean
  preferredGpu?: number | null
  extraConfig?: JsonObject
  schedules?: EvictionSchedule[]
}

export function buildPatch(base: DeploymentDraft, draft: DeploymentDraft): DraftPatch {
  const patch: DraftPatch = {}
  if (base.loadPolicy !== draft.loadPolicy) patch.loadPolicy = draft.loadPolicy
  if (base.pinPermanent !== draft.pinPermanent) patch.pinPermanent = draft.pinPermanent
  if (base.preferredGpu !== draft.preferredGpu) patch.preferredGpu = draft.preferredGpu
  const strip = (o: JsonObject) => Object.fromEntries(Object.entries(o).filter(([k]) => !INTERNAL.has(k)))
  const extra = mergePatch(strip(base.extraConfig), strip(draft.extraConfig))
  if (extra) patch.extraConfig = extra
  if (!deepEqual(base.schedules, draft.schedules)) patch.schedules = draft.schedules
  return patch
}

const TOP_LEVEL_LABELS: Record<string, string> = {
  loadPolicy: "Política de carga",
  pinPermanent: "Pin duro",
  preferredGpu: "GPU preferida",
  schedules: "Horarios",
}

/** Lista legible de cambios, con la etiqueta y el impacto de cada campo. */
export function describeChanges(
  base: DeploymentDraft,
  draft: DeploymentDraft,
  fields: SchemaField[],
): DraftChange[] {
  const changes: DraftChange[] = []
  for (const key of ["loadPolicy", "pinPermanent", "preferredGpu", "schedules"] as const) {
    if (!deepEqual(base[key], draft[key])) {
      changes.push({
        key,
        label: TOP_LEVEL_LABELS[key],
        from: base[key] as Json,
        to: draft[key] as Json,
        // La GPU preferida solo se nota en la siguiente carga.
        reload: key === "preferredGpu",
        memory: key === "preferredGpu",
      })
    }
  }
  const known = new Set<string>()
  for (const field of fields) {
    known.add(field.key)
    const before = getPath(base.extraConfig, field.path)
    const after = getPath(draft.extraConfig, field.path)
    if (!deepEqual(before, after)) {
      changes.push({
        key: field.key,
        label: field.label,
        from: before,
        to: after,
        reload: field.reload,
        memory: field.memory,
      })
    }
  }
  // Cambios en claves que el esquema no describe (JSON en crudo).
  const flat = (o: JsonObject, prefix = ""): Record<string, Json> =>
    Object.entries(o).reduce<Record<string, Json>>((acc, [k, v]) => {
      const key = prefix ? `${prefix}.${k}` : k
      if (INTERNAL.has(k) && !prefix) return acc
      if (isObject(v) && !known.has(key)) Object.assign(acc, flat(v, key))
      else acc[key] = v
      return acc
    }, {})
  const a = flat(base.extraConfig)
  const b = flat(draft.extraConfig)
  for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (known.has(key) || [...known].some((k) => key.startsWith(`${k}.`))) continue
    if (!deepEqual(a[key], b[key])) {
      changes.push({ key, label: key, from: a[key], to: b[key], reload: true, memory: false })
    }
  }
  return changes
}

export function formatValue(value: Json | undefined): string {
  if (value === undefined || value === null) return "—"
  if (typeof value === "boolean") return value ? "sí" : "no"
  if (Array.isArray(value)) return value.length ? value.map((v) => formatValue(v)).join(", ") : "[]"
  if (isObject(value)) return JSON.stringify(value)
  return String(value)
}

/** ¿Se cumplen las dependencias de un campo (``requires``) en el borrador? */
export function requirementsMet(field: SchemaField, extra: JsonObject): boolean {
  if (!field.requires) return true
  return Object.entries(field.requires).every(([key, expected]) => {
    const value = getPath(extra, key.split("."))
    return deepEqual(value, expected)
  })
}
