import type {
  RuntimePolicy,
  RuntimePreset,
  RuntimePresetCreate,
  RuntimePresetResolution,
} from "@/types"

/** Política por defecto del backend (``RuntimePolicy`` en schemas/profiles.py). */
export const DEFAULT_POLICY: RuntimePolicy = {
  selection: "prefer_loaded",
  restoreDefaultWhenIdle: true,
  downgradeGraceSeconds: 600,
  minimumResidencySeconds: 120,
  smallRequestsExtendLargeTtl: false,
  remoteWarmThresholdSeconds: 10,
  maxContextTokens: null,
  maxOutputTokens: null,
  validatedContextTokens: {},
  validatedVramMb: {},
  contextProbe: {},
  defaultOutputReservationTokens: null,
}

export const EMPTY_PRESET: RuntimePresetCreate = {
  presetKey: "64k",
  label: "Normal (64k)",
  contextMode: "fixed",
  contextTokens: 65536,
  parallelism: 2,
  kvCacheDtype: "q8",
  placementPolicy: "auto",
  preferredGpus: null,
  executionPolicy: "auto",
  enabled: true,
  isDefault: false,
  priority: 100,
}

/** Escalera inicial 32k / 64k (por defecto, p2, q8) / 128k / MAX. */
export function suggestedLadder(): RuntimePresetCreate[] {
  return [
    { ...EMPTY_PRESET, isDefault: true, priority: 20 },
    { ...EMPTY_PRESET, presetKey: "32k", label: "Ligero (32k)", contextTokens: 32768, priority: 10 },
    { ...EMPTY_PRESET, presetKey: "128k", label: "Alto (128k)", contextTokens: 131072, priority: 30 },
    {
      ...EMPTY_PRESET,
      presetKey: "max",
      label: "Máximo efectivo",
      contextMode: "max",
      contextTokens: null,
      priority: 40,
    },
  ]
}

export function presetToForm(preset: RuntimePreset): RuntimePresetCreate {
  return {
    presetKey: preset.presetKey,
    label: preset.label,
    contextMode: preset.contextMode,
    contextTokens: preset.contextTokens,
    parallelism: preset.parallelism,
    kvCacheDtype: preset.kvCacheDtype,
    placementPolicy: preset.placementPolicy,
    preferredGpus: preset.preferredGpus,
    executionPolicy: preset.executionPolicy,
    enabled: preset.enabled,
    isDefault: preset.isDefault,
    priority: preset.priority,
    placement: preset.placement ?? null,
  }
}

/** Contexto con el que se ordena un peldaño: el fijo, o el efectivo si es MAX. */
export function rungContext(preset: RuntimePreset, resolution?: RuntimePresetResolution): number {
  if (preset.contextMode === "fixed") return preset.contextTokens ?? 0
  return resolution?.effectiveContext ?? Number.MAX_SAFE_INTEGER
}

export function sortRungs(
  presets: RuntimePreset[],
  resolutions: Record<string, RuntimePresetResolution>,
): RuntimePreset[] {
  return [...presets].sort((a, b) => {
    const diff = rungContext(a, resolutions[a.presetKey]) - rungContext(b, resolutions[b.presetKey])
    return diff !== 0 ? diff : a.priority - b.priority
  })
}

/** Qué límite recorta el contexto pedido (el más bajo de los tres). */
export function limitingReason(r: RuntimePresetResolution | undefined): string | null {
  if (!r || r.effectiveContext == null) return null
  const limits: [number | null, string][] = [
    [r.modelMax, "límite del modelo"],
    [r.hardwareMax, "límite del hardware"],
    [r.policyMax, "límite de la política"],
  ]
  const active = limits.filter(([v]) => v != null && v > 0) as [number, string][]
  if (!active.length) return null
  const [min, label] = active.reduce((acc, cur) => (cur[0] < acc[0] ? cur : acc))
  if (r.requestedContext != null && r.requestedContext <= min) return null
  return `${label}`
}

/** Escala logarítmica del eje de contexto: 0..1 entre 4k y ``max``. */
export function axisPosition(tokens: number, max: number): number {
  const lo = Math.log2(4096)
  const hi = Math.log2(Math.max(max, 8192))
  const v = Math.log2(Math.max(tokens, 4096))
  return Math.min(1, Math.max(0, (v - lo) / (hi - lo)))
}

export const AXIS_MARKS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

/** Bytes por elemento de cada precisión de KV, para la ayuda. */
export const KV_HINT: Record<RuntimePreset["kvCacheDtype"], string> = {
  fp16: "sin cuantizar · 2 bytes",
  bf16: "sin cuantizar · 2 bytes",
  fp8: "mitad de memoria · 1 byte",
  q8: "mitad de memoria · 1 byte",
  q4: "un cuarto · 0,5 bytes",
}

export const PRESET_KEY_RE = /^[a-z0-9][a-z0-9-]*$/
export const ALIAS_RE = /^[a-z0-9][a-z0-9.\-:]*$/
