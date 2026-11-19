import type {
  RuntimePolicy,
  RuntimePreset,
  RuntimePresetCreate,
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
  enabled: true,
  isDefault: false,
  priority: 100,
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
    enabled: preset.enabled,
    isDefault: preset.isDefault,
    priority: preset.priority,
    placement: preset.placement ?? null,
  }
}

export const PRESET_KEY_RE = /^[a-z0-9][a-z0-9-]*$/
export const ALIAS_RE = /^[a-z0-9][a-z0-9.\-:]*$/
