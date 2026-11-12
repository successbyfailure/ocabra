import { useCallback, useEffect, useState } from "react"
import { api } from "@/api/client"
import type { ModelProfile, RuntimePreset, RuntimePresetResolution } from "@/types"

export interface LadderRung {
  preset: RuntimePreset
  resolution: RuntimePresetResolution | null
}

/**
 * Presets del perfil por defecto del modelo, ya resueltos contra el hardware.
 * Cuando hay presets activos, el contexto, los slots, la precisión de la KV y
 * las GPUs de cada carga los fija el preset, no la configuración base.
 */
export function usePresetLadder(modelId: string | null | undefined) {
  const [rungs, setRungs] = useState<LadderRung[]>([])
  const [profileId, setProfileId] = useState<string | null>(null)
  const [profile, setProfile] = useState<ModelProfile | null>(null)
  const [resolutions, setResolutions] = useState<Record<string, RuntimePresetResolution>>({})

  const reload = useCallback(async () => {
    if (!modelId) return
    try {
      const profiles = await api.profiles.listByModel(modelId)
      // El perfil LLM por defecto (con o sin presets): es el que lleva la
      // escalera de Capacidad.
      const llm = profiles.filter((p) => p.profileKind !== "router" && p.category === "llm")
      const profile = llm.find((p) => p.isDefault && p.runtimePresets.length) ??
        llm.find((p) => p.runtimePresets.length) ??
        llm.find((p) => p.isDefault) ??
        llm[0]
      if (!profile) {
        setRungs([])
        setProfileId(null)
        setProfile(null)
        setResolutions({})
        return
      }
      let resolved: RuntimePresetResolution[] = []
      try {
        if (profile.runtimePresets.length) resolved = await api.profiles.resolveRuntimePresets(profile.profileId)
      } catch {
        /* sin resolución, se enseñan los presets tal cual */
      }
      const enabled = profile.runtimePresets.filter((r) => r.enabled)
      setProfileId(profile.profileId)
      setProfile(profile)
      setResolutions(Object.fromEntries(resolved.map((r) => [r.presetKey, r])))
      setRungs(
        enabled.map((preset) => ({
          preset,
          resolution: resolved.find((r) => r.presetKey === preset.presetKey) ?? null,
        })),
      )
    } catch {
      setRungs([])
    }
  }, [modelId])

  useEffect(() => {
    void reload()
  }, [reload])

  return { rungs, profileId, profile, resolutions, reload, active: rungs.length > 0 }
}
