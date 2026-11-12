import { useCallback, useEffect, useState } from "react"
import { api } from "@/api/client"
import type { RuntimePreset, RuntimePresetResolution } from "@/types"

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

  const reload = useCallback(async () => {
    if (!modelId) return
    try {
      const profiles = await api.profiles.listByModel(modelId)
      const candidates = profiles.filter(
        (p) => p.profileKind !== "router" && p.category === "llm" && p.runtimePresets.some((r) => r.enabled),
      )
      const profile = candidates.find((p) => p.isDefault) ?? candidates[0]
      if (!profile) {
        setRungs([])
        setProfileId(null)
        return
      }
      let resolved: RuntimePresetResolution[] = []
      try {
        resolved = await api.profiles.resolveRuntimePresets(profile.profileId)
      } catch {
        /* sin resolución, se enseñan los presets tal cual */
      }
      const enabled = profile.runtimePresets.filter((r) => r.enabled)
      setProfileId(profile.profileId)
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

  return { rungs, profileId, reload, active: rungs.length > 0 }
}
