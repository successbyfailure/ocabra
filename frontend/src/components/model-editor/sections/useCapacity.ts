import { useEffect, useState } from "react"
import { modelEditorApi } from "@/api/modelEditor"
import type { CapacityInfo } from "../types"

const cache = new Map<string, CapacityInfo>()

/** Datos de arquitectura y capacidad de un modelo (GET /capacity), cacheados
 *  por combinación de parámetros. */
export function useCapacity(
  modelId: string | undefined,
  params: { gpu?: number | null; slots?: number[]; kvDtype?: string; context?: number | null } = {},
) {
  const key = modelId ? `${modelId}|${JSON.stringify(params)}` : ""
  const [data, setData] = useState<CapacityInfo | null>(() => (key ? (cache.get(key) ?? null) : null))
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    if (!modelId) return
    const hit = cache.get(key)
    if (hit) {
      setData(hit)
      return
    }
    let cancelled = false
    setLoading(true)
    const timer = window.setTimeout(() => {
      modelEditorApi
        .capacity(modelId, params)
        .then((res) => {
          cache.set(key, res)
          if (!cancelled) setData(res)
        })
        .catch(() => {
          if (!cancelled) setData(null)
        })
        .finally(() => {
          if (!cancelled) setLoading(false)
        })
    }, 250)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return { data, loading }
}
