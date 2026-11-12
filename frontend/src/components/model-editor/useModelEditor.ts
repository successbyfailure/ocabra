import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { modelEditorApi } from "@/api/modelEditor"
import { buildPatch, describeChanges, draftFromModel, getPath, setPath } from "./draft"
import type { ConfigSchema, DeploymentDraft, Json, PreviewResult, RawModel, SchemaField } from "./types"

/**
 * Estado del editor de un despliegue: el modelo guardado, su esquema de
 * opciones, el borrador y la vista previa. Nada se escribe hasta ``save``.
 */
export function useModelEditor(modelId: string | null) {
  const [model, setModel] = useState<RawModel | null>(null)
  const [schema, setSchema] = useState<ConfigSchema | null>(null)
  const [base, setBase] = useState<DeploymentDraft | null>(null)
  const [draft, setDraft] = useState<DeploymentDraft | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const reload = useCallback(async () => {
    if (!modelId) return
    try {
      const [m, s] = await Promise.all([modelEditorApi.get(modelId), modelEditorApi.schema(modelId)])
      setModel(m)
      setSchema(s)
      const d = draftFromModel(m)
      setBase(d)
      setDraft(d)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [modelId])

  useEffect(() => {
    setModel(null)
    setSchema(null)
    setBase(null)
    setDraft(null)
    void reload()
  }, [reload])

  /** Refresca el estado vivo (estado, VRAM…) sin tocar el borrador. */
  const refreshLive = useCallback(async () => {
    if (!modelId) return
    try {
      setModel(await modelEditorApi.get(modelId))
    } catch {
      /* se conserva lo último */
    }
  }, [modelId])

  const fields = schema?.fields ?? []
  const fieldByKey = useMemo(() => new Map(fields.map((f) => [f.key, f])), [fields])

  const setField = useCallback((path: string[], value: Json | undefined) => {
    setDraft((d) => (d ? { ...d, extraConfig: setPath(d.extraConfig, path, value) } : d))
  }, [])

  const update = useCallback((patch: Partial<DeploymentDraft>) => {
    setDraft((d) => (d ? { ...d, ...patch } : d))
  }, [])

  /** Valor efectivo: el del borrador o, si no está puesto, el por defecto. */
  const valueOf = useCallback(
    (key: string): Json | undefined => {
      const field = fieldByKey.get(key)
      const path = field?.path ?? key.split(".")
      const value = draft ? getPath(draft.extraConfig, path) : undefined
      return value !== undefined ? value : (field?.default ?? undefined)
    },
    [draft, fieldByKey],
  )

  const isSet = useCallback(
    (key: string) => {
      const field = fieldByKey.get(key)
      return draft ? getPath(draft.extraConfig, field?.path ?? key.split(".")) !== undefined : false
    },
    [draft, fieldByKey],
  )

  const changes = useMemo(
    () => (base && draft ? describeChanges(base, draft, fields) : []),
    [base, draft, fields],
  )

  const discard = useCallback(() => setDraft(base), [base])
  const { preview, loading: previewLoading } = usePreview(modelId, draft, fields)

  const save = useCallback(async () => {
    if (!modelId || !base || !draft) return null
    const patch = buildPatch(base, draft)
    if (Object.keys(patch).length === 0) return model
    setSaving(true)
    try {
      await modelEditorApi.patch(modelId, patch)
      const fresh = await modelEditorApi.get(modelId)
      setModel(fresh)
      const d = draftFromModel(fresh)
      setBase(d)
      setDraft(d)
      return fresh
    } finally {
      setSaving(false)
    }
  }, [modelId, base, draft, model])

  return {
    model,
    schema,
    fields,
    fieldByKey,
    base,
    draft,
    error,
    saving,
    changes,
    dirty: changes.length > 0,
    preview,
    previewLoading,
    previewErrors: preview?.errors ?? [],
    setField,
    update,
    valueOf,
    isSet,
    discard,
    save,
    reload,
    refreshLive,
  }
}

export type ModelEditorState = ReturnType<typeof useModelEditor>

/** Vista previa del borrador, recalculada con un pequeño retardo. */
export function usePreview(modelId: string | null, draft: DeploymentDraft | null, fields: SchemaField[]) {
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [loading, setLoading] = useState(false)
  const seq = useRef(0)
  // Solo lo que afecta a la memoria o al comando dispara una nueva consulta.
  const key = useMemo(() => {
    if (!draft) return ""
    return JSON.stringify({ gpu: draft.preferredGpu, extra: draft.extraConfig })
  }, [draft])

  useEffect(() => {
    if (!modelId || !draft) return
    const id = ++seq.current
    setLoading(true)
    const timer = window.setTimeout(async () => {
      try {
        const result = await modelEditorApi.preview(modelId, {
          preferredGpu: draft.preferredGpu,
          extraConfig: draft.extraConfig,
        })
        if (id === seq.current) setPreview(result)
      } catch {
        /* la vista previa es orientativa */
      } finally {
        if (id === seq.current) setLoading(false)
      }
    }, 350)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, key, fields.length])

  return { preview, loading }
}
