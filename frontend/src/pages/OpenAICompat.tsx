import { useEffect, useState } from "react"
import { AlertCircle, Loader2, RefreshCw } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { ModelProfile, OpenAITTSVoiceMapEntry } from "@/types"

const OPENAI_TTS_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
const OPENAI_VOICE_NAMES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

interface ProbeResult {
  state: "idle" | "loading" | "done" | "error"
  voices?: string[]
  modelType?: string
  error?: string
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/**
 * Compatibilidad de voces OpenAI — deliberadamente independiente de los
 * routers (ver docs/CONTRACTS.md §8.10). Un cliente que solo sabe mandar
 * los nombres oficiales de OpenAI (modelo tts-1/tts-1-hd/gpt-4o-mini-tts,
 * voz alloy/echo/...) no puede apuntar a ningún nombre propio de oCabra,
 * ni siquiera un router. Esta página fija, para cada combinación, a qué
 * perfil propio responde.
 */
export function OpenAICompat() {
  const [entries, setEntries] = useState<OpenAITTSVoiceMapEntry[]>([])
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState<string | null>(null)
  const [probes, setProbes] = useState<Record<string, ProbeResult>>({})

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const [entryList, profileList] = await Promise.all([
        api.openaiTtsCompat.list(),
        api.profiles.listAll(),
      ])
      setEntries(entryList)
      setProfiles(profileList)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const entryFor = (model: string, voice: string) =>
    entries.find((e) => e.openaiModel === model && e.voiceName === voice)

  const setPin = async (model: string, voice: string, targetProfileId: string) => {
    const key = `${model}:${voice}`
    setSaving(key)
    try {
      if (!targetProfileId) {
        await api.openaiTtsCompat.delete(model, voice)
        toast.success(`${model} + ${voice} desasignado`)
      } else {
        await api.openaiTtsCompat.upsert(model, voice, targetProfileId)
        toast.success(`${model} + ${voice} → ${targetProfileId}`)
      }
      await refresh()
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setSaving(null)
    }
  }

  const probe = async (model: string, voice: string) => {
    const key = `${model}:${voice}`
    setProbes((prev) => ({ ...prev, [key]: { state: "loading" } }))
    try {
      const res = await fetch(
        `/v1/audio/voices?model=${encodeURIComponent(model)}&voice=${encodeURIComponent(voice)}`,
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as Record<string, unknown>
      setProbes((prev) => ({
        ...prev,
        [key]: {
          state: "done",
          voices: Array.isArray(data.voices) ? data.voices.map(String) : [],
          modelType: String(data.model_type ?? ""),
        },
      }))
    } catch (err) {
      setProbes((prev) => ({ ...prev, [key]: { state: "error", error: errMsg(err) } }))
    }
  }

  const enabledProfiles = profiles.filter((p) => p.enabled).sort((a, b) => a.profileId.localeCompare(b.profileId))

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Compatibilidad OpenAI (TTS)</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Un cliente que solo sabe los nombres oficiales de OpenAI (modelo{" "}
            <code>tts-1</code>/<code>tts-1-hd</code>/<code>gpt-4o-mini-tts</code>, voz{" "}
            <code>alloy</code>/<code>echo</code>...) no puede apuntar a ningún nombre propio de
            oCabra. Fija aquí a qué perfil responde cada combinación. Sin asignar, ese nombre se
            resuelve literal (404, como hoy). Independiente de los routers — no toca{" "}
            <code>routing_targets</code>.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={loading}
          className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Actualizar
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle size={16} className="mt-0.5" />
          {error}
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/30 text-left text-xs text-muted-foreground">
              <th className="px-3 py-2 font-medium">Modelo OpenAI</th>
              <th className="px-3 py-2 font-medium">Voz</th>
              <th className="px-3 py-2 font-medium">Perfil propio</th>
              <th className="px-3 py-2 font-medium">Verificar</th>
            </tr>
          </thead>
          <tbody>
            {OPENAI_TTS_MODELS.map((model) =>
              OPENAI_VOICE_NAMES.map((voice) => {
                const key = `${model}:${voice}`
                const entry = entryFor(model, voice)
                const result = probes[key]
                return (
                  <tr key={key} className="border-b border-border/50 last:border-0">
                    <td className="px-3 py-1.5 font-mono text-xs">{model}</td>
                    <td className="px-3 py-1.5 font-mono text-xs">{voice}</td>
                    <td className="px-3 py-1.5">
                      <select
                        value={entry?.targetProfileId ?? ""}
                        disabled={saving === key || loading}
                        onChange={(e) => void setPin(model, voice, e.target.value)}
                        className="w-full max-w-xs rounded-md border border-input bg-background px-2 py-1 text-xs disabled:opacity-50"
                      >
                        <option value="">Sin asignar (404)</option>
                        {enabledProfiles.map((p) => (
                          <option key={p.profileId} value={p.profileId}>
                            {p.profileId}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-3 py-1.5">
                      <div className="flex items-center gap-1.5 text-xs">
                        <button
                          type="button"
                          onClick={() => void probe(model, voice)}
                          disabled={result?.state === "loading"}
                          className="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 hover:bg-muted disabled:opacity-50"
                        >
                          {result?.state === "loading" && <Loader2 size={10} className="animate-spin" />}
                          Verificar
                        </button>
                        {result?.state === "error" && (
                          <span className="text-destructive">Error: {result.error}</span>
                        )}
                        {result?.state === "done" && result.modelType === "placeholder" && (
                          <span className="text-amber-600 dark:text-amber-400">no cargado, reintenta</span>
                        )}
                        {result?.state === "done" && result.modelType !== "placeholder" && (
                          <span className="font-mono text-foreground">→ {result.voices?.[0] ?? "?"}</span>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
