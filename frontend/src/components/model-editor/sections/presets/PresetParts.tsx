/** Piezas de los presets que usa Capacidad: la política de selección y los
 * alias de un peldaño. (La escalera vieja de PresetsSection se retiró.) */
import { useEffect, useState } from "react"
import { Copy, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { modelEditorApi } from "@/api/modelEditor"
import { fmtContext } from "@/components/dashboard/format"
import { cn } from "@/lib/cn"
import type { ModelProfile, RuntimePolicy, RuntimePreset } from "@/types"
import { Panel, Segmented, Toggle, inputCls } from "../../ui"
import { ALIAS_RE, DEFAULT_POLICY } from "./presetLogic"

const errMsg = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback)

export function AliasList({
  profile,
  preset,
  readOnly,
  onChanged,
  inset = true,
}: {
  profile: ModelProfile
  preset: RuntimePreset
  readOnly: boolean
  onChanged: () => Promise<void>
  /** Margen izquierdo para alinear con la escalera de PresetsSection. */
  inset?: boolean
}) {
  const aliases = profile.aliases.filter((a) => a.presetKey === preset.presetKey)
  const [draft, setDraft] = useState("")
  const [hidden, setHidden] = useState(true)
  const [busy, setBusy] = useState(false)
  const valid = ALIAS_RE.test(draft)
  const add = async () => {
    setBusy(true)
    try {
      await modelEditorApi.createAlias(profile.profileId, { alias_id: draft, preset_key: preset.presetKey, hidden })
      setDraft("")
      toast.success(`Alias ${draft} creado`)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo crear el alias"))
    } finally {
      setBusy(false)
    }
  }
  const remove = async (aliasId: string) => {
    setBusy(true)
    try {
      await modelEditorApi.deleteAlias(profile.profileId, aliasId)
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo borrar el alias"))
    } finally {
      setBusy(false)
    }
  }
  if (!aliases.length && readOnly) return null
  return (
    <div className={cn("mt-2 flex flex-wrap items-center gap-1.5 text-[11px]", inset && "pl-[13.25rem]")}>
      <span className="text-muted-foreground">Alias:</span>
      {aliases.map((a) => (
        <span key={a.aliasId} className="inline-flex items-center gap-1 rounded-md border border-border bg-muted/40 px-1.5 py-0.5 font-mono">
          {a.aliasId}
          {!a.hidden && <span className="font-sans text-[10px] text-sky-300">público</span>}
          <button
            type="button"
            title="Copiar"
            onClick={() => {
              void navigator.clipboard?.writeText(a.aliasId)
              toast.success("Copiado")
            }}
            className="text-muted-foreground hover:text-foreground"
          >
            <Copy size={10} />
          </button>
          {!readOnly && (
            <button type="button" title="Borrar alias" disabled={busy} onClick={() => void remove(a.aliasId)} className="text-red-300 hover:text-red-200">
              <X size={10} />
            </button>
          )}
        </span>
      ))}
      {!aliases.length && <span className="text-muted-foreground/70">ninguno</span>}
      {!readOnly && (
        <span className="inline-flex items-center gap-1">
          <input
            className={cn(inputCls, "h-6 w-40 py-0 font-mono text-[11px]", draft && !valid && "border-red-500")}
            placeholder={`${profile.profileId}:${preset.presetKey}`}
            value={draft}
            onChange={(e) => setDraft(e.target.value.trim())}
          />
          <label className="inline-flex items-center gap-1 text-muted-foreground" title="Oculto: resuelve pero no sale en /v1/models">
            <input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} /> oculto
          </label>
          <button
            type="button"
            disabled={busy || !valid}
            onClick={() => void add()}
            className="rounded border border-border px-1.5 py-0.5 hover:bg-muted disabled:opacity-40"
          >
            Añadir
          </button>
        </span>
      )}
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  disabled,
  placeholder,
  unit,
}: {
  label: string
  value: number | null
  onChange: (v: number | null) => void
  disabled?: boolean
  placeholder?: string
  unit?: string
}) {
  return (
    <label className="grid gap-1 text-xs">
      <span className="font-medium">{label}</span>
      <span className="flex items-center gap-1.5">
        <input
          type="number"
          min={0}
          aria-label={label}
          className={cn(inputCls, "tabular-nums")}
          disabled={disabled}
          placeholder={placeholder}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value === "" ? null : Math.max(0, Number(e.target.value)))}
        />
        {unit && <span className="text-muted-foreground">{unit}</span>}
      </span>
    </label>
  )
}

export function PolicyPanel({ profile, readOnly, onChanged }: { profile: ModelProfile; readOnly: boolean; onChanged: () => Promise<void> }) {
  const stored = profile.runtimePolicy ?? DEFAULT_POLICY
  const [policy, setPolicy] = useState<RuntimePolicy>(stored)
  const [saving, setSaving] = useState(false)
  useEffect(() => setPolicy(profile.runtimePolicy ?? DEFAULT_POLICY), [profile.runtimePolicy])
  const set = (patch: Partial<RuntimePolicy>) => setPolicy((p) => ({ ...p, ...patch }))
  const dirty = JSON.stringify(policy) !== JSON.stringify(stored)

  const save = async () => {
    setSaving(true)
    try {
      // Se manda la política completa: los campos calibrados (validados por la
      // sonda, reservas) viajan tal cual y no se pierden al guardar.
      await api.profiles.update(profile.profileId, { runtimePolicy: policy })
      toast.success("Política guardada")
      await onChanged()
    } catch (e) {
      toast.error(errMsg(e, "No se pudo guardar la política"))
    } finally {
      setSaving(false)
    }
  }

  const grace = policy.downgradeGraceSeconds
  const residency = policy.minimumResidencySeconds
  const total = Math.max(grace, residency, 1) * 1.15
  const validated = Object.entries(policy.validatedContextTokens ?? {})
  const vram = policy.validatedVramMb ?? {}

  return (
    <Panel
      title="Cómo se elige y se cambia de preset"
      subtitle="Se guarda con su propio botón, aparte del resto del editor."
      actions={
        !readOnly && (
          <button
            type="button"
            disabled={saving || !dirty}
            onClick={() => void save()}
            className="rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground disabled:opacity-40"
          >
            {saving ? "Guardando…" : "Guardar política"}
          </button>
        )
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="grid gap-3">
          <div className="grid gap-1 text-xs">
            <span className="font-medium">Al llegar una petición</span>
            <Segmented
              size="sm"
              disabled={readOnly}
              value={policy.selection}
              options={[
                { value: "prefer_loaded", label: "Usar el ya cargado si le basta" },
                { value: "prefer_default", label: "Preferir el por defecto" },
              ]}
              onChange={(v) => set({ selection: v })}
            />
          </div>
          <Toggle
            disabled={readOnly}
            checked={policy.restoreDefaultWhenIdle}
            onChange={(v) => set({ restoreDefaultWhenIdle: v })}
            label={<span className="text-xs">Volver al preset por defecto cuando quede ocioso</span>}
          />
          <Toggle
            disabled={readOnly}
            checked={policy.smallRequestsExtendLargeTtl}
            onChange={(v) => set({ smallRequestsExtendLargeTtl: v })}
            label={<span className="text-xs">Las peticiones pequeñas alargan la vida del preset grande</span>}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <NumberField label="Gracia antes de bajar" unit="s" disabled={readOnly} value={grace} onChange={(v) => set({ downgradeGraceSeconds: v ?? 0 })} />
            <NumberField label="Residencia mínima" unit="s" disabled={readOnly} value={residency} onChange={(v) => set({ minimumResidencySeconds: v ?? 0 })} />
            <NumberField label="Contexto máximo validado" unit="tokens" disabled={readOnly} placeholder="sin tope" value={policy.maxContextTokens} onChange={(v) => set({ maxContextTokens: v || null })} />
            <NumberField label="Máximo de salida (tokens)" disabled={readOnly} placeholder="sin tope" value={policy.maxOutputTokens} onChange={(v) => set({ maxOutputTokens: v || null })} />
            <NumberField
              label="Reserva de salida por petición"
              unit="tokens"
              disabled={readOnly}
              placeholder="la global"
              value={policy.defaultOutputReservationTokens}
              onChange={(v) => set({ defaultOutputReservationTokens: v || null })}
            />
          </div>
        </div>

        <div className="space-y-3">
          {/* Línea de tiempo: desde que el preset grande deja de hacer falta. */}
          <div className="rounded-lg border border-border bg-background/40 p-3 text-[11px]" aria-hidden>
            <div className="mb-2 font-medium text-foreground">Tras subir a un preset mayor</div>
            <div className="relative h-7 rounded bg-muted/40">
              <div className="absolute inset-y-0 left-0 rounded-l bg-violet-500/30" style={{ width: `${(residency / total) * 100}%` }} />
              <div
                className="absolute inset-y-1 rounded bg-amber-500/40"
                style={{ left: 0, width: `${(grace / total) * 100}%` }}
              />
              <span className="absolute top-0.5 border-l-2 border-sky-400" style={{ left: `${(grace / total) * 100}%`, height: "1.5rem" }} />
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-muted-foreground">
              <span><span className="inline-block h-2 w-2 rounded-sm bg-violet-500/50" /> residencia mínima {residency} s</span>
              <span><span className="inline-block h-2 w-2 rounded-sm bg-amber-500/60" /> gracia {grace} s</span>
              <span>
                <span className="inline-block h-2 w-0.5 bg-sky-400" />{" "}
                {policy.restoreDefaultWhenIdle ? "vuelve al por defecto si sigue ocioso" : "se queda en el grande"}
              </span>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-background/40 p-3 text-[11px]">
            <div className="mb-1 font-medium text-foreground">Valores medidos (sonda de contexto)</div>
            {validated.length === 0 ? (
              <span className="text-muted-foreground">Ninguno todavía. Usa «Sondear contexto» en un peldaño.</span>
            ) : (
              <ul className="space-y-0.5">
                {validated.map(([key, tokens]) => (
                  <li key={key} className="flex justify-between gap-2 font-mono">
                    <span>{key}</span>
                    <span>
                      {fmtContext(tokens)}
                      {vram[key] ? ` · ${(vram[key] / 1024).toFixed(1)} GB` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </Panel>
  )
}

