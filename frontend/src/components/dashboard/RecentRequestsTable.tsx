import { useCallback, useEffect, useState } from "react"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import type { RecentRequest } from "@/types"
import { BackendIcon } from "./BackendIcon"
import { fmtClock, fmtContext, fmtDuration, fmtRealtime, fmtTokens, fmtTps } from "./format"
import { useGpuStore } from "@/stores/gpuStore"
import { GpuBadge } from "./GpuBadge"
import { RequestKindBadge } from "./RequestKindBadge"

const TH = "px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground"

/**
 * Peticiones terminadas con su rendimiento por fase: espera en cola, t/s de
 * prefill y de generación, TTFT, tokens (entrada · cache · salida), preset y
 * GPU. Las que siguen en curso no salen aquí: viven en la card de su modelo.
 *
 * La IP y el user agent solo llegan rellenos para admins; el backend los manda
 * a null al resto, así que aquí basta con no pintarlos si no hay dato.
 */
export function RecentRequestsTable() {
  const [rows, setRows] = useState<RecentRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [onlyErrors, setOnlyErrors] = useState(false)

  const load = useCallback(async () => {
    try {
      const result = await api.stats.recent(40)
      setRows(result.requests.filter((r) => !r.inFlight))
    } catch {
      /* se conserva lo último */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), 15_000)
    return () => window.clearInterval(id)
  }, [load])

  const isError = (r: RecentRequest) => Boolean(r.error) || (r.statusCode != null && r.statusCode >= 400)
  const visible = onlyErrors ? rows.filter(isError) : rows

  if (loading) {
    return <p className="rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">Cargando…</p>
  }
  if (rows.length === 0) {
    return <EmptyState title="Sin peticiones recientes" description="Aparecerán aquí cuando terminen." />
  }

  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <input type="checkbox" checked={onlyErrors} onChange={(e) => setOnlyErrors(e.target.checked)} />
        Solo errores ({rows.filter(isError).length})
      </label>
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-muted/30">
              <tr>
                <th className={`${TH} text-left`}>Hora</th>
                <th className={`${TH} text-left`}>Modelo</th>
                <th className={`${TH} text-left`}>Tipo</th>
                <th className={`${TH} text-left`}>Usuario / API key</th>
                <th className={`${TH} text-right`} title="Espera en la cola de admisión">Cola</th>
                <th className={`${TH} text-right`} title="Tiempo hasta el primer token">TTFT</th>
                <th className={`${TH} text-right`}>Prefill</th>
                <th className={`${TH} text-right`}>Generación</th>
                <th className={`${TH} text-right`}>Duración</th>
                <th className={`${TH} text-right`} title="Entrada · de cache · salida">Tokens</th>
                <th className={`${TH} text-left`}>Estado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {visible.map((r) => (
                <Row key={r.id} r={r} error={isError(r)} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function Row({ r, error }: { r: RecentRequest; error: boolean }) {
  const gpus = useGpuStore((s) => s.gpus)
  const hora = new Date(r.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  const queueMs = r.queueMs ?? r.runtimeTelemetry?.queue_ms ?? null
  const ttft = r.runtimeTelemetry?.ttft_ms ?? null
  const meta = [r.runtimePreset ? `preset ${r.runtimePreset}` : null, r.effectiveContext ? `ctx ${fmtContext(r.effectiveContext)}` : null]
    .filter(Boolean)
    .join(" · ")
  const status = r.statusCode != null ? String(r.statusCode) : r.error ? "error" : "—"
  return (
    <>
      <tr className={`transition-colors hover:bg-muted/30 ${error ? "bg-red-500/5" : ""}`}>
        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{hora}</td>
        <td className="max-w-[14rem] px-3 py-2">
          <div className="flex items-center gap-1.5">
            {r.backendType && <BackendIcon backendType={r.backendType} showLabel={false} />}
            <span className="truncate font-medium" title={r.modelId}>{r.modelId}</span>
          </div>
          {(meta || r.gpuIndex != null) && (
            <div className="flex items-center gap-1.5 truncate text-[10.5px] text-muted-foreground">
              {r.gpuIndex != null && <GpuBadge index={r.gpuIndex} gpus={gpus} />}
              {meta}
            </div>
          )}
        </td>
        <td className="px-3 py-2">
          <RequestKindBadge kind={r.requestKind ?? r.backendType} />
        </td>
        <td className="px-3 py-2">
          <div className={r.username ? "text-muted-foreground" : "italic text-muted-foreground/60"}>{r.username ?? "anónimo"}</div>
          {r.apiKeyName && (
            <div className="max-w-[12rem] truncate font-mono text-[11px] text-muted-foreground/70" title={`API key: ${r.apiKeyName}`}>
              🔑 {r.apiKeyName}
            </div>
          )}
          {(r.clientAddr || r.userAgent) && (
            <div className="max-w-[14rem] truncate font-mono text-[10.5px] text-muted-foreground/60" title={r.userAgent ?? undefined}>
              {r.clientAddr ?? "?"}
              {r.userAgent ? ` · ${r.userAgent}` : ""}
            </div>
          )}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{queueMs ? fmtDuration(queueMs) : "—"}</td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{ttft != null ? fmtDuration(ttft) : "—"}</td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmtTps(r.promptTokensPerSecond)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
          {r.audioSeconds && r.durationMs ? (
            <span title={`${fmtClock(r.audioSeconds)} de audio en ${fmtDuration(r.durationMs)}`}>
              {fmtRealtime(r.audioSeconds / (r.durationMs / 1000))} t. real
            </span>
          ) : (
            fmtTps(r.predictedTokensPerSecond)
          )}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmtDuration(r.durationMs)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
          {r.inputTokens != null || r.outputTokens != null ? (
            <>
              {fmtTokens(r.inputTokens ?? 0)}
              {r.cachedTokens ? <span className="text-muted-foreground/60"> ·{fmtTokens(r.cachedTokens)}c</span> : null} ·{" "}
              {fmtTokens(r.outputTokens ?? 0)}
            </>
          ) : (
            "—"
          )}
        </td>
        <td className="px-3 py-2">
          <span
            className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${
              error
                ? "border-red-500/40 bg-red-500/15 text-red-200"
                : r.statusCode != null
                  ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-200"
                  : "border-border bg-muted/50 text-muted-foreground"
            }`}
          >
            {status}
          </span>
        </td>
      </tr>
      {r.error && (
        <tr className="bg-red-500/5">
          <td colSpan={11} className="px-3 pb-2 pt-0 font-mono text-[11px] text-red-300/90">
            {r.error}
          </td>
        </tr>
      )}
    </>
  )
}
