import { ShieldCheck } from "lucide-react"
import type { ZeroDataSummary } from "@/types"

/**
 * A «0 data» node keeps nothing in its database: only the last few requests'
 * metrics, in memory, without who asked. Says so, with what they add up to.
 */
export function ZeroDataNotice({ summary }: { summary: ZeroDataSummary }) {
  const parts = [
    `${summary.requests} de ${summary.kept}`,
    summary.errors ? `${summary.errors} con error` : null,
    summary.avgDurationMs != null ? `${(summary.avgDurationMs / 1000).toFixed(1)} s de media` : null,
    `${summary.inputTokens.toLocaleString()} tokens de entrada · ${summary.outputTokens.toLocaleString()} de salida`,
    summary.avgPredictedTokensPerSecond != null ? `${summary.avgPredictedTokensPerSecond} t/s generando` : null,
  ].filter(Boolean)
  return (
    <div
      className="flex items-start gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm"
      data-testid="zero-data-notice"
    >
      <ShieldCheck size={16} className="mt-0.5 shrink-0 text-emerald-600 dark:text-emerald-300" />
      <div>
        <p className="font-medium">Modo 0 data: solo las últimas {summary.kept} peticiones</p>
        <p className="text-xs text-muted-foreground">
          Se guardan en memoria, sin quién las hizo ni el texto de los errores, y se pierden al reiniciar. Nada
          llega a la base de datos, así que el resto de estadísticas está vacío.
        </p>
        <p className="mt-1 text-xs">{parts.join(" · ")}</p>
      </div>
    </div>
  )
}
