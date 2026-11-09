import { Hourglass } from "lucide-react"
import type { ActiveRequestProgress, ModelActivity, ModelState } from "@/types"
import { BackendIcon } from "./BackendIcon"
import { requestHue } from "./ContextBar"
import { fmtDuration, fmtTokens } from "./format"
import { requestColorOrder } from "./ModelLiveCard"
import { RequestKindBadge } from "./RequestKindBadge"

/**
 * Peticiones esperando slot, por modelo y en el orden en que se atenderán.
 *
 * El número (#) es el turno real del gate de admisión (menos servido primero,
 * no orden de llegada). Las filas van en el orden en que se prevé que entren:
 * el backend simula la liberación de slots y de KV, así que una petición
 * pequeña puede adelantar a una grande que todavía no cabe; la fila lo dice.
 */
export function QueueSection({
  activity,
  models,
}: {
  activity: Record<string, ModelActivity>
  models: Record<string, ModelState>
}) {
  const groups = Object.entries(activity)
    .map(([modelId, act]) => ({
      modelId,
      act,
      // Orden de arranque previsto (KV incluido); las que no tienen previsión
      // van al final en su orden de turno.
      queued: (act.requests ?? [])
        .filter((r) => r.stage === "queued")
        .sort(
          (a, b) =>
            (a.predictedStartOrder ?? 1e9) - (b.predictedStartOrder ?? 1e9) ||
            (a.queuePosition ?? 0) - (b.queuePosition ?? 0),
        ),
    }))
    .filter((g) => g.queued.length > 0)

  if (groups.length === 0) {
    return (
      <p className="rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
        Sin peticiones en cola: todo lo que llega entra directamente.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      {groups.map(({ modelId, act, queued }) => {
        const model = models[modelId]
        const order = requestColorOrder(act.requests)
        return (
          <div key={modelId} className="rounded-2xl border border-border bg-card p-3 shadow-sm">
            <div className="mb-2 flex items-center gap-2 text-sm">
              {model && <BackendIcon backendType={model.backendType} showLabel={false} />}
              <span className="font-semibold">{model?.displayName ?? modelId}</span>
              <span className="text-xs text-muted-foreground">
                {queued.length} en cola · {act.slots ?? "?"} slots
              </span>
            </div>
            <ol className="space-y-1">
              {queued.map((r) => (
                <QueueRow key={r.requestId} req={r} hue={requestHue(order.get(r.requestId) ?? 0)} />
              ))}
            </ol>
          </div>
        )
      })}
    </div>
  )
}

function QueueRow({ req, hue }: { req: ActiveRequestProgress; hue: string }) {
  const turn = req.queuePosition ?? 0
  const start = req.predictedStartOrder
  // El turno es el orden del gate; el arranque previsto tiene en cuenta el KV
  // libre, así que una petición pequeña puede adelantar a una grande.
  const note =
    start == null
      ? req.queueWaitEstimateMs == null && req.kvReservedTokens
        ? { text: "no cabe en el KV libre", cls: "text-red-300" }
        : null
      : start < turn
        ? { text: "se adelanta: cabe en el KV libre", cls: "text-emerald-300" }
        : start > turn
          ? { text: "espera KV: la adelantan otras", cls: "text-amber-300" }
          : null
  return (
    <li className="flex items-center justify-between gap-2 rounded-lg bg-amber-500/5 px-2.5 py-1.5 text-[11.5px]">
      <span className="flex min-w-0 items-center gap-1.5">
        <span className="w-6 shrink-0 text-right font-semibold tabular-nums text-amber-300">
          #{(req.queuePosition ?? 0) + 1}
        </span>
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: hue }} />
        <span className={`truncate ${req.username ? "font-medium" : "italic text-muted-foreground"}`}>
          {req.username ?? "anónimo"}
        </span>
        <RequestKindBadge kind={req.requestKind} />
        <span className="truncate text-muted-foreground">
          ~{fmtTokens(req.prompt?.totalTokens)} tok entrada
          {req.kvReservedTokens ? ` · reserva ${fmtTokens(req.kvReservedTokens)}` : ""}
        </span>
        {note && <span className={`shrink-0 text-[10.5px] ${note.cls}`}>{note.text}</span>}
      </span>
      <span className="flex shrink-0 items-center gap-3 tabular-nums">
        <span className="text-muted-foreground" title="Lleva esperando">
          <Hourglass size={11} className="mr-0.5 inline" />
          {fmtDuration(req.queuedMs ?? req.elapsedMs)}
        </span>
        <span className="text-amber-300" title="Espera estimada hasta conseguir slot">
          {req.queueWaitEstimateMs == null
            ? "entra en ?"
            : req.queueWaitEstimateMs < 1000
              ? "entra ya"
              : `entra en ~${fmtDuration(req.queueWaitEstimateMs)}`}
        </span>
        <span className="text-muted-foreground" title="Duración estimada una vez dentro">
          dura {req.expectedDurationMs != null ? `~${fmtDuration(req.expectedDurationMs)}` : "?"}
        </span>
      </span>
    </li>
  )
}
