import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import type { ContextProbeJob, RuntimePreset } from "@/types";

const FINISHED = new Set(["done", "error"]);

const STATUS_LABEL: Record<ContextProbeJob["status"], string> = {
  pending: "En cola",
  preparing: "Preparando GPUs",
  planning: "Calculando el reparto",
  loading: "Cargando el modelo",
  probing: "Ampliando el contexto",
  validating: "Probando con el pool lleno",
  vision: "Midiendo la visión",
  restoring: "Restaurando",
  done: "Terminado",
  error: "Error",
};

function fmtTokens(tokens: number): string {
  return tokens >= 1024 ? `${Math.round(tokens / 1024)}k` : String(tokens);
}

function fmtMb(map: Record<string, number>): string {
  return Object.entries(map)
    .map(([gpu, mb]) => `GPU ${gpu}: ${mb.toLocaleString()} MB`)
    .join(" · ");
}

interface Props {
  profileId: string;
  preset: RuntimePreset;
  onApplied: () => void;
}

/**
 * Measure the largest context a preset fits on its GPUs: the model is loaded
 * with a small context and grown in place, reading real VRAM per GPU, until
 * the safety margin (vram_buffer_mb). See ocabra/core/context_probe.py.
 */
export function ContextProbe({ profileId, preset, onApplied }: Props) {
  const [open, setOpen] = useState(false);
  const [fullValidation, setFullValidation] = useState(false);
  const [measureVision, setMeasureVision] = useState(true);
  // «Contexto objetivo, lo más rápido»: el contexto que quieres, y busca las
  // menos capas en RAM con las que cabe (para modelos repartidos GPU+RAM).
  const [mode, setMode] = useState<"max" | "target">(
    preset.placement?.nCpuMoe != null || preset.placement?.gpuLayers != null ? "target" : "max",
  );
  const [targetK, setTargetK] = useState<number>(
    Math.max(8, Math.round((preset.contextTokens ?? 32768) / 1024)),
  );
  const [job, setJob] = useState<ContextProbeJob | null>(null);
  const [applying, setApplying] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  };

  const poll = () => {
    stopPolling();
    timer.current = setInterval(async () => {
      try {
        const next = await api.profiles.getContextProbe(profileId, preset.presetKey);
        setJob(next);
        if (FINISHED.has(next.status)) stopPolling();
      } catch {
        stopPolling();
      }
    }, 1500);
  };

  useEffect(() => {
    if (!open) return;
    // Show the last probe of this preset, and keep following it if it runs.
    api.profiles
      .getContextProbe(profileId, preset.presetKey)
      .then((last) => {
        setJob(last);
        if (!FINISHED.has(last.status)) poll();
      })
      .catch(() => setJob(null));
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, profileId, preset.presetKey]);

  const start = async () => {
    try {
      setJob(
        await api.profiles.startContextProbe(profileId, preset.presetKey, {
          fullValidation: mode === "max" && fullValidation,
          measureVision: mode === "max" && measureVision,
          targetContext: mode === "target" ? targetK * 1024 : null,
        }),
      );
      poll();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo iniciar la medición");
    }
  };

  const apply = async () => {
    setApplying(true);
    try {
      await api.profiles.applyContextProbe(profileId, preset.presetKey);
      toast.success(
        job?.result?.offload
          ? `${preset.label}: ${fmtTokens(job.result.offload.targetContext)} con ${job.result.offload.inRam} capas en RAM`
          : `Contexto de ${preset.label} actualizado`,
      );
      onApplied();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "No se pudo aplicar");
    } finally {
      setApplying(false);
    }
  };

  const running = job !== null && !FINISHED.has(job.status);
  const result = job?.status === "done" ? job.result : null;
  const gpus = preset.preferredGpus?.length ? preset.preferredGpus.join(", ") : "las del preset";

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded border border-border px-2 py-1 text-[11px] hover:bg-muted"
        title="Carga el modelo con poco contexto y lo amplía en caliente hasta el margen de seguridad"
      >
        Medir contexto máximo
      </button>
    );
  }

  return (
    <div className="mt-2 w-full space-y-2 rounded-md border border-violet-500/30 bg-violet-500/5 p-3 text-[11px]">
      <div className="flex items-center justify-between">
        <span className="font-medium">Contexto máximo de {preset.label}</span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-muted-foreground hover:text-foreground"
        >
          Cerrar
        </button>
      </div>

      {!running && (
        <div className="space-y-2">
          <p className="text-amber-300">
            Descarga lo que haya en la GPU {gpus} y pone este modelo en mantenimiento mientras
            mide (sus peticiones reciben 503). Los modelos fijados vuelven al terminar.
          </p>
          <div className="flex flex-wrap gap-3" data-testid="probe-mode">
            <label className="flex items-center gap-1.5">
              <input type="radio" checked={mode === "max"} onChange={() => setMode("max")} />
              Contexto máximo con este reparto
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" checked={mode === "target"} onChange={() => setMode("target")} />
              Contexto objetivo, lo más rápido (GPU+RAM)
            </label>
          </div>
          {mode === "target" && (
            <div className="space-y-1">
              <label className="flex items-center gap-2">
                Contexto
                <input
                  type="number"
                  min={4}
                  max={1024}
                  value={targetK}
                  onChange={(e) => setTargetK(Math.max(4, Math.min(1024, Number(e.target.value) || 32)))}
                  className="w-20 rounded border border-border bg-background px-1.5 py-0.5"
                  data-testid="probe-target"
                />
                k {job && !job.contextIsPool ? "por petición" : ""}
              </label>
              <p className="text-muted-foreground">
                Busca las menos capas en RAM (expertos MoE, o capas enteras) con las que cabe ese
                contexto: cada capa que se queda en RAM hace el modelo más lento. Parte de la
                proyección de llama.cpp, carga de verdad cada candidato y comprueba antes que la
                RAM libre alcanza.
              </p>
            </div>
          )}
          {mode === "max" && (<>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={measureVision}
              onChange={(e) => setMeasureVision(e.target.checked)}
            />
            Medir también lo que ocupa el proyector de visión (si el modelo tiene)
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={fullValidation}
              onChange={(e) => setFullValidation(e.target.checked)}
            />
            Probar con el pool lleno (ve el pico real en uso; con contextos grandes, minutos)
          </label>
          </>)}
          <button
            type="button"
            onClick={() => void start()}
            className="rounded border border-violet-500/50 px-2 py-1 hover:bg-violet-500/10"
          >
            {job ? "Medir de nuevo" : "Empezar"}
          </button>
        </div>
      )}

      {job && (
        <div className="space-y-1">
          <div>
            Estado: <span className="font-medium">{STATUS_LABEL[job.status]}</span>
            {job.evicted.length > 0 && (
              <span className="text-muted-foreground"> · descargados: {job.evicted.join(", ")}</span>
            )}
          </div>
          {job.steps.length > 0 && (
            <table className="w-full text-[10px]">
              <thead className="text-muted-foreground">
                <tr>
                  {job.targetContext ? <th className="text-left">En RAM</th> : null}
                  <th className="text-left">Contexto</th>
                  <th className="text-left">Memoria</th>
                  <th className="text-left">Cabe</th>
                </tr>
              </thead>
              <tbody>
                {job.steps.map((step, i) => (
                  <tr key={i}>
                    {job.targetContext ? <td>{step.inRam ?? "—"} capas</td> : null}
                    <td>{fmtTokens(step.context)}</td>
                    <td>
                      {Object.keys(step.usedMb).length ? fmtMb(step.usedMb) : step.note}
                      {step.projectedMb && Object.keys(step.usedMb).length ? (
                        <span className="text-muted-foreground"> (proyección {fmtMb(step.projectedMb)})</span>
                      ) : null}
                    </td>
                    <td className={step.fits ? "text-emerald-300" : "text-red-300"}>
                      {step.fits ? "sí" : "no"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {job.status === "error" && <div className="text-red-300">{job.error}</div>}
        </div>
      )}

      {result?.offload && (
        <div className="space-y-1 rounded border border-emerald-500/30 bg-emerald-500/5 p-2" data-testid="probe-offload-result">
          <div className="text-[12px]">
            {fmtTokens(result.offload.targetContext)} caben con{" "}
            <span className="font-semibold">
              {result.offload.inRam} {result.offload.moe ? "capas de expertos" : "capas"} en RAM
            </span>{" "}
            de {result.offload.layers}
            {result.offload.firstGuess !== result.offload.inRam && (
              <span className="text-muted-foreground"> (la proyección decía {result.offload.firstGuess})</span>
            )}
          </div>
          <div>GPU medida: {fmtMb(result.usedMb)} · límite {fmtMb(result.limitMb)}</div>
          {result.offload.ramNeedMb != null && (
            <div className="text-muted-foreground">
              RAM residente: {(result.offload.ramNeedMb / 1024).toFixed(1)} GB
              {result.offload.ramAvailableMb != null
                ? ` de ${(result.offload.ramAvailableMb / 1024).toFixed(1)} GB libres`
                : ""}
              {result.offload.mlock ? " (con mlock: el fichero entero queda bloqueado)" : ""}
            </div>
          )}
          <button
            type="button"
            disabled={applying}
            onClick={() => void apply()}
            className="mt-1 rounded border border-emerald-500/50 px-2 py-1 hover:bg-emerald-500/10 disabled:opacity-40"
          >
            Aplicar al preset ({fmtTokens(result.offload.targetContext)}, {result.offload.inRam} en RAM)
          </button>
        </div>
      )}

      {result && !result.offload && (
        <div className="space-y-1 rounded border border-emerald-500/30 bg-emerald-500/5 p-2">
          <div className="text-[12px]">
            Máximo: <span className="font-semibold">{result.maxContext.toLocaleString()}</span>{" "}
            tokens{" "}
            {job && !job.contextIsPool && (
              <span className="text-muted-foreground">
                (pool; {Math.floor(result.maxContext / Math.max(1, job.parallelism)).toLocaleString()}{" "}
                por petición)
              </span>
            )}
            {result.cappedBy === "train" && (
              <span className="text-muted-foreground"> · tope del entrenamiento del modelo</span>
            )}
          </div>
          <div>Usado: {fmtMb(result.usedMb)}</div>
          <div className="text-muted-foreground">
            Límite: {fmtMb(result.limitMb)} (margen {result.marginMb} MB)
          </div>
          <div className="text-muted-foreground">
            KV por 1k tokens:{" "}
            {Object.entries(result.kvMbPer1k)
              .map(([gpu, mb]) => `GPU ${gpu} ${mb} MB`)
              .join(" · ")}
          </div>
          {result.fullValidated && result.peakMb && (
            <div>
              Pico con el pool lleno ({result.validationPromptTokens?.toLocaleString() ?? "?"}{" "}
              tokens): {fmtMb(result.peakMb)}
              {result.reducedByValidation
                ? ` · reducido ${result.reducedByValidation.toLocaleString()} tokens`
                : ""}
            </div>
          )}
          {result.vision && (
            <div>
              Visión ({result.vision.projector}): {fmtMb(result.vision.costMb)} · con ella cargada caben{" "}
              {result.vision.maxContextWithVision.toLocaleString()} tokens
            </div>
          )}
          <button
            type="button"
            disabled={applying}
            onClick={() => void apply()}
            className="mt-1 rounded border border-emerald-500/50 px-2 py-1 hover:bg-emerald-500/10 disabled:opacity-40"
          >
            Aplicar al preset ({result.maxContext.toLocaleString()} tokens)
          </button>
        </div>
      )}
    </div>
  );
}
