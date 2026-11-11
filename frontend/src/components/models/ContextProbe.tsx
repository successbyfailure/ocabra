import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import type { ContextProbeJob, RuntimePreset } from "@/types";

const FINISHED = new Set(["done", "error"]);

const STATUS_LABEL: Record<ContextProbeJob["status"], string> = {
  pending: "En cola",
  preparing: "Preparando GPUs",
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
          fullValidation,
          measureVision,
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
      toast.success(`Contexto de ${preset.label} actualizado`);
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
                  <th className="text-left">Contexto</th>
                  <th className="text-left">Memoria</th>
                  <th className="text-left">Cabe</th>
                </tr>
              </thead>
              <tbody>
                {job.steps.map((step, i) => (
                  <tr key={i}>
                    <td>{fmtTokens(step.context)}</td>
                    <td>{Object.keys(step.usedMb).length ? fmtMb(step.usedMb) : step.note}</td>
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

      {result && (
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
