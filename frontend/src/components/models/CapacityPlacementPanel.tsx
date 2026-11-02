import { CapacityPanel, Chip } from "@/components/models/CapacityPanel";
import { RuntimePresetsPanel } from "@/components/models/RuntimePresetsPanel";
import { FieldHint, SubBlock, UseCaseManagedNotice } from "@/components/models/fieldPrimitives";
import { formatTokenCount } from "@/lib/modelContext";
import type { BackendExtraConfig, GPUState, ModelMemoryEstimate, ModelProfile, ModelState } from "@/types";

const MEMORY_ESTIMATE_SOURCE_LABELS: Record<ModelMemoryEstimate["source"], string> = {
  gguf_header: "header GGUF",
  runtime_probe: "probe de runtime",
  heuristic: "heurística",
};
const mbToGb = (mb: number) => (mb / 1024).toFixed(1);

const LLAMA_KV_CACHE_TYPES = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"] as const;

interface LlamaCppPlacementProps {
  disabledGpus: Set<number>;
  onDisabledGpusChange: (next: Set<number>) => void;
  splitStrategy: "evenly" | "favor_main" | "single" | "custom" | "";
  onSplitStrategyChange: (v: "evenly" | "favor_main" | "single" | "custom") => void;
  mainGpu: number | "";
  onMainGpuChange: (v: number | "") => void;
  splitMode: "layer" | "row" | "none" | "";
  onSplitModeChange: (v: "layer" | "row" | "none" | "") => void;
  tensorSplit: Record<number, string>;
  onTensorSplitChange: (next: Record<number, string>) => void;
  cacheTypeK: string;
  onCacheTypeKChange: (v: string) => void;
  cacheTypeV: string;
  onCacheTypeVChange: (v: string) => void;
  flashAttn: boolean;
  parallelSlots: string;
  onParallelSlotsChange: (v: string) => void;
  kvUnified: "" | "true" | "false";
  onKvUnifiedChange: (v: "" | "true" | "false") => void;
  kvUnifiedPerSlot: string;
  onKvUnifiedPerSlotChange: (v: string) => void;
  contBatching: boolean;
  onContBatchingChange: (v: boolean) => void;
  keepAliveSeconds: string;
  onKeepAliveSecondsChange: (v: string) => void;
}

interface CapacityPlacementPanelProps {
  model: ModelState;
  gpus: GPUState[];
  extraConfig?: BackendExtraConfig;
  hasEnabledPresets: boolean;
  executionProfiles: ModelProfile[];
  useCaseActive: boolean;
  onUseCaseChange: (active: boolean) => void;

  memoryEstimate: ModelMemoryEstimate | null;
  memoryEstimateError: string | null;
  estimatingMemory: boolean;
  probingMemory: boolean;
  onRecalculate: () => void;
  onProbe: () => void;

  // Present only for backendType === "llama_cpp".
  llamaCpp?: LlamaCppPlacementProps;
}

function GpuPlacementChip({
  gpu,
  enabled,
  onToggle,
  isMain,
  ratio,
  onRatioChange,
}: {
  gpu: GPUState;
  enabled: boolean;
  onToggle: (next: boolean) => void;
  isMain: boolean;
  ratio?: string;
  onRatioChange?: (value: string) => void;
}) {
  const fraction = gpu.totalVramMb > 0 ? Math.min(1, gpu.usedVramMb / gpu.totalVramMb) : 0;
  return (
    <div className={`rounded-md border p-2.5 text-sm ${enabled ? "border-border" : "border-border/40 opacity-60"}`}>
      <div className="flex items-center justify-between gap-2">
        <label className="flex items-center gap-2 font-medium">
          <input type="checkbox" checked={enabled} onChange={(event) => onToggle(event.target.checked)} />
          GPU {gpu.index}
          {isMain && (
            <span className="rounded border border-primary/40 bg-primary/10 px-1 py-0.5 text-[10px] font-normal uppercase text-primary">
              principal
            </span>
          )}
        </label>
        <span className="text-xs text-muted-foreground">{gpu.name}</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-sky-500" style={{ width: `${Math.max(2, fraction * 100)}%` }} />
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {mbToGb(gpu.usedVramMb)} / {mbToGb(gpu.totalVramMb)} GB en uso
        </span>
        {onRatioChange && enabled && (
          <input
            type="number"
            min="0"
            step="0.5"
            value={ratio ?? ""}
            onChange={(event) => onRatioChange(event.target.value)}
            placeholder="ratio"
            className="w-16 rounded border border-border bg-background px-1.5 py-0.5 text-right"
          />
        )}
      </div>
    </div>
  );
}

function ProbeEstimateBlock({
  memoryEstimate,
  memoryEstimateError,
  estimatingMemory,
  probingMemory,
  onRecalculate,
  onProbe,
  showProbeButton,
}: {
  memoryEstimate: ModelMemoryEstimate | null;
  memoryEstimateError: string | null;
  estimatingMemory: boolean;
  probingMemory: boolean;
  onRecalculate: () => void;
  onProbe: () => void;
  showProbeButton: boolean;
}) {
  return (
    <div className="space-y-3 text-sm">
      {(estimatingMemory || probingMemory) && (
        <div className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
          {probingMemory ? "Ejecutando validación real del engine..." : "Recalculando estimación..."}
        </div>
      )}
      {memoryEstimate &&
        (() => {
          const usedMb =
            memoryEstimate.modelLoadingMemoryMb ?? memoryEstimate.estimatedEngineMbPerGpu ?? memoryEstimate.estimatedWeightsMb;
          const capMb = memoryEstimate.totalVramMb;
          const fraction = usedMb != null && capMb ? Math.min(1, usedMb / capMb) : 0;
          const barColor =
            memoryEstimate.fitsCurrentGpu === false ? "bg-red-500" : fraction > 0.85 ? "bg-amber-500" : "bg-emerald-500";
          return (
            <>
              {usedMb != null && capMb != null && (
                <div
                  className={`rounded-md border px-3 py-2 text-xs ${
                    memoryEstimate.fitsCurrentGpu === false ? "border-red-500/30 bg-red-500/5" : "border-emerald-500/30 bg-emerald-500/5"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium">
                      {memoryEstimate.fitsCurrentGpu === false ? "No cabe" : "Cabe"} en GPU #{memoryEstimate.gpuIndex ?? "-"}
                    </span>
                    <span className="text-muted-foreground">
                      {mbToGb(usedMb)} / {mbToGb(capMb)} GB
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div className={`h-full transition-all ${barColor}`} style={{ width: `${Math.max(4, fraction * 100)}%` }} />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {memoryEstimate.estimatedWeightsMb != null && (
                  <Chip label="Pesos" value={`${mbToGb(memoryEstimate.estimatedWeightsMb)} GB`} />
                )}
                {memoryEstimate.estimatedEngineMbPerGpu != null && (
                  <Chip label="Engine/GPU" value={`${mbToGb(memoryEstimate.estimatedEngineMbPerGpu)} GB`} />
                )}
                {memoryEstimate.estimatedKvCacheMb != null && (
                  <Chip label="KV cache" value={`${memoryEstimate.estimatedKvCacheMb} MB`} />
                )}
                {memoryEstimate.requestedContextLength != null && (
                  <Chip label="Contexto pedido" value={formatTokenCount(memoryEstimate.requestedContextLength)} />
                )}
                {memoryEstimate.estimatedMaxContextLength != null && (
                  <Chip label="Contexto máximo" value={formatTokenCount(memoryEstimate.estimatedMaxContextLength)} />
                )}
                {memoryEstimate.maximumConcurrency != null && (
                  <Chip label="Concurrencia máx." value={`${memoryEstimate.maximumConcurrency.toFixed(2)}x`} />
                )}
                {memoryEstimate.budgetVramMb != null && (
                  <Chip label="Budget VRAM" value={`${mbToGb(memoryEstimate.budgetVramMb)} GB`} />
                )}
                {memoryEstimate.enginePresent != null && (
                  <Chip label="Engine presente" value={memoryEstimate.enginePresent ? "sí" : "no"} />
                )}
              </div>

              <div className="text-[11px] text-muted-foreground">
                {`fuente: ${MEMORY_ESTIMATE_SOURCE_LABELS[memoryEstimate.source]} · estado: ${memoryEstimate.status}`}
              </div>

              {memoryEstimate.warning && <p className="text-xs text-amber-300">{memoryEstimate.warning}</p>}
              {memoryEstimate.notes.length > 0 && (
                <div className="space-y-1 text-xs text-muted-foreground">
                  {memoryEstimate.notes.map((note, index) => (
                    <p key={`memory-note-${index}`}>{note}</p>
                  ))}
                </div>
              )}
            </>
          );
        })()}
      {memoryEstimateError && <p className="text-xs text-amber-300">{memoryEstimateError}</p>}
      <div>
        <button
          type="button"
          onClick={onRecalculate}
          className="rounded-md border border-border px-3 py-2 text-xs text-foreground hover:bg-muted"
        >
          Recalcular estimación
        </button>
        {showProbeButton && (
          <button
            type="button"
            onClick={onProbe}
            className="ml-2 rounded-md border border-sky-500/40 px-3 py-2 text-xs text-sky-100 hover:bg-sky-500/10"
          >
            Validar con engine
          </button>
        )}
      </div>
    </div>
  );
}

/** Unified "capacity, cache and placement" panel — merges what used to be 3
 * separate FieldSections (Memoria y contexto / Presets de contexto /
 * Estimación por probe) plus, for llama_cpp, the Multi-GPU and Concurrencia
 * sections and the KV cache sub-block that lived inside "llama.cpp Runtime".
 * Controlled component: all values/setters are owned by ModelConfigModal,
 * this component only arranges the JSX. */
export function CapacityPlacementPanel({
  model,
  gpus,
  extraConfig,
  hasEnabledPresets,
  executionProfiles,
  useCaseActive,
  onUseCaseChange,
  memoryEstimate,
  memoryEstimateError,
  estimatingMemory,
  probingMemory,
  onRecalculate,
  onProbe,
  llamaCpp,
}: CapacityPlacementPanelProps) {
  return (
    <div className="space-y-4">
      <CapacityPanel
        modelId={model.modelId}
        gpus={gpus}
        extraConfig={extraConfig}
        onUseCaseChange={onUseCaseChange}
        hideUseCase={hasEnabledPresets}
      />

      {llamaCpp && gpus.length > 0 && (
        <SubBlock title="Placement" description="Reparto de capas entre GPUs, GPU principal y exclusión selectiva de dispositivos.">
          <p className="text-xs text-muted-foreground">
            El placement de GPU no depende del caso de uso — se configura siempre aquí, incluso con un caso de uso activo.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {gpus.map((gpu) => (
              <GpuPlacementChip
                key={gpu.index}
                gpu={gpu}
                enabled={!llamaCpp.disabledGpus.has(gpu.index)}
                onToggle={(next) => {
                  const nextSet = new Set(llamaCpp.disabledGpus);
                  if (next) nextSet.delete(gpu.index);
                  else nextSet.add(gpu.index);
                  llamaCpp.onDisabledGpusChange(nextSet);
                }}
                isMain={llamaCpp.mainGpu === gpu.index}
                ratio={llamaCpp.splitStrategy === "custom" ? llamaCpp.tensorSplit[gpu.index] : undefined}
                onRatioChange={
                  llamaCpp.splitStrategy === "custom"
                    ? (value) => llamaCpp.onTensorSplitChange({ ...llamaCpp.tensorSplit, [gpu.index]: value })
                    : undefined
                }
              />
            ))}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <fieldset className="space-y-2">
              <legend className="text-sm text-muted-foreground">Estrategia de split</legend>
              {(
                [
                  ["evenly", "Even split (auto-balanceado por VRAM)"],
                  ["custom", "Custom ratios"],
                  ["single", "Single GPU"],
                ] as const
              ).map(([value, label]) => (
                <label key={value} className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="llama-split-strategy"
                    value={value}
                    checked={llamaCpp.splitStrategy === value}
                    onChange={() => llamaCpp.onSplitStrategyChange(value)}
                  />
                  <span>{label}</span>
                </label>
              ))}
              <FieldHint>
                `Even split` calcula ratios proporcionales al VRAM total de cada GPU al cargar el modelo. `Single GPU` fija
                `split_mode=none`.
              </FieldHint>
            </fieldset>

            <div className="space-y-3">
              <label className="block text-sm">
                <span className="mb-1 block text-muted-foreground">Main GPU</span>
                <select
                  value={llamaCpp.mainGpu === "" ? "" : String(llamaCpp.mainGpu)}
                  onChange={(event) =>
                    llamaCpp.onMainGpuChange(event.target.value === "" ? "" : Number(event.target.value))
                  }
                  className="w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  <option value="">auto</option>
                  {gpus
                    .filter((gpu) => !llamaCpp.disabledGpus.has(gpu.index))
                    .map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        GPU {gpu.index} - {gpu.name}
                      </option>
                    ))}
                </select>
                <FieldHint>GPU que aloja los tensores no repartidos. Mapea a `--main-gpu`.</FieldHint>
              </label>

              <label className="block text-sm">
                <span className="mb-1 block text-muted-foreground">Split mode</span>
                <select
                  value={llamaCpp.splitMode}
                  onChange={(event) => llamaCpp.onSplitModeChange(event.target.value as "layer" | "row" | "none" | "")}
                  className="w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  <option value="">layer (default)</option>
                  <option value="layer">layer</option>
                  <option value="row">row</option>
                  <option value="none">none</option>
                </select>
                <FieldHint>
                  `layer` reparte por capas (default), `row` reparte dentro de cada capa, `none` mantiene todo en una GPU.
                </FieldHint>
              </label>
            </div>
          </div>
        </SubBlock>
      )}

      {llamaCpp && (
        <SubBlock title="Cache (KV)" description="Cuantizar la KV cache reduce VRAM por token de contexto.">
          {useCaseActive && <UseCaseManagedNotice fields="Context size, parallel slots y cache type K/V" />}
          <div className="grid gap-3 md:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">cache_type_k</span>
              <select
                value={llamaCpp.cacheTypeK}
                onChange={(event) => llamaCpp.onCacheTypeKChange(event.target.value)}
                disabled={useCaseActive}
                className="w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
                title="Tipo de cuantización de la cache K (keys)."
              >
                <option value="">auto (f16)</option>
                {LLAMA_KV_CACHE_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">cache_type_v</span>
              <select
                value={llamaCpp.cacheTypeV}
                onChange={(event) => llamaCpp.onCacheTypeVChange(event.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2"
                title="Tipo de cuantización de la cache V (values). Requiere flash attention si != f16."
                disabled={useCaseActive || (!llamaCpp.flashAttn && llamaCpp.cacheTypeV !== "" && llamaCpp.cacheTypeV !== "f16")}
              >
                <option value="">auto (f16)</option>
                {LLAMA_KV_CACHE_TYPES.map((value) => (
                  <option key={value} value={value} disabled={!llamaCpp.flashAttn && value !== "f16"}>
                    {value}
                  </option>
                ))}
              </select>
              {!llamaCpp.flashAttn && llamaCpp.cacheTypeV !== "" && llamaCpp.cacheTypeV !== "f16" && (
                <FieldHint>cache_type_v cuantizado requiere `flash attention` activo. Marca el checkbox de arriba o vuelve a `f16`.</FieldHint>
              )}
            </label>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">kv_unified</span>
              <select
                value={llamaCpp.kvUnified}
                onChange={(event) => llamaCpp.onKvUnifiedChange(event.target.value as "" | "true" | "false")}
                className="w-full rounded-md border border-border bg-background px-3 py-2"
              >
                <option value="">default</option>
                <option value="true">activada</option>
                <option value="false">desactivada</option>
              </select>
              <FieldHint>Un solo pool físico repartido dinámicamente entre los slots.</FieldHint>
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">kv_unified_per_slot</span>
              <input
                type="number"
                min="1"
                value={llamaCpp.kvUnifiedPerSlot}
                onChange={(event) => llamaCpp.onKvUnifiedPerSlotChange(event.target.value)}
                disabled={llamaCpp.kvUnified !== "true"}
                placeholder="sin límite por slot"
                className="w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
              />
              <FieldHint>Vacío permite a una petición usar todo el pool disponible.</FieldHint>
            </label>
          </div>
        </SubBlock>
      )}

      {llamaCpp && (
        <SubBlock title="Slots" description="Peticiones concurrentes, batching continuo y TTL de inactividad.">
          {useCaseActive && <UseCaseManagedNotice fields="parallel_slots" />}
          <div className="grid gap-4 md:grid-cols-3">
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">parallel_slots</span>
              <input
                type="number"
                min="1"
                value={llamaCpp.parallelSlots}
                onChange={(event) => llamaCpp.onParallelSlotsChange(event.target.value)}
                disabled={useCaseActive}
                placeholder="1"
                className="w-full rounded-md border border-border bg-background px-3 py-2 disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-muted-foreground">keep_alive_seconds</span>
              <input
                type="number"
                min="0"
                value={llamaCpp.keepAliveSeconds}
                onChange={(event) => llamaCpp.onKeepAliveSecondsChange(event.target.value)}
                placeholder="global"
                className="w-full rounded-md border border-border bg-background px-3 py-2"
              />
              <FieldHint>Override del idle eviction global. `0` desactiva la evicción.</FieldHint>
            </label>
            <label className="inline-flex items-center gap-2 self-end pb-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={llamaCpp.contBatching}
                onChange={(event) => llamaCpp.onContBatchingChange(event.target.checked)}
              />
              cont batching
            </label>
          </div>
        </SubBlock>
      )}

      {!hasEnabledPresets && (
        <SubBlock
          title="Estimación por probe (runtime)"
          description="Medición real opcional del engine para la configuración actual; complementa la predicción por arquitectura de arriba."
        >
          <ProbeEstimateBlock
            memoryEstimate={memoryEstimate}
            memoryEstimateError={memoryEstimateError}
            estimatingMemory={estimatingMemory}
            probingMemory={probingMemory}
            onRecalculate={onRecalculate}
            onProbe={onProbe}
            showProbeButton={model.backendType === "vllm"}
          />
        </SubBlock>
      )}

      {executionProfiles.length > 0 && (
        <SubBlock title="Presets de contexto" description="Atajos por perfil; expande para editar, crear o medir en frío.">
          {executionProfiles.map((profile) => (
            // A closed <details> does NOT unmount its children — RuntimePresetsPanel
            // keeps loading/editing state across collapse. Don't "optimize" this to
            // a conditional render on `open`, that would reset edits on collapse.
            <details key={profile.profileId} className="rounded-md border border-border/60">
              <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
                {profile.displayName || profile.profileId}
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {profile.runtimePresets
                    .map(
                      (preset) =>
                        `${preset.label || preset.presetKey}${preset.isDefault ? " (default)" : ""}${preset.enabled ? "" : " (off)"}`,
                    )
                    .join(" · ") || "sin presets"}
                </span>
              </summary>
              <div className="border-t border-border/60 p-3">
                <RuntimePresetsPanel profile={profile} onChanged={() => {}} />
              </div>
            </details>
          ))}
        </SubBlock>
      )}
    </div>
  );
}
