import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ModelEditor } from "@/pages/ModelEditor"
import { useAuthStore } from "@/stores/authStore"
import { useGpuStore } from "@/stores/gpuStore"
import type { ConfigSchema, RawModel, SchemaField } from "../types"
import { resetCapacityCache } from "../sections/useCapacity"
import { MemoryBars } from "../sections/capacity/Workbench"

const modelEditorApi = vi.hoisted(() => ({
  get: vi.fn(),
  insights: vi.fn(),
  schema: vi.fn(),
  preview: vi.fn(),
  capacity: vi.fn(),
  patch: vi.fn(),
  setMaintenance: vi.fn(),
  ollamaSettings: vi.fn(),
  applyOllamaSettings: vi.fn(),
  createAlias: vi.fn(),
  deleteAlias: vi.fn(),
  autoSetup: vi.fn(),
}))
const api = vi.hoisted(() => ({
  logicalModels: { get: vi.fn(), patch: vi.fn(), addDeployment: vi.fn(), removeDeployment: vi.fn() },
  models: {
    get: vi.fn(),
    load: vi.fn(),
    unload: vi.fn(),
    cancelLoad: vi.fn(),
    downsize: vi.fn(),
    speculativeCandidates: vi.fn(),
  },
  profiles: { listByModel: vi.fn(), listAll: vi.fn(), resolveRuntimePresets: vi.fn() },
  groups: { list: vi.fn(), listModels: vi.fn() },
  stats: { recent: vi.fn() },
}))
vi.mock("@/api/modelEditor", () => ({ modelEditorApi }))
vi.mock("@/api/client", () => ({ api }))
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock("@/components/dashboard/useLiveActivity", () => ({
  useLiveActivity: () => ({ data: { activity: {}, vramByGpu: { "llamacpp_mks/m": { "1": 20000 } }, stuckThresholdSeconds: 600 }, live: true, receivedAt: 0 }),
}))
vi.mock("@/components/models/CompileModal", () => ({ CompileModal: () => null }))

const f = (key: string, group: string, extra: Partial<SchemaField> = {}): SchemaField => ({
  key,
  path: key.split("."),
  type: "int",
  label: key,
  group,
  help: "",
  default: null,
  nullable: true,
  reload: true,
  memory: true,
  advanced: false,
  ...extra,
})

const schema: ConfigSchema = {
  backend: "llamacpp_mks",
  section: "llama_cpp",
  groups: [],
  vramOverride: true,
  isLlm: true,
  unknownKeys: [],
  fields: [
    f("keep_alive_seconds", "load", { label: "Descargar tras inactividad", default: 300, reload: false, memory: false }),
    f("llama_cpp.ctx_size", "context", { label: "Contexto", default: 4096, widget: "context" }),
    f("llama_cpp.parallel_slots", "context", { label: "Slots", default: 1 }),
    f("llama_cpp.flash_attn", "acceleration", { type: "bool", label: "Flash attention", default: false }),
    f("llama_cpp.split_mode", "gpus", {
      type: "enum",
      label: "Modo de reparto",
      options: [
        { value: "layer", label: "Por capas" },
        { value: "none", label: "Una sola GPU" },
      ],
    }),
  ],
}

const model = (over: Partial<RawModel> = {}): RawModel =>
  ({
    model_id: "llamacpp_mks/m",
    backend_model_id: "m",
    backend_type: "llamacpp_mks",
    display_name: "Modelo M",
    status: "configured",
    load_policy: "on_demand",
    pin_permanent: false,
    preferred_gpu: null,
    current_gpu: [],
    vram_used_mb: 0,
    disk_size_bytes: null,
    capabilities: { reasoning: false } as RawModel["capabilities"],
    extra_config: { llama_cpp: { ctx_size: 32768, flash_attn: true, model_path: "/data/m.gguf" } },
    schedules: [],
    ...over,
  }) as RawModel

function renderEditor(section = "summary") {
  return render(
    <MemoryRouter initialEntries={[`/models/edit/${encodeURIComponent("llamacpp_mks/m")}?s=${section}`]}>
      <Routes>
        <Route path="/models/edit/*" element={<ModelEditor />} />
        <Route path="/models" element={<div>lista</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

function setRole(role: "user" | "model_manager" | "system_admin") {
  useAuthStore.setState({
    user: { id: "u", username: "u", email: null, role, createdAt: "2026-01-01T00:00:00Z" },
    isLoading: false,
  } as never)
}

describe("ModelEditor", () => {
  beforeEach(() => {
    resetCapacityCache()
    setRole("model_manager")
    useGpuStore.setState({
      gpus: [
        { index: 0, name: "NVIDIA GeForce RTX 3060", totalVramMb: 12288, freeVramMb: 10000, usedVramMb: 2288, utilizationPct: 0, temperatureC: 40, powerDrawW: 20, powerLimitW: 170, lockedVramMb: 0, processes: [] },
        { index: 1, name: "NVIDIA GeForce RTX 3090", totalVramMb: 24576, freeVramMb: 4000, usedVramMb: 20576, utilizationPct: 0, temperatureC: 40, powerDrawW: 20, powerLimitW: 350, lockedVramMb: 0, processes: [] },
      ],
      lastUpdated: null,
    })
    modelEditorApi.get.mockResolvedValue(model())
    modelEditorApi.schema.mockResolvedValue(schema)
    modelEditorApi.insights.mockResolvedValue({
      rates: { prefillTps: 2000, prefillSource: "historical", generationTps: 40, typicalOutputTokens: 800 },
      typicalLoadMs: 45000,
      template: { effortLevels: false, thinkingSwitch: true },
      profileIds: ["m"],
      maintenance: false,
    })
    api.stats.recent.mockResolvedValue({
      requests: [
        { id: "r1", modelId: "m", requestedModel: "m", startedAt: "2026-11-12T10:00:00Z", durationMs: 1200, statusCode: 200, inputTokens: 100, outputTokens: 20, predictedTokensPerSecond: 40, inFlight: false },
        { id: "r2", modelId: "otro", startedAt: "2026-11-12T10:00:00Z", durationMs: 1, statusCode: 200, inFlight: false },
      ],
    })
    modelEditorApi.preview.mockResolvedValue({
      errors: [],
      unknownKeys: [],
      estimate: { model_loading_memory_mb: 21000, requested_context_length: 32768, estimated_max_context_length: 65536 },
      gpuIndices: [1],
      gpus: [{ index: 1, mb: 21000, breakdown: { weights: 19000, kv: 1600, overhead: 400 } }],
      launch: { argv: ["--server-bin", "/bin/llama-server", "--ctx-size", "32768"], env: { CUDA_VISIBLE_DEVICES: "1" } },
    })
    modelEditorApi.capacity.mockResolvedValue({ applicable: false })
    modelEditorApi.patch.mockResolvedValue(model())
    api.logicalModels.get.mockResolvedValue({
      modelKey: "m",
      displayName: "Modelo M",
      category: "llm",
      defaultBackend: "llamacpp_mks",
      deployments: [
        { modelId: "llamacpp_mks/m", backend: "llamacpp_mks", isDefault: true, installed: true, status: "configured", loadPolicy: "on_demand", pinPermanent: false, preferredGpu: null, artifact: null, profiles: [] },
      ],
      compatibleBackends: [],
      publicProfiles: [],
      hiddenProfiles: [],
      registryMetadata: null,
    })
    api.models.get.mockRejectedValue(new Error("no camel state in tests"))
    api.models.speculativeCandidates.mockResolvedValue([])
    api.profiles.listByModel.mockResolvedValue([])
  })

  afterEach(() => vi.clearAllMocks())

  it("capacity shows the base memory per GPU with this model's live usage subtracted", async () => {
    renderEditor("capacity")
    const bars = await screen.findByTestId("memory-bars")
    // #1 tiene 20.576 MB usados y 20.000 son de este modelo: le quedan 24.000 MB.
    await waitFor(() => expect((bars.querySelector('[data-gpu-bar="1"]') as HTMLElement).textContent).toContain("21 GB de 23 GB libres"))
    // La leyenda está junto a las barras y nombra cada parte.
    const legend = screen.getByTestId("memory-legend")
    expect(legend.textContent).toContain("pesos del modelo")
    expect(legend.textContent).toContain("KV (contexto)")
    // Sin desglose por procesos, lo no atribuible es "sistema" (rayado) y el
    // resto, otros modelos.
    expect(legend.textContent).toContain("sistema")
    expect(legend.textContent).toContain("otros modelos")
    expect(legend.textContent).not.toContain("no cabe")
    expect(bars.querySelector('[data-gpu-bar="1"] [data-part="system"]')).toBeTruthy()
    expect(bars.querySelector('[data-gpu-bar="1"] [data-part="impossible"]')).toBeNull()
    // Los controles de memoria están en el mismo banco de trabajo.
    const bench = screen.getByTestId("workbench")
    expect(bench.contains(bars)).toBe(true)
    // Dónde corre (GPUs + CPU) va en el mismo panel, encima de los controles.
    expect(screen.getByTestId("device-placement").textContent).toContain("CPU")
    expect(screen.getByTestId("rung-tests").textContent).toContain("Probar tal cual")
  })

  it("memory bars split system memory from other models and hatch what does not fit", async () => {
    useGpuStore.setState({
      gpus: [
        // 2.288 MB usados: 2.000 son de otro modelo de oCabra, el resto es
        // sistema (288) y se suma el margen de 512 que guarda el planificador.
        { index: 0, name: "NVIDIA GeForce RTX 3060", totalVramMb: 12288, freeVramMb: 10000, usedVramMb: 2288, utilizationPct: 0, temperatureC: 40, powerDrawW: 20, powerLimitW: 170, lockedVramMb: 0,
          processes: [{ pid: 7, processName: "whisper", processType: "compute", usedVramMb: 2000 }] },
      ],
      lastUpdated: null,
    })
    modelEditorApi.preview.mockResolvedValue({
      errors: [], unknownKeys: [],
      estimate: { model_loading_memory_mb: 13000 },
      gpuIndices: [0],
      gpus: [{ index: 0, mb: 13000, breakdown: { weights: 12000, kv: 1000 } }],
      launch: { argv: [], env: {} },
    })
    renderEditor("capacity")
    const bars = await screen.findByTestId("memory-bars")
    const bar = bars.querySelector('[data-gpu-bar="0"]') as HTMLElement
    await waitFor(() => expect(bar.querySelector('[data-part="impossible"]')).toBeTruthy())
    expect((bar.querySelector('[data-part="system"]') as HTMLElement).title).toContain("800 MB")
    // Orden: sistema, otros modelos, este modelo; lo que se sale, rayado rojo.
    expect([...bar.querySelectorAll("[data-part]")].map((e) => e.getAttribute("data-part"))).toEqual(["system", "others", "weights", "kv", "impossible"])
    expect((bar.querySelector('[data-part="impossible"]') as HTMLElement).style.backgroundImage).toContain("repeating-linear-gradient")
    expect((bar.querySelector('[data-part="system"]') as HTMLElement).style.backgroundImage).toContain("repeating-linear-gradient")
    expect(screen.getByTestId("memory-legend").textContent).toContain("no cabe")
    expect(bars.textContent).toContain("No cabe en estas GPUs.")
    expect(screen.getByTestId("device-placement").textContent).toContain("otros modelos 2.0 GB · sistema 800 MB")
  })

  it("marking both GPUs together in the base writes split_policy always", async () => {
    renderEditor("capacity")
    const devices = await screen.findByTestId("device-placement")
    await waitFor(() => expect(devices.querySelectorAll("[data-gpu-card]").length).toBeGreaterThan(2))
    fireEvent.click(screen.getByRole("button", { name: "Todas a la vez" }))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalled())
    expect(modelEditorApi.patch.mock.calls[0][1].extraConfig.llama_cpp.split_policy).toBe("always")
  })

  it("keeps a GPU selectable with MoE offload when the node defaults gpu_layers to zero", async () => {
    useGpuStore.setState({
      gpus: [
        { index: 0, name: "NVIDIA GeForce GTX 1660", totalVramMb: 6144, freeVramMb: 677, usedVramMb: 5467, utilizationPct: 0, temperatureC: 40, powerDrawW: 20, powerLimitW: 120, lockedVramMb: 0, processes: [] },
      ],
      lastUpdated: null,
    })
    modelEditorApi.schema.mockResolvedValue({
      ...schema,
      fields: [...schema.fields, f("llama_cpp.gpu_layers", "gpus", { label: "Capas en GPU", default: 0 })],
    })
    modelEditorApi.get.mockResolvedValue(
      model({
        preferred_gpu: 0,
        extra_config: { llama_cpp: { ctx_size: 65536, n_cpu_moe: 8, split_mode: "none" } },
      }),
    )
    modelEditorApi.capacity.mockResolvedValue({
      applicable: true,
      weights_mb: 16283,
      overhead_mb: 400,
      kv_dtype: "q8",
      arch: { layers: 30, n_kv_heads: 8, native_context: 262144, kv_mb_per_1k_tokens: 22.5 },
      capacity: [],
      vram_curve: [],
    })

    renderEditor("capacity")
    const gpu = await screen.findByLabelText("Usar GPU 0")
    await waitFor(() => expect((gpu as HTMLInputElement).checked).toBe(false))
    fireEvent.click(gpu.closest("button") as HTMLButtonElement)
    await waitFor(() => expect((screen.getByLabelText("Usar GPU 0") as HTMLInputElement).checked).toBe(true))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalled())
    expect(modelEditorApi.patch.mock.calls[0][1]).toMatchObject({ extraConfig: { llama_cpp: { gpu_layers: 999 } } })
  })

  it("an LLM without presets offers to create one to measure its real context", async () => {
    api.profiles.listByModel.mockResolvedValue([
      { profileId: "m", profileKind: "execution", baseModelId: "llamacpp_mks/m", displayName: "m", description: null, category: "llm",
        loadOverrides: null, requestDefaults: null, assets: null, enabled: true, isDefault: true, routingTargets: null,
        runtimePolicy: null, runtimePresets: [], aliases: [], createdAt: null, updatedAt: null },
    ])
    renderEditor("capacity")
    expect(await screen.findByTestId("probe-needs-preset")).toBeTruthy()
  })

  it("the unified context control sets the number of simultaneous requests", async () => {
    modelEditorApi.capacity.mockResolvedValue({
      applicable: true, weights_mb: 10000, overhead_mb: 400, kv_dtype: "fp16",
      arch: { layers: 40, n_kv_heads: 8, native_context: 131072, kv_mb_per_1k_tokens: 100 },
      capacity: [], vram_curve: [],
    })
    renderEditor("capacity")
    const control = await screen.findByTestId("context-control")
    // Un solo control: la barra con lo elegido y lo que cabe, y los botones de
    // peticiones a la vez con el contexto que cabría con cada número.
    expect(screen.queryByTestId("slots-chart")).toBeNull()
    expect(control.querySelector('[data-testid="ctx-slider"]')).toBeTruthy()
    expect(control.querySelectorAll("[data-slots]").length).toBeGreaterThanOrEqual(5)
    await waitFor(() => expect(control.querySelector('[data-slots="1"]')!.textContent).toMatch(/≤ \d+k/))
    fireEvent.click(control.querySelector('[data-slots="4"]') as HTMLElement)
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalled())
    expect(modelEditorApi.patch.mock.calls[0][1].extraConfig.llama_cpp.parallel_slots).toBe(4)
  })

  it("the context bar hatches the range that cannot fit and flags a choice inside it", async () => {
    modelEditorApi.capacity.mockResolvedValue({
      applicable: true, weights_mb: 10000, overhead_mb: 400, kv_dtype: "fp16",
      arch: { layers: 40, n_kv_heads: 8, native_context: 131072, kv_mb_per_1k_tokens: 100 },
      capacity: [], vram_curve: [],
    })
    modelEditorApi.preview.mockResolvedValue({
      errors: [], unknownKeys: [],
      estimate: { model_loading_memory_mb: 21000, requested_context_length: 32768 },
      gpuIndices: [1],
      gpus: [{ index: 1, mb: 21000, breakdown: { weights: 19000, kv: 1600, overhead: 400 } }],
      launch: { argv: [], env: {} },
    })
    renderEditor("capacity")
    const control = await screen.findByTestId("context-control")
    // 24.576 MB − 1.024 de sistema (512 sin atribuir + 512 de margen del
    // planificador) − 19.400 fijos: ~4.100 MB de KV a 100 MB/1k, unos 41k.
    const hatch = await waitFor(() => {
      const el = control.querySelector('[data-testid="context-impossible"]') as HTMLElement
      expect(el).toBeTruthy()
      return el
    })
    expect(hatch.style.backgroundImage).toContain("repeating-linear-gradient")
    await waitFor(() => expect(screen.getByTestId("context-verdict").textContent).toMatch(/de 23 GB · cabe · hasta 40\.9k/))
    expect(screen.getByTestId("context-choice").className).toContain("bg-emerald-400")
    // Elegir 64k (dentro del rayado) marca la elección como imposible.
    fireEvent.change(screen.getByTestId("ctx-slider"), { target: { value: "16" } })
    await waitFor(() => expect(screen.getByTestId("ctx-value").textContent).toBe("64k"))
    expect(screen.getByTestId("context-choice").className).toContain("bg-red-400")
  })

  it("with a shared KV pool the context control sizes the whole pool", async () => {
    modelEditorApi.capacity.mockResolvedValue({
      applicable: true, weights_mb: 10000, overhead_mb: 400, kv_dtype: "fp16",
      arch: { layers: 40, n_kv_heads: 8, native_context: 131072, kv_mb_per_1k_tokens: 100 },
      capacity: [], vram_curve: [],
    })
    modelEditorApi.get.mockResolvedValue(model({ extra_config: { llama_cpp: { ctx_size: 65536, parallel_slots: 4, kv_unified: true } } }))
    renderEditor("capacity")
    const control = await screen.findByTestId("context-control")
    expect(control.textContent).toContain("Contexto del pool (compartido)")
    await waitFor(() => expect(screen.getByTestId("context-verdict").textContent).toContain("en total"))
    // Con pool, el número de peticiones no cambia el tamaño que cabe: los
    // botones no llevan un contexto por petición.
    expect(control.querySelector('[data-slots="4"]')!.textContent).toBe("4")
  })

  it("old section links (gpus, context, presets) land on Capacidad", async () => {
    renderEditor("presets")
    expect(await screen.findByTestId("capacity-section")).toBeTruthy()
  })

  it("auto setup plans rungs for the marked devices and applies them after confirming", async () => {
    const plan = {
      rungs: [
        { key: "normal", label: "3090 · 96k", gpus: [1], context: 98304, placement: { gpu_layers: null, n_cpu_moe: null }, vram_by_gpu: { "1": 23000 }, host_mb: 0 },
        { key: "extended", label: "Dual · 256k · RAM", gpus: [0, 1], context: 262144, placement: { gpu_layers: null, n_cpu_moe: 6 }, vram_by_gpu: { "0": 9000, "1": 17000 }, host_mb: 4096 },
      ],
      needsCpu: true,
      warnings: ["«Dual · 256k · RAM» deja 4 GB en RAM"],
      budgets: [],
      applied: [],
    }
    modelEditorApi.autoSetup.mockImplementation(async (_id: string, body: { apply: boolean }) => (body.apply ? { ...plan, applied: ["normal", "extended"] } : plan))
    renderEditor("capacity")
    fireEvent.click(await screen.findByTestId("auto-setup-open"))
    const panel = screen.getByTestId("auto-setup")
    // Una casilla por GPU (todas marcadas) y la CPU, desmarcada.
    expect((within(panel).getByLabelText("Calcular con GPU 0") as HTMLInputElement).checked).toBe(true)
    fireEvent.click(within(panel).getByLabelText("Calcular con GPU 0"))
    fireEvent.click(within(panel).getByLabelText("Calcular con CPU"))
    fireEvent.click(screen.getByTestId("auto-setup-calculate"))
    const result = await screen.findByTestId("auto-setup-result")
    expect(modelEditorApi.autoSetup).toHaveBeenLastCalledWith("llamacpp_mks/m", { gpus: [1], cpu: true, apply: false })
    const ext = result.querySelector('[data-auto-rung="extended"]') as HTMLElement
    expect(ext.textContent).toContain("256k")
    expect(ext.textContent).toContain("RAM: 4.0 GB")
    expect(ext.textContent).toContain("expertos de 6 capas en RAM")
    expect(result.querySelector('[data-auto-rung="normal"]')!.textContent).toContain("todo en GPU")
    expect(result.textContent).toContain("deja 4 GB en RAM")
    expect(result.textContent).toContain("parte del modelo en la RAM")
    // Aplicar pide confirmación, llama con apply y recarga los peldaños.
    const reloadsBefore = api.profiles.listByModel.mock.calls.length
    fireEvent.click(screen.getByTestId("auto-setup-apply"))
    expect(modelEditorApi.autoSetup).toHaveBeenCalledTimes(1)
    fireEvent.click(await screen.findByRole("button", { name: "Reemplazar" }))
    await waitFor(() => expect(modelEditorApi.autoSetup).toHaveBeenLastCalledWith("llamacpp_mks/m", { gpus: [1], cpu: true, apply: true }))
    await waitFor(() => expect(api.profiles.listByModel.mock.calls.length).toBeGreaterThan(reloadsBefore))
  })

  it("auto setup asks to mark the CPU when the model only fits with it", async () => {
    modelEditorApi.autoSetup.mockResolvedValue({ rungs: [], needsCpu: true, warnings: [], budgets: [], applied: [] })
    renderEditor("capacity")
    fireEvent.click(await screen.findByTestId("auto-setup-open"))
    fireEvent.click(screen.getByTestId("auto-setup-calculate"))
    const result = await screen.findByTestId("auto-setup-result")
    expect(result.textContent).toContain("marca la CPU")
    expect(screen.queryByTestId("auto-setup-apply")).toBeNull()
  })

  it("auto setup is not offered for non llama.cpp backends", async () => {
    modelEditorApi.schema.mockResolvedValue({ ...schema, backend: "vllm", section: "vllm" })
    modelEditorApi.get.mockResolvedValue(model({ backend_type: "vllm", model_id: "llamacpp_mks/m" }))
    renderEditor("capacity")
    expect(await screen.findByTestId("capacity-section")).toBeTruthy()
    expect(screen.queryByTestId("auto-setup-open")).toBeNull()
  })

  it("with presets, capacity lists only the presets (default focused) plus the shared settings", async () => {
    const preset = (key: string, ctx: number, isDefault = false) => ({
      id: key, profileId: "m", presetKey: key, label: key, contextMode: "fixed", contextTokens: ctx, parallelism: 2,
      kvCacheDtype: "q8", placementPolicy: "auto", preferredGpus: null, executionPolicy: "auto", enabled: true,
      isDefault, priority: 100, createdAt: null, updatedAt: null,
    })
    api.profiles.listByModel.mockResolvedValue([
      { profileId: "m", profileKind: "execution", baseModelId: "llamacpp_mks/m", displayName: "m", description: null, category: "llm",
        loadOverrides: null, requestDefaults: null, assets: null, enabled: true, isDefault: true, routingTargets: null,
        runtimePolicy: null, runtimePresets: [preset("normal", 32768, true), preset("largo", 131072)], aliases: [], createdAt: null, updatedAt: null },
    ])
    api.profiles.resolveRuntimePresets.mockResolvedValue([
      { presetKey: "normal", resolved: true, requestedContext: 32768, effectiveContext: 32768, modelMax: 262144, hardwareMax: 90000, policyMax: null,
        requestedParallelism: 2, effectiveParallelism: 2, targetGpus: [1], fits: true, runtimeKey: "x", warnings: [], estimatedVramMb: 21000, vramByGpu: { "1": 21000 } },
      { presetKey: "largo", resolved: true, requestedContext: 131072, effectiveContext: 0, modelMax: 262144, hardwareMax: 90000, policyMax: null,
        requestedParallelism: 2, effectiveParallelism: 2, targetGpus: [1], fits: false, runtimeKey: null, warnings: ["no cabe"], estimatedVramMb: null, vramByGpu: {} },
    ])
    renderEditor("capacity")
    const selector = await screen.findByTestId("rung-selector")
    // Con presets la base nunca se carga sola: no es un peldaño, es "Común a todos".
    await waitFor(() => expect(selector.querySelector('[data-rung="normal"]')).toBeTruthy())
    expect(selector.querySelector('[data-rung="base"]')).toBeNull()
    expect(selector.querySelector('[data-rung="largo"]')).toBeTruthy()
    // Entra en el preset por defecto, en su banco de trabajo.
    expect(await screen.findByText("Peldaño «normal»")).toBeTruthy()
    // Dos pruebas: tal cual (cualquier backend) y contexto máximo (solo mks,
    // y con GPUs declaradas: este peldaño es automático).
    expect(screen.getByText("¿Carga y responde así?")).toBeTruthy()
    expect(screen.getByText(/Marca en «Dónde corre» las GPUs del peldaño/)).toBeTruthy()
    fireEvent.click(selector.querySelector('[data-rung="common"]') as HTMLElement)
    const shared = await screen.findByTestId("shared-canvas")
    // Lo que fija cada preset no aparece en lo común.
    expect(shared.querySelector('[data-field="llama_cpp.ctx_size"]')).toBeNull()
    expect(shared.querySelector('[data-field="llama_cpp.split_mode"]')).toBeNull()
  })

  it("embedding models get the encoder canvas and no preset ladder", async () => {
    modelEditorApi.get.mockResolvedValue(
      model({
        capabilities: { embeddings: true, chat: false } as RawModel["capabilities"],
        extra_config: { llama_cpp: { ctx_size: 32768, parallel_slots: 4 } },
      }),
    )
    renderEditor("capacity")
    const panel = await screen.findByTestId("encoder-panel")
    // 32768 total / 4 slots (sin pool) = 8k por texto.
    expect(panel.textContent).toContain("8k tokens por texto")
    expect(screen.queryByText("Nuevo peldaño")).toBeNull()
    expect(screen.queryByTestId("capacity-chart")).toBeNull()
  })

  it("ollama models get the global server card and no preset ladder", async () => {
    modelEditorApi.get.mockResolvedValue(model({ model_id: "ollama/m", backend_type: "ollama", extra_config: {} }))
    modelEditorApi.schema.mockResolvedValue({ ...schema, backend: "ollama", section: "ollama", fields: [f("ollama.num_ctx", "context", { label: "Contexto", default: 65536, widget: "context" })] })
    modelEditorApi.ollamaSettings.mockResolvedValue({ numParallel: 2, maxContext: 65536, running: { numParallel: 2, maxContext: 65536 }, pending: false, applyStatus: "idle", applyDetail: null })
    render(
      <MemoryRouter initialEntries={[`/models/edit/${encodeURIComponent("ollama/m")}?s=capacity`]}>
        <Routes>
          <Route path="/models/edit/*" element={<ModelEditor />} />
        </Routes>
      </MemoryRouter>,
    )
    const card = await screen.findByTestId("ollama-global")
    expect(card.textContent).toContain("64k")
    expect(screen.queryByText("Nuevo peldaño")).toBeNull()
  })

  it("editing a field lists it as a pending change and saves a merge patch", async () => {
    renderEditor("load")
    await screen.findByTestId("load-policy")
    fireEvent.click(screen.getByRole("button", { name: /Caliente/ }))
    const pending = await screen.findByTestId("pending-changes")
    expect(pending.textContent).toContain("Política de carga")
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalledWith("llamacpp_mks/m", { loadPolicy: "warm" }))
  })

  it("clearing a stored option sends null for it only", async () => {
    modelEditorApi.get.mockResolvedValue(model({ extra_config: { keep_alive_seconds: 60, llama_cpp: { ctx_size: 32768 } } }))
    renderEditor("load")
    await screen.findByTestId("load-policy")
    fireEvent.click(await screen.findByRole("button", { name: /^Global/ }))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() =>
      expect(modelEditorApi.patch).toHaveBeenCalledWith("llamacpp_mks/m", { extraConfig: { keep_alive_seconds: null } }),
    )
  })

  it("an invalid stored value does not block saving an unrelated change", async () => {
    modelEditorApi.preview.mockResolvedValue({
      errors: [{ key: "llama_cpp.split_mode", label: "Modo de reparto", message: "valor no válido" }],
      unknownKeys: [],
      estimate: {},
      gpuIndices: [1],
      gpus: [],
      launch: null,
    })
    renderEditor("load")
    await screen.findByTestId("load-policy")
    await waitFor(() => expect(modelEditorApi.preview).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: /Caliente/ }))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalledWith("llamacpp_mks/m", { loadPolicy: "warm" }))
  })

  it("a plain user sees everything read-only", async () => {
    setRole("user")
    renderEditor("load")
    await screen.findByTestId("load-policy")
    expect(screen.getByText("solo lectura")).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Guardar" })).toBeNull()
    expect((screen.getByRole("button", { name: /Caliente/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it("schedule days are stored as cron days (L = 1, D = 0)", async () => {
    renderEditor("load")
    await screen.findByTestId("schedule-grid")
    fireEvent.click(screen.getByRole("button", { name: /Añadir franja/ }))
    const row = await screen.findByTestId("schedule-row")
    fireEvent.click(within(row).getByTitle("domingo"))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(modelEditorApi.patch).toHaveBeenCalled())
    const body = modelEditorApi.patch.mock.calls[0][1]
    expect(body.schedules[0].days).toEqual([0, 1, 2, 3, 4, 5])
  })

  it("summary shows measured performance, cold-load cost and only this model's requests", async () => {
    renderEditor()
    const perf = await screen.findByTestId("performance")
    expect(perf.textContent).toContain("2.0k t/s")
    expect(perf.textContent).toContain("45 s")
    // 4000/2000 + 800/40 = 22 s
    expect(perf.textContent).toContain("22 s")
    const recent = await screen.findByTestId("model-recent")
    expect(recent.querySelectorAll("tr")).toHaveLength(1)
  })

  it("warns when the template only switches thinking on and off", async () => {
    modelEditorApi.schema.mockResolvedValue({
      ...schema,
      fields: [...schema.fields, f("default_reasoning_effort", "response", { type: "enum", label: "Esfuerzo", options: [{ value: "low", label: "Bajo" }] })],
    })
    renderEditor("response")
    expect(await screen.findByText(/solo distingue pensar o no pensar/)).toBeTruthy()
  })

  it("shows the exact launch command", async () => {
    renderEditor()
    fireEvent.click(await screen.findByRole("button", { name: /Comando de arranque/ }))
    expect((await screen.findByTestId("launch-command")).textContent).toContain("CUDA_VISIBLE_DEVICES=1")
  })
})

describe("MemoryBars", () => {
  const gpu3090 = {
    index: 1, name: "NVIDIA GeForce RTX 3090", totalVramMb: 24576, freeVramMb: 1300, usedVramMb: 23276, utilizationPct: 0,
    temperatureC: 40, powerDrawW: 20, powerLimitW: 350, lockedVramMb: 0, processes: [],
  }

  it("draws a measured preset (unknown part key) after the other loaded models and stripes what needs evicting", () => {
    // Otro modelo ocupa ~22,7 GB; el peldaño medido necesita 23,2 GB (cabe en
    // los 23,5 GB que dejan el sistema y el margen del planificador).
    render(<MemoryBars gpus={[gpu3090]} memory={[{ gpu: 1, parts: { measured: 23276 } }]} liveVramByGpu={{}} />)
    const bar = screen.getByTestId("memory-bars").querySelector('[data-gpu-bar="1"]') as HTMLElement
    expect([...bar.querySelectorAll("[data-part]")].map((e) => e.getAttribute("data-part"))).toEqual(["system", "others", "measured", "impossible"])
    expect((bar.querySelector('[data-part="impossible"]') as HTMLElement).title).toContain("desalojar")
    expect(screen.getByTestId("memory-legend").textContent).toContain("medido (todo el modelo)")
    // Cabe desalojando: aviso, no error.
    expect(screen.getByTestId("memory-bars").textContent).toContain("Cabe, pero desalojando otros modelos.")
  })
})
