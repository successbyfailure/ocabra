import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ModelEditor } from "@/pages/ModelEditor"
import { useAuthStore } from "@/stores/authStore"
import { useGpuStore } from "@/stores/gpuStore"
import type { ConfigSchema, RawModel, SchemaField } from "../types"

const modelEditorApi = vi.hoisted(() => ({
  get: vi.fn(),
  insights: vi.fn(),
  schema: vi.fn(),
  preview: vi.fn(),
  capacity: vi.fn(),
  patch: vi.fn(),
  setMaintenance: vi.fn(),
  createAlias: vi.fn(),
  deleteAlias: vi.fn(),
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

  it("shows the per-GPU preview with this model's live usage subtracted", async () => {
    renderEditor()
    const panel = await screen.findByTestId("preview-panel")
    await waitFor(() => expect(within(panel).getByText(/Cabe ahora/)).toBeTruthy())
    // #1 tiene 20.576 MB usados y 20.000 son de este modelo: le quedan 24.000 MB.
    const bar = panel.querySelector("[data-gpu-bar=\"1\"]") as HTMLElement
    expect(bar.textContent).toContain("21 GB / 23 GB libres")
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
