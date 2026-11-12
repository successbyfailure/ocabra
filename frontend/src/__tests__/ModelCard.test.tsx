// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ModelCard, type ModelCardActions } from "@/components/models/ModelCard"
import type { ModelActivityMap, ModelState, RuntimePreset } from "@/types"

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

const NOW = Date.parse("2026-11-12T12:00:00Z")

function preset(presetKey: string, overrides: Partial<RuntimePreset> = {}): RuntimePreset {
  return {
    id: presetKey,
    profileId: "chaos",
    presetKey,
    label: presetKey,
    contextMode: "fixed",
    contextTokens: 32768,
    parallelism: 2,
    kvCacheDtype: "q8",
    placementPolicy: "auto",
    preferredGpus: null,
    executionPolicy: "auto",
    enabled: true,
    isDefault: false,
    priority: 0,
    createdAt: null,
    updatedAt: null,
    ...overrides,
  }
}

function model(overrides: Partial<ModelState> = {}): ModelState {
  return {
    modelId: "llama_cpp/chaos",
    displayName: "Chaos Agent",
    backendType: "llama_cpp",
    status: "loading",
    loadPolicy: "warm",
    pinPermanent: false,
    preferredGpu: null,
    currentGpu: [0, 1],
    vramUsedMb: 23_000,
    ramUsedMb: 0,
    diskSizeBytes: 16 * 1024 ** 3,
    capabilities: {
      chat: true,
      completion: true,
      tools: true,
      vision: false,
      embeddings: false,
      pooling: false,
      rerank: false,
      classification: false,
      score: false,
      reasoning: true,
      imageGeneration: false,
      imageEditing: false,
      audioTranscription: false,
      musicGeneration: false,
      tts: false,
      audioInput: false,
      audioOutput: false,
      videoInput: false,
      streaming: true,
      contextLength: 131_072,
      nativeContextLength: 262_144,
    },
    lastRequestAt: null,
    loadedAt: null,
    runtimePhase: "upsizing",
    runtimeSlots: 4,
    canCancelLoad: true,
    canDownsize: false,
    ...overrides,
  }
}

const EMPTY: ModelActivityMap = { activity: {}, contexts: {}, stuckThresholdSeconds: 300 }

function actions(): ModelCardActions {
  return {
    onLoad: vi.fn(),
    onUnload: vi.fn(),
    onCancelLoad: vi.fn(),
    onDownsize: vi.fn(),
    onEdit: vi.fn(),
    onPlayground: vi.fn(),
    onUpdate: vi.fn(),
    onCompile: vi.fn(),
    onDelete: vi.fn(),
  }
}

function renderCard(item: ModelState, live: ModelActivityMap = EMPTY) {
  const acts = actions()
  render(
    <ModelCard
      deploymentId={item.modelId}
      states={{ [item.modelId]: item }}
      live={live}
      gpus={[]}
      nowMs={NOW}
      actions={acts}
    />,
  )
  return acts
}

describe("ModelCard", () => {
  it("shows the upsize phase and a working load-cancel action", () => {
    const acts = renderCard(model())
    expect(screen.getByText("Ampliando")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: /Cancelar carga/i }))
    expect(acts.onCancelLoad).toHaveBeenCalledWith("llama_cpp/chaos")
  })

  it("offers a one-GPU downsize only when the backend marks it safe", () => {
    const acts = renderCard(model({ status: "loaded", runtimePhase: "loaded", canCancelLoad: false, canDownsize: true }))
    fireEvent.click(screen.getByRole("button", { name: /1 GPU/i }))
    expect(acts.onDownsize).toHaveBeenCalledWith("llama_cpp/chaos")
  })

  it("loaded: live counters, VRAM per GPU, active rung and the idle-unload countdown", () => {
    const item = model({
      status: "loaded",
      runtimePhase: "loaded",
      canCancelLoad: false,
      loadPolicy: "on_demand",
      lastRequestAt: new Date(NOW - 60_000).toISOString(),
      profiles: [
        {
          profileId: "chaos",
          profileKind: "execution",
          baseModelId: "llama_cpp/chaos",
          displayName: null,
          description: null,
          category: "llm",
          loadOverrides: null,
          requestDefaults: null,
          assets: null,
          enabled: true,
          isDefault: true,
          routingTargets: null,
          runtimePolicy: null,
          runtimePresets: [preset("normal", { isDefault: true }), preset("largo", { contextMode: "max", priority: 1 })],
          aliases: [],
          createdAt: null,
          updatedAt: null,
        },
      ],
    })
    const live: ModelActivityMap = {
      ...EMPTY,
      idleTimeoutSeconds: 300,
      vramByGpu: { "llama_cpp/chaos": { "0": 10_240, "1": 12_288 } },
      activity: {
        "llama_cpp/chaos": {
          inFlight: 0,
          oldestSeconds: 0,
          runtimePreset: "largo",
          generationTokensPerSecondTotal: 42,
        },
      },
    }
    renderCard(item, live)
    expect(screen.getByText("Cargado")).toBeTruthy()
    expect(screen.getByText("10.0 GB")).toBeTruthy()
    expect(screen.getByText("12.0 GB")).toBeTruthy()
    expect(screen.getByText("42.0 t/s")).toBeTruthy()
    expect(document.querySelector('[data-rung="largo"]')?.getAttribute("data-active")).toBe("true")
    expect(document.querySelector('[data-rung="normal"]')?.getAttribute("data-active")).toBeNull()
    expect(screen.getByText(/Se descarga por inactividad en 4m00s/)).toBeTruthy()
    expect(screen.getByText("usado hace 1m")).toBeTruthy()
  })

  it("error: shows the message and links to the editor summary", () => {
    const acts = renderCard(
      model({ status: "error", runtimePhase: "error", canCancelLoad: false, errorMessage: "CUDA out of memory\ntrace" }),
    )
    expect(screen.getByText("CUDA out of memory")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Ver en el editor" }))
    expect(acts.onEdit).toHaveBeenCalledWith("llama_cpp/chaos", "s=summary")
  })

  it("shows the maintenance chip only when the model is in maintenance", () => {
    renderCard(model({ status: "configured", runtimePhase: "configured", canCancelLoad: false, currentGpu: [], maintenance: true }))
    expect(screen.getByTestId("maintenance-chip").textContent).toContain("Mantenimiento")
  })

  it("off: Cargar asks the page to load it", () => {
    const acts = renderCard(model({ status: "configured", runtimePhase: "configured", canCancelLoad: false, currentGpu: [] }))
    fireEvent.click(screen.getByRole("button", { name: /Cargar/ }))
    expect(acts.onLoad).toHaveBeenCalledWith("llama_cpp/chaos")
  })
})
