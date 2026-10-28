// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { RuntimePresetsPanel } from "@/components/models/RuntimePresetsPanel"
import type { ModelProfile, RuntimePresetResolution } from "@/types"

const previewRuntimePreset = vi.fn(
  async (): Promise<RuntimePresetResolution> => ({
    presetKey: "preview",
    resolved: true,
    requestedContext: 32768,
    effectiveContext: 28867,
    modelMax: 262144,
    hardwareMax: 28867,
    policyMax: null,
    requestedParallelism: 2,
    effectiveParallelism: 2,
    targetGpus: [1],
    fits: false,
    runtimeKey: "runtime:preview",
    warnings: ["contexto 32768 no cabe con 2 slots (máx 28867); reducido a 28867"],
  }),
)

vi.mock("@/api/client", () => ({
  api: {
    profiles: {
      listRuntimePresets: vi.fn(async () => []),
      resolveRuntimePresets: vi.fn(async () => []),
      previewRuntimePreset: (...args: unknown[]) => previewRuntimePreset(...(args as [])),
      update: vi.fn(async () => undefined),
    },
    gpus: {
      list: vi.fn(async () => [
        { index: 0, name: "RTX 3060", totalVramMb: 12288, freeVramMb: 11000, usedVramMb: 1288, utilizationPct: 0, temperatureC: 40, powerDrawW: 20, powerLimitW: 150, lockedVramMb: 0, processes: [] },
        { index: 1, name: "RTX 3090", totalVramMb: 24576, freeVramMb: 20000, usedVramMb: 4576, utilizationPct: 0, temperatureC: 45, powerDrawW: 30, powerLimitW: 305, lockedVramMb: 0, processes: [] },
      ]),
    },
  },
}))

function makeProfile(): ModelProfile {
  return {
    profileId: "qwen3.8:27b",
    profileKind: "execution",
    baseModelId: "llama_cpp/qwen3.8-27b",
    displayName: "Qwen 3.8 27B",
    description: null,
    category: "llm",
    loadOverrides: null,
    requestDefaults: null,
    assets: null,
    enabled: true,
    isDefault: true,
    routingTargets: null,
    runtimePolicy: null,
    runtimePresets: [],
    aliases: [],
    createdAt: null,
    updatedAt: null,
  }
}

describe("RuntimePresetsPanel", () => {
  it("shows a GPU picker and a live capacity preview while editing a draft preset", async () => {
    render(<RuntimePresetsPanel profile={makeProfile()} onChanged={vi.fn()} />)

    fireEvent.click(await screen.findByText("Añadir preset"))

    // GPU picker renders one toggle button per real GPU, not a free-text field.
    await screen.findByText(/GPU 0 · RTX 3060/)
    screen.getByText(/GPU 1 · RTX 3090/)

    // Live preview calls the preview endpoint (debounced) without saving anything.
    await waitFor(() => expect(previewRuntimePreset).toHaveBeenCalled(), { timeout: 2000 })
    await screen.findByText(/No cabe/)
    await screen.findByText(/contexto 32768 no cabe con 2 slots/)
  })
})
