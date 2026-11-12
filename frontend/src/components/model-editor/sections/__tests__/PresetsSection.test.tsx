// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ModelProfile, RuntimePreset, RuntimePresetResolution } from "@/types"
import { PresetsSection } from "../PresetsSection"
import type { SectionProps } from "../types"

const listByModel = vi.fn()
const resolveRuntimePresets = vi.fn()
const createRuntimePreset = vi.fn(async (..._a: unknown[]) => undefined)
const updateRuntimePreset = vi.fn(async (..._a: unknown[]) => undefined)
const deleteRuntimePreset = vi.fn(async (..._a: unknown[]) => undefined)
const updateProfile = vi.fn(async (..._a: unknown[]) => undefined)

vi.mock("@/api/client", () => ({
  api: {
    profiles: {
      listByModel: (...a: unknown[]) => listByModel(...a),
      resolveRuntimePresets: (...a: unknown[]) => resolveRuntimePresets(...a),
      createRuntimePreset: (...a: unknown[]) => createRuntimePreset(...a),
      updateRuntimePreset: (...a: unknown[]) => updateRuntimePreset(...a),
      deleteRuntimePreset: (...a: unknown[]) => deleteRuntimePreset(...a),
      previewRuntimePreset: vi.fn(async () => null),
      benchmarkRuntimePreset: vi.fn(),
      update: (...a: unknown[]) => updateProfile(...a),
    },
  },
}))
vi.mock("@/api/modelEditor", () => ({
  modelEditorApi: { createAlias: vi.fn(), deleteAlias: vi.fn() },
}))
vi.mock("@/components/models/ContextProbe", () => ({ ContextProbe: () => null }))

function preset(key: string, tokens: number | null, extra: Partial<RuntimePreset> = {}): RuntimePreset {
  return {
    id: key,
    profileId: "qwen",
    presetKey: key,
    label: `L-${key}`,
    contextMode: tokens == null ? "max" : "fixed",
    contextTokens: tokens,
    parallelism: 2,
    kvCacheDtype: "q8",
    placementPolicy: "auto",
    preferredGpus: null,
    executionPolicy: "auto",
    enabled: true,
    isDefault: false,
    priority: 100,
    createdAt: null,
    updatedAt: null,
    ...extra,
  }
}

function profile(presets: RuntimePreset[], extra: Partial<ModelProfile> = {}): ModelProfile {
  return {
    profileId: "qwen",
    profileKind: "execution",
    baseModelId: "llamacpp_mks/qwen",
    displayName: "Qwen",
    description: null,
    category: "llm",
    loadOverrides: null,
    requestDefaults: null,
    assets: null,
    enabled: true,
    isDefault: true,
    routingTargets: null,
    runtimePolicy: null,
    runtimePresets: presets,
    aliases: [],
    createdAt: null,
    updatedAt: null,
    ...extra,
  }
}

function resolution(key: string, effective: number): RuntimePresetResolution {
  return {
    presetKey: key,
    resolved: true,
    requestedContext: effective,
    effectiveContext: effective,
    modelMax: 262144,
    hardwareMax: 262144,
    policyMax: null,
    requestedParallelism: 2,
    effectiveParallelism: 2,
    targetGpus: [1],
    fits: true,
    runtimeKey: null,
    warnings: [],
  }
}

function props(readOnly = false): SectionProps {
  return {
    editor: {
      model: { model_id: "llamacpp_mks/qwen", capabilities: { native_context_length: 262144 } },
      refreshLive: vi.fn(async () => undefined),
    } as unknown as SectionProps["editor"],
    readOnly,
    isAdmin: false,
    logical: null,
    refreshLogical: vi.fn(async () => undefined),
    gpus: [],
    liveVramByGpu: {},
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  resolveRuntimePresets.mockResolvedValue([])
})

describe("PresetsSection", () => {
  it("renders the ladder sorted by context, MAX last", async () => {
    listByModel.mockResolvedValue([
      profile([preset("max", null), preset("128k", 131072), preset("32k", 32768, { isDefault: true })]),
    ])
    resolveRuntimePresets.mockResolvedValue([resolution("max", 200000)])
    render(<PresetsSection {...props()} />)
    const ladder = await screen.findByTestId("ladder")
    await waitFor(() => {
      const keys = Array.from(ladder.querySelectorAll("[data-rung]")).map((el) => el.getAttribute("data-rung"))
      expect(keys).toEqual(["32k", "128k", "max"])
    })
  })

  it("does not let the default rung be disabled or deleted while others exist", async () => {
    listByModel.mockResolvedValue([profile([preset("64k", 65536, { isDefault: true }), preset("128k", 131072)])])
    render(<PresetsSection {...props()} />)
    const disabled = (id: string) => (screen.getByTestId(id) as HTMLButtonElement).disabled
    await screen.findByTestId("toggle-64k")
    expect(disabled("toggle-64k")).toBe(true)
    expect(disabled("delete-64k")).toBe(true)
    expect(disabled("toggle-128k")).toBe(false)
    expect(disabled("delete-128k")).toBe(false)
  })

  it("creates the 4-rung suggested ladder in one click", async () => {
    listByModel.mockResolvedValue([profile([])])
    render(<PresetsSection {...props()} />)
    fireEvent.click(await screen.findByTestId("create-ladder"))
    await waitFor(() => expect(createRuntimePreset).toHaveBeenCalledTimes(4))
    const keys = createRuntimePreset.mock.calls.map((c) => (c[1] as { presetKey: string }).presetKey)
    expect(keys).toEqual(["64k", "32k", "128k", "max"])
    expect((createRuntimePreset.mock.calls[0][1] as { isDefault: boolean }).isDefault).toBe(true)
  })

  it("saves the policy without dropping calibrated fields", async () => {
    const calibrated = {
      selection: "prefer_loaded" as const,
      restoreDefaultWhenIdle: true,
      downgradeGraceSeconds: 300,
      minimumResidencySeconds: 120,
      smallRequestsExtendLargeTtl: false,
      remoteWarmThresholdSeconds: 10,
      maxContextTokens: 262144,
      maxOutputTokens: 65536,
      validatedContextTokens: { normal: 98304 },
      validatedVramMb: { normal: 24000 },
      contextProbe: { normal: { pool_tokens: 98304 } },
      defaultOutputReservationTokens: 16384,
    }
    listByModel.mockResolvedValue([profile([preset("normal", 98304, { isDefault: true })], { runtimePolicy: calibrated })])
    render(<PresetsSection {...props()} />)
    fireEvent.change(await screen.findByLabelText("Máximo de salida (tokens)"), { target: { value: "32768" } })
    fireEvent.click(screen.getByText("Guardar política"))
    await waitFor(() =>
      expect(updateProfile).toHaveBeenCalledWith("qwen", {
        runtimePolicy: { ...calibrated, maxOutputTokens: 32768 },
      }),
    )
  })

  it("is read-only without the role", async () => {
    listByModel.mockResolvedValue([profile([preset("64k", 65536, { isDefault: true })])])
    render(<PresetsSection {...props(true)} />)
    const rung = await screen.findByTestId("ladder")
    expect(within(rung).queryByText("Editar")).toBeNull()
    expect(screen.queryByText("Guardar política")).toBeNull()
    expect(screen.queryByText("Añadir peldaño")).toBeNull()
  })
})
