import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ModelProfile } from "@/types"
import { PolicyPanel } from "../presets/PresetParts"

const updateProfile = vi.fn(async (..._a: unknown[]) => undefined)
vi.mock("@/api/client", () => ({
  api: { profiles: { update: (...a: unknown[]) => updateProfile(...a) } },
}))
vi.mock("@/api/modelEditor", () => ({ modelEditorApi: { createAlias: vi.fn(), deleteAlias: vi.fn() } }))

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

const profile = {
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
  runtimePolicy: calibrated,
  runtimePresets: [],
  aliases: [],
  createdAt: null,
  updatedAt: null,
} as unknown as ModelProfile

beforeEach(() => vi.clearAllMocks())

describe("PolicyPanel", () => {
  it("saves the policy without dropping calibrated fields", async () => {
    render(<PolicyPanel profile={profile} readOnly={false} onChanged={async () => undefined} />)
    fireEvent.change(await screen.findByLabelText("Máximo de salida (tokens)"), { target: { value: "32768" } })
    fireEvent.click(screen.getByText("Guardar política"))
    await waitFor(() =>
      expect(updateProfile).toHaveBeenCalledWith("qwen", { runtimePolicy: { ...calibrated, maxOutputTokens: 32768 } }),
    )
  })
})
