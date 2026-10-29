// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { GenerationServicesSettings } from "@/components/settings/GenerationServicesSettings"
import type { GPUState, ServerConfig } from "@/types"

// Only the fields GenerationServicesSettings actually reads — a realistic
// config object matching what /ocabra/config returns in production.
const CONFIG = {
  hunyuanIdleUnloadSeconds: 600,
  hunyuanGenerationGracePeriodS: 120,
  hunyuanPreferredGpu: 1,
  trellis2IdleUnloadSeconds: 300,
  trellis2GenerationGracePeriodS: 120,
  trellis2PreferredGpu: 1,
  comfyuiIdleUnloadSeconds: 600,
  comfyuiGenerationGracePeriodS: 120,
  comfyuiPreferredGpu: 0,
  a1111IdleUnloadSeconds: 600,
  a1111GenerationGracePeriodS: 120,
  a1111PreferredGpu: 0,
  acestepIdleUnloadSeconds: 600,
  acestepGenerationGracePeriodS: 180,
  acestepPreferredGpu: 1,
  unslothIdleUnloadSeconds: 0,
  unslothGenerationGracePeriodS: -1,
  unslothPreferredGpu: 1,
  generationGpuUtilThresholdPct: 50,
} as unknown as ServerConfig

const GPUS: GPUState[] = []

describe("GenerationServicesSettings", () => {
  it("renders all six services without throwing (2026-10-29: a missing key in the initial-state object — masked by an `as LocalState` cast — crashed this tab with no error boundary, blanking the whole Settings page)", () => {
    render(<GenerationServicesSettings config={CONFIG} gpus={GPUS} onSave={vi.fn()} />)

    expect(screen.getByText("Hunyuan3D")).toBeTruthy()
    expect(screen.getByText("TRELLIS.2")).toBeTruthy()
    expect(screen.getByText("ComfyUI")).toBeTruthy()
    expect(screen.getByText("Automatic1111")).toBeTruthy()
    expect(screen.getByText("ACE-Step")).toBeTruthy()
    expect(screen.getByText("Unsloth Studio")).toBeTruthy()
  })
})
