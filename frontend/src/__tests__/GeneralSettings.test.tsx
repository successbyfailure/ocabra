// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import { GeneralSettings } from "@/components/settings/GeneralSettings"
import type { ServerConfig } from "@/types"

// jsdom doesn't implement scrollIntoView, which @radix-ui/react-select calls
// when opening — not a real bug, just a test-environment gap.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

const { modelsList } = vi.hoisted(() => ({ modelsList: vi.fn() }))

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

vi.mock("@/api/client", () => ({
  api: {
    models: { list: modelsList },
  },
}))

const CONFIG = {
  logLevel: "info",
  energyCostEurKwh: 0,
  idleTimeoutSeconds: 300,
  realtimeDefaultSttModel: "",
  realtimeDefaultTtsModel: "",
} as unknown as ServerConfig

describe("GeneralSettings realtime STT/TTS model lists", () => {
  it("lists models by their real capabilities, not a hardcoded backend allowlist (2026-10-29: the old TTS_BACKENDS/STT_BACKENDS sets missed 'piper' and 'vibeasr', leaving the dropdowns permanently empty)", async () => {
    modelsList.mockResolvedValue([
      {
        modelId: "piper/glados-es",
        displayName: "GLaDOS (ES)",
        backendType: "piper",
        capabilities: { tts: true, audioTranscription: false },
      },
      {
        modelId: "vibeasr/microsoft--VibeVoice-ASR-BitNet",
        displayName: "VibeASR",
        backendType: "vibeasr",
        capabilities: { tts: false, audioTranscription: true },
      },
      {
        modelId: "whisper/openai/whisper-large-v3-turbo",
        displayName: "Whisper Large v3 Turbo",
        backendType: "whisper",
        capabilities: { tts: false, audioTranscription: true },
      },
      {
        modelId: "vllm/some-chat-model",
        displayName: "Some Chat Model",
        backendType: "vllm",
        capabilities: { tts: false, audioTranscription: false },
      },
    ])

    render(<GeneralSettings config={CONFIG} onSave={vi.fn()} />)

    await waitFor(() => {
      expect(modelsList).toHaveBeenCalled()
    })

    // Open the STT dropdown and confirm whisper + vibeasr both show up.
    const sttTrigger = screen.getByText("Modelo STT (Whisper)").parentElement!.querySelector("button")!
    sttTrigger.click()
    await waitFor(() => {
      expect(screen.getByText("Whisper Large v3 Turbo")).toBeTruthy()
    })
    expect(screen.getByText("VibeASR")).toBeTruthy()
  })
})
