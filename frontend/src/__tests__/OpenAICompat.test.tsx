// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { OpenAICompat } from "@/pages/OpenAICompat"
import type { ModelProfile, OpenAITTSVoiceMapEntry } from "@/types"

const api = vi.hoisted(() => ({
  openaiTtsCompat: {
    list: vi.fn(),
    upsert: vi.fn(),
    delete: vi.fn(),
  },
  profiles: {
    listAll: vi.fn(),
  },
}))

vi.mock("@/api/client", () => ({ api }))
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

function profile(profileId: string): ModelProfile {
  return {
    profileId,
    profileKind: "execution",
    baseModelId: `piper/${profileId}`,
    displayName: null,
    description: null,
    category: "tts",
    loadOverrides: null,
    requestDefaults: null,
    assets: null,
    enabled: true,
    isDefault: false,
    routingTargets: null,
    voiceTargets: null,
    runtimePolicy: null,
    runtimePresets: [],
    aliases: [],
    createdAt: null,
    updatedAt: null,
  }
}

function entry(openaiModel: string, voiceName: string, targetProfileId: string): OpenAITTSVoiceMapEntry {
  return { openaiModel, voiceName, targetProfileId, createdAt: null, updatedAt: null }
}

function jsonResponse(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
}

describe("OpenAICompat", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    api.openaiTtsCompat.list.mockReset()
    api.openaiTtsCompat.upsert.mockReset()
    api.openaiTtsCompat.delete.mockReset()
    api.profiles.listAll.mockReset()
  })

  it("shows the configured pin for a (model, voice) pair", async () => {
    api.openaiTtsCompat.list.mockResolvedValue([entry("tts-1", "alloy", "piper-es-mx-claude-high")])
    api.profiles.listAll.mockResolvedValue([profile("piper-es-mx-claude-high")])

    render(<OpenAICompat />)

    await waitFor(() => expect(api.openaiTtsCompat.list).toHaveBeenCalled())
    const selects = screen.getAllByRole("combobox")
    const alloyRowSelect = selects[0] as HTMLSelectElement
    expect(alloyRowSelect.value).toBe("piper-es-mx-claude-high")
  })

  it("pins an OpenAI (model, voice) pair to a profile", async () => {
    api.openaiTtsCompat.list.mockResolvedValue([])
    api.profiles.listAll.mockResolvedValue([profile("piper-es-mx-claude-high")])
    api.openaiTtsCompat.upsert.mockResolvedValue(entry("tts-1", "alloy", "piper-es-mx-claude-high"))

    render(<OpenAICompat />)
    await waitFor(() => expect(api.profiles.listAll).toHaveBeenCalled())

    const alloyRowSelect = screen.getAllByRole("combobox")[0]
    fireEvent.change(alloyRowSelect, { target: { value: "piper-es-mx-claude-high" } })

    await waitFor(() =>
      expect(api.openaiTtsCompat.upsert).toHaveBeenCalledWith("tts-1", "alloy", "piper-es-mx-claude-high"),
    )
  })

  it("clears a pin by deleting it when the select goes back to unassigned", async () => {
    api.openaiTtsCompat.list.mockResolvedValue([entry("tts-1", "alloy", "piper-es-mx-claude-high")])
    api.profiles.listAll.mockResolvedValue([profile("piper-es-mx-claude-high")])
    api.openaiTtsCompat.delete.mockResolvedValue(undefined)

    render(<OpenAICompat />)
    await waitFor(() => expect(api.openaiTtsCompat.list).toHaveBeenCalled())

    const alloyRowSelect = screen.getAllByRole("combobox")[0]
    fireEvent.change(alloyRowSelect, { target: { value: "" } })

    await waitFor(() => expect(api.openaiTtsCompat.delete).toHaveBeenCalledWith("tts-1", "alloy"))
  })

  it("verifies a pin live and shows the resolved voice", async () => {
    api.openaiTtsCompat.list.mockResolvedValue([entry("tts-1", "alloy", "piper-es-mx-claude-high")])
    api.profiles.listAll.mockResolvedValue([profile("piper-es-mx-claude-high")])
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(jsonResponse({ voices: ["es_MX-claude-high"], model_type: "piper" })),
    )

    render(<OpenAICompat />)
    await waitFor(() => expect(api.openaiTtsCompat.list).toHaveBeenCalled())

    fireEvent.click(screen.getAllByRole("button", { name: "Verificar" })[0])

    await waitFor(() => expect(screen.getByText("→ es_MX-claude-high")).toBeTruthy())
    expect(fetch).toHaveBeenCalledWith("/v1/audio/voices?model=tts-1&voice=alloy")
  })
})
