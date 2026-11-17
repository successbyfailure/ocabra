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
  openaiImageCompat: {
    list: vi.fn(() => Promise.resolve([])),
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

/** The first voice row's select, once the profile options are rendered.
 *  Waiting for the API call is not enough: setting a <select> to a value it
 *  has no <option> for is ignored, and on a slow runner (GitHub CI) the
 *  options were not there yet, so nothing was saved. */
async function firstRowSelectWithOption(value: string): Promise<HTMLSelectElement> {
  return waitFor(() => {
    const select = screen.getAllByRole("combobox")[0] as HTMLSelectElement
    expect(select.querySelector(`option[value="${value}"]`)).not.toBeNull()
    return select
  })
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
    const alloyRowSelect = await firstRowSelectWithOption("piper-es-mx-claude-high")
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
    // The pinned value must be selected first, or going back to "" is no change.
    const alloyRowSelect = await firstRowSelectWithOption("piper-es-mx-claude-high")
    await waitFor(() => expect(alloyRowSelect.value).toBe("piper-es-mx-claude-high"))
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

  it("pins an OpenAI image model name to an image router", async () => {
    api.openaiTtsCompat.list.mockResolvedValue([])
    const router = { ...profile("recomendado-imagen"), category: "image" as const, profileKind: "router" as const }
    api.profiles.listAll.mockResolvedValue([profile("piper-es-mx-claude-high"), router])
    api.openaiImageCompat.list.mockResolvedValue([])
    api.openaiImageCompat.upsert.mockResolvedValue({
      openaiModel: "gpt-image-1",
      targetProfileId: "recomendado-imagen",
      createdAt: null,
      updatedAt: null,
    })

    render(<OpenAICompat />)
    const select = (await screen.findByLabelText("Perfil para gpt-image-1")) as HTMLSelectElement
    await waitFor(() => expect(select.options.length).toBe(2)) // sin asignar + solo el de imagen
    fireEvent.change(select, { target: { value: "recomendado-imagen" } })

    await waitFor(() =>
      expect(api.openaiImageCompat.upsert).toHaveBeenCalledWith("gpt-image-1", "recomendado-imagen"),
    )
  })
})
