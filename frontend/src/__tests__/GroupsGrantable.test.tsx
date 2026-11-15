import { describe, expect, it, vi } from "vitest"

const api = vi.hoisted(() => ({
  models: { list: vi.fn() },
  profiles: { listAll: vi.fn() },
  openaiTtsCompat: { list: vi.fn() },
  openaiImageCompat: { list: vi.fn() },
}))
vi.mock("@/api/client", () => ({ api }))
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { listGrantableModels } from "@/pages/Groups"

describe("listGrantableModels", () => {
  it("ofrece modelos, routers y nombres de OpenAI", async () => {
    api.models.list.mockResolvedValue([
      { modelId: "diffusers/qwen-image-2.1", displayName: "Qwen Image", backendType: "diffusers" },
    ])
    api.profiles.listAll.mockResolvedValue([
      { profileId: "recomendado-imagen", profileKind: "router", displayName: "Imagen recomendada" },
      { profileId: "qwen-image-2.1", profileKind: "execution", displayName: null },
    ])
    api.openaiTtsCompat.list.mockResolvedValue([
      { openaiModel: "tts-1", voiceName: "alloy" },
      { openaiModel: "tts-1", voiceName: "echo" },
    ])
    api.openaiImageCompat.list.mockResolvedValue([{ openaiModel: "gpt-image-1" }])

    const items = await listGrantableModels()
    const ids = items.map((i) => `${i.backendType}:${i.modelId}`)
    expect(ids).toEqual([
      "diffusers:diffusers/qwen-image-2.1",
      "router:recomendado-imagen",
      "openai:gpt-image-1",
      "openai:tts-1",
    ])
  })

  it("si falla la lista de perfiles, siguen saliendo los modelos", async () => {
    api.models.list.mockResolvedValue([{ modelId: "m", displayName: "m", backendType: "vllm" }])
    api.profiles.listAll.mockRejectedValue(new Error("boom"))
    api.openaiTtsCompat.list.mockRejectedValue(new Error("boom"))
    api.openaiImageCompat.list.mockRejectedValue(new Error("boom"))
    expect((await listGrantableModels()).map((i) => i.modelId)).toEqual(["m"])
  })
})
