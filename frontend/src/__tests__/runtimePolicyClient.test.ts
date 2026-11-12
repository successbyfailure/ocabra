import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"

describe("runtime_policy round trip", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("saving a profile's policy keeps keys the UI does not edit (reserve_mb_by_preset)", async () => {
    const stored = {
      profile_id: "qwen",
      profile_kind: "execution",
      base_model_id: "llamacpp_mks/qwen",
      category: "llm",
      enabled: true,
      is_default: true,
      runtime_presets: [],
      aliases: [],
      runtime_policy: {
        selection: "prefer_loaded",
        downgrade_grace_seconds: 600,
        reserve_mb_by_preset: { dual: { "0": 2700 } },
        context_probe: { dual: { max: 1 } },
      },
    }
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => ({
      ok: true,
      status: 200,
      json: async () => (init?.method === "PATCH" ? stored : stored),
    }))
    vi.stubGlobal("fetch", fetchMock)
    const profile = await api.profiles.get("qwen")
    await api.profiles.update("qwen", { runtimePolicy: { ...profile.runtimePolicy!, downgradeGraceSeconds: 300 } })
    const body = JSON.parse(String(fetchMock.mock.calls[fetchMock.mock.calls.length - 1]?.[1]?.body))
    expect(body.runtime_policy.reserve_mb_by_preset).toEqual({ dual: { "0": 2700 } })
    expect(body.runtime_policy.downgrade_grace_seconds).toBe(300)
    expect(body.runtime_policy.context_probe).toEqual({ dual: { max: 1 } })
  })
})
