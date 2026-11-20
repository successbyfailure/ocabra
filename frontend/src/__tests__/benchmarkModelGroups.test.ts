import { describe, expect, it } from "vitest"
import type { ModelProfile, ProfileCategory } from "@/types"
import { backendVersionLabel, benchmarkModelGroups } from "@/pages/Benchmarks"

function profile(
  profileId: string,
  baseModelId: string,
  category: ProfileCategory = "llm",
): ModelProfile {
  return {
    profileId,
    profileKind: "execution",
    baseModelId,
    displayName: profileId,
    description: null,
    category,
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

describe("benchmark model groups", () => {
  it("groups the same logical model and exposes each backend as an option", () => {
    const groups = benchmarkModelGroups([
      profile("qwen3:8b", "llamacpp_mks/qwen3:8b"),
      profile("qwen3:8b-ollama", "ollama/qwen3:8b"),
      profile("gemma4:12b", "llamacpp_mks/gemma4:12b"),
    ], "bfcl", "simple_python")

    const qwen = groups.find((group) => group.key === "qwen3:8b")!
    expect(qwen.options.map((item) => item.profileId)).toEqual(["qwen3:8b", "qwen3:8b-ollama"])
  })

  it("only offers profile categories compatible with the selected suite", () => {
    const profiles = [
      profile("chat", "llamacpp_mks/chat"),
      profile("whisper", "whisper/whisper", "stt"),
      profile("embed", "llamacpp_mks/embed", "embedding"),
      profile("rerank", "vllm/rerank", "rerank"),
    ]

    expect(benchmarkModelGroups(profiles, "ifeval", "quick").map((group) => group.key)).toEqual(["chat"])
    expect(benchmarkModelGroups(profiles, "escriba", "stt").map((group) => group.key)).toEqual(["whisper"])
    expect(benchmarkModelGroups(profiles, "escriba", "embeddings").map((group) => group.key).sort()).toEqual(["embed", "rerank"])
  })

  it("shows the exact source revision when a native backend has no package version", () => {
    expect(backendVersionLabel({
      type: "llamacpp_mks",
      installedVersion: null,
      sourceCommit: "96fff6c0123456789abcdef",
    })).toBe("llamacpp_mks 96fff6c01234")
    expect(backendVersionLabel({
      type: "vllm",
      installedVersion: "0.12.0",
      sourceCommit: "ignored",
    })).toBe("vllm 0.12.0")
  })
})
