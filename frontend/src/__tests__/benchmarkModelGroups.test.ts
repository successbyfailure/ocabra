import { describe, expect, it } from "vitest"
import type { LogicalModel, ModelProfile, ProfileCategory } from "@/types"
import { backendVersionLabel, benchmarkModelGroups, localModelOptions } from "@/pages/Benchmarks"

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

describe("benchmarks en un nodo de la federación", () => {
  it("convierte los modelos que anuncia el peer en opciones con su categoría", async () => {
    const { peerModelOptions } = await import("@/pages/Benchmarks")
    const options = peerModelOptions({
      models: [
        { model_id: "llamacpp_mks/qwen3-embedding:0.6b", status: "loaded", profiles: ["qwen3-embedding:0.6b"],
          capabilities: { completion: true, embeddings: true } },
        { model_id: "llamacpp_mks/gemma4:12b", status: "configured", profiles: ["gemma4:12b"],
          capabilities: { chat: true } },
      ],
    } as never)
    expect(options.map((item) => [item.profileId, item.category])).toEqual([
      ["qwen3-embedding:0.6b", "embedding"],
      ["gemma4:12b", "llm"],
    ])
    // Las opciones del peer se agrupan igual que los perfiles locales.
    const groups = benchmarkModelGroups(options, "escriba", "embeddings")
    expect(groups.map((group) => group.key)).toEqual(["qwen3-embedding:0.6b"])
  })
})

function logical(modelKey: string, deployed: string[], compatible: Array<[string, boolean]>): LogicalModel {
  return {
    modelKey,
    displayName: modelKey,
    category: "llm",
    defaultBackend: deployed[0] as LogicalModel["defaultBackend"],
    deployments: deployed.map((backend, index) => ({
      modelId: `${backend}/${modelKey}`,
      backend: backend as LogicalModel["defaultBackend"],
      isDefault: index === 0,
      installed: true,
      status: null,
      loadPolicy: "on_demand",
      pinPermanent: false,
      preferredGpu: null,
      artifact: null,
      profiles: [],
    })),
    compatibleBackends: compatible.map(([backend, installed]) => ({
      backend: backend as LogicalModel["defaultBackend"],
      status: "compatible",
      installed,
      reason: "GGUF",
      deployed: deployed.includes(backend),
    })),
    publicProfiles: [modelKey],
    hiddenProfiles: deployed.slice(1).map((backend) => `${modelKey}-${backend.replace("_", "-")}`),
    registryMetadata: null,
  }
}

describe("backends a elegir al medir aquí o en un pod", () => {
  const profiles = [
    profile("gemma4:12b", "llamacpp_mks/gemma4:12b"),
    profile("gemma4:12b-ollama", "ollama/gemma4:12b"),
  ]
  const models = [
    logical("gemma4:12b", ["llamacpp_mks", "ollama"], [["llamacpp_mks", true], ["llama_cpp", true], ["ollama", true]]),
  ]

  it("pide los otros backends por id de despliegue, también los no desplegados", () => {
    const options = localModelOptions(profiles, models, "local")
    expect(options.map((item) => [item.profileId, item.pending ?? false])).toEqual([
      ["gemma4:12b", false],
      ["llama_cpp/gemma4:12b", true],
      ["ollama/gemma4:12b", false],
    ])
    // El nombre derivado no se ofrece: la API de inferencia lo rechaza.
    expect(options.some((item) => item.profileId === "gemma4:12b-ollama")).toBe(false)
    const groups = benchmarkModelGroups(options, "bfcl", "simple_python")
    expect(groups).toHaveLength(1)
  })

  it("aquí solo ofrece sin desplegar lo que está instalado; en un pod, lo que el pod sabe montar", () => {
    const notInstalled = [logical("gemma4:12b", ["llamacpp_mks"], [["llamacpp_mks", true], ["llama_cpp", false], ["bitnet", true]])]
    expect(localModelOptions(profiles.slice(0, 1), notInstalled, "local").map((item) => item.profileId)).toEqual([
      "gemma4:12b", "bitnet/gemma4:12b",
    ])
    expect(localModelOptions(profiles.slice(0, 1), notInstalled, "cloud").map((item) => item.profileId)).toEqual([
      "gemma4:12b", "llama_cpp/gemma4:12b",
    ])
  })
})
