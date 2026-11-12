import { describe, expect, it } from "vitest"
import { buildPatch, describeChanges, draftFromModel, mergePatch, setPath } from "../draft"
import type { RawModel, SchemaField } from "../types"

const base = (extra: Record<string, unknown>): RawModel =>
  ({
    model_id: "llamacpp_mks/m",
    backend_model_id: "m",
    backend_type: "llamacpp_mks",
    display_name: "m",
    status: "configured",
    load_policy: "on_demand",
    pin_permanent: false,
    preferred_gpu: null,
    current_gpu: [],
    vram_used_mb: 0,
    disk_size_bytes: null,
    capabilities: {} as RawModel["capabilities"],
    extra_config: extra,
    schedules: [],
  }) as RawModel

describe("mergePatch", () => {
  it("sends null for a stored key the user cleared, and nothing for untouched ones", () => {
    const stored = { llama_cpp: { ctx_size: 32768, gpu_layers: 99 } }
    const draft = setPath(stored, ["llama_cpp", "ctx_size"], undefined)
    expect(mergePatch(stored, draft)).toEqual({ llama_cpp: { ctx_size: null } })
  })

  it("touching one speculative option keeps spec_type (deep diff, not a section rewrite)", () => {
    const stored = { llama_cpp: { speculative: { spec_type: "draft-mtp", draft_n: 3 } } }
    const draft = setPath(stored, ["llama_cpp", "speculative", "draft_n"], 5)
    expect(mergePatch(stored, draft)).toEqual({ llama_cpp: { speculative: { draft_n: 5 } } })
  })

  it("lists are replaced whole", () => {
    const stored = { llama_cpp: { tensor_split: [1, 1] } }
    expect(mergePatch(stored, { llama_cpp: { tensor_split: [1, 2] } })).toEqual({ llama_cpp: { tensor_split: [1, 2] } })
  })

  it("no change, no patch", () => {
    expect(mergePatch({ a: { b: 1 } }, { a: { b: 1 } })).toBeNull()
  })
})

describe("buildPatch", () => {
  it("never sends the derived paths or internal keys it did not change", () => {
    const model = base({
      llama_cpp: { model_path: "/data/x.gguf", ctx_size: 4096 },
      _runtime: { parallelism: 2 },
      registry_metadata: { source: "ollama" },
    })
    const b = draftFromModel(model)
    const d = { ...b, extraConfig: setPath(b.extraConfig, ["llama_cpp", "ctx_size"], 8192) }
    expect(buildPatch(b, d)).toEqual({ extraConfig: { llama_cpp: { ctx_size: 8192 } } })
  })

  it("top-level fields only when they change", () => {
    const b = draftFromModel(base({}))
    expect(buildPatch(b, { ...b, loadPolicy: "warm" })).toEqual({ loadPolicy: "warm" })
    expect(buildPatch(b, b)).toEqual({})
  })
})

describe("describeChanges", () => {
  const field = (key: string, extra: Partial<SchemaField> = {}): SchemaField =>
    ({ key, path: key.split("."), type: "int", label: key, group: "context", help: "", default: null, nullable: true, reload: true, memory: true, advanced: false, ...extra }) as SchemaField

  it("labels known fields and flags reloads; unknown keys still appear", () => {
    const b = draftFromModel(base({ llama_cpp: { ctx_size: 4096 } }))
    let extra = setPath(b.extraConfig, ["llama_cpp", "ctx_size"], 8192)
    extra = setPath(extra, ["llama_cpp", "weird"], 1)
    const changes = describeChanges(b, { ...b, extraConfig: extra, loadPolicy: "pin" }, [field("llama_cpp.ctx_size", { label: "Contexto" })])
    expect(changes.map((c) => [c.label, c.reload])).toEqual([
      ["Política de carga", false],
      ["Contexto", true],
      ["llama_cpp.weird", true],
    ])
  })
})
