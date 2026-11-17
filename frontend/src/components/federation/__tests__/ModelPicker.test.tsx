import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"
import type { CloudNode, CloudPreloadPick, CloudRecipe, CloudReplicableModel } from "@/types"
import { EMPTY_PICK, ModelPicker } from "../ModelPicker"
import { presetFromNode } from "../CloudSection"

const GB = 1024 ** 3
const ours = (id: string, gb: number): CloudReplicableModel =>
  ({ model_id: id, display_name: id, backend_type: "llamacpp_mks", replicable: true, size_bytes: gb * GB, loaded: false, reason: null }) as CloudReplicableModel
const nemo: CloudRecipe = {
  id: "r1",
  node_model_id: "llamacpp_mks/nemotron-120b",
  display_name: "Nemotron 120B",
  backend_type: "llamacpp_mks",
  source: "huggingface: ggml-org/Nemotron",
  size_bytes: 77 * GB,
  context: 262144,
  parallel: 4,
  vram_estimate_mb: null,
  last_node_name: "ocabra-pro-6000-63b924",
  last_gpu_type_id: "PRO 6000",
  last_used_at: new Date().toISOString(),
  captured_at: new Date().toISOString(),
}

function Harness({ gpuGb, from, onPick }: { gpuGb: number; from?: string; onPick: (p: CloudPreloadPick) => void }) {
  const [v, setV] = useState<CloudPreloadPick>(EMPTY_PICK)
  return (
    <ModelPicker
      value={v}
      onChange={(p) => {
        setV(p)
        onPick(p)
      }}
      gpuMemoryGb={gpuGb}
      autoRecipesFrom={from}
    />
  )
}

afterEach(() => vi.restoreAllMocks())

describe("ModelPicker", () => {
  it("picking more than fits only downloads the extra one instead of warning", async () => {
    vi.spyOn(api.federation.cloud, "replicableModels").mockResolvedValue([ours("a", 16), ours("b", 20)])
    vi.spyOn(api.federation.cloud, "recipes").mockResolvedValue([])
    let last = EMPTY_PICK
    render(<Harness gpuGb={24} onPick={(p) => (last = p)} />)
    fireEvent.click(await screen.findByRole("checkbox", { name: /^a/ }))
    fireEvent.click(screen.getByRole("checkbox", { name: /^b/ }))
    expect(last.models).toEqual(["a", "b"])
    expect(last.downloadOnly).toEqual(["b"]) // it would not fit next to «a»
    expect(screen.queryByTestId("preload-too-big")).toBeNull()
    // Asking to load both is what does not fit: then it says so.
    fireEvent.change(screen.getByLabelText("Qué hacer con b"), { target: { value: "load" } })
    expect(screen.getByTestId("preload-too-big")).toBeTruthy()
  })

  it("offers what pods had under «Recientes en pods» and relaunch picks it", async () => {
    vi.spyOn(api.federation.cloud, "replicableModels").mockResolvedValue([ours("gemma", 16)])
    vi.spyOn(api.federation.cloud, "recipes").mockResolvedValue([nemo])
    let last = EMPTY_PICK
    render(<Harness gpuGb={96} from="ocabra-pro-6000-63b924" onPick={(p) => (last = p)} />)
    expect(await screen.findByText("Recientes en pods")).toBeTruthy()
    expect(screen.getByText(/256k ctx · 4 slots · en pro-6000/)).toBeTruthy()
    await waitFor(() => expect(last.recipes).toEqual(["r1"]))
  })

  it("adds a GGUF from a Hugging Face search as a recipe and picks it", async () => {
    vi.spyOn(api.federation.cloud, "replicableModels").mockResolvedValue([])
    vi.spyOn(api.federation.cloud, "recipes").mockResolvedValue([])
    vi.spyOn(api.registry, "searchHF").mockResolvedValue([
      { repoId: "ggml-org/Nemotron-GGUF", downloads: 1200, gated: false } as never,
    ])
    vi.spyOn(api.registry, "getHFVariants").mockResolvedValue([
      { variantId: "a", label: "Q4", artifact: "Nemotron-Q4_K_M.gguf", sizeGb: 70, quantization: "Q4_K_M" },
      { variantId: "b", label: "Q8", artifact: "mmproj-Q8_0.gguf", sizeGb: 1, quantization: "Q8_0" },
    ] as never)
    const create = vi.spyOn(api.federation.cloud, "createRecipe").mockResolvedValue({ ...nemo, id: "new" })
    let last = EMPTY_PICK
    render(<Harness gpuGb={96} onPick={(p) => (last = p)} />)
    fireEvent.click(await screen.findByRole("button", { name: /Buscar en Hugging Face u Ollama/ }))
    fireEvent.change(screen.getByLabelText("Buscar modelo"), { target: { value: "nemotron" } })
    fireEvent.click(screen.getByRole("button", { name: /^Buscar$/ }))
    fireEvent.click(await screen.findByText("ggml-org/Nemotron-GGUF"))
    expect(await screen.findByText("Nemotron-Q4_K_M")).toBeTruthy()
    expect(screen.queryByText(/mmproj/)).toBeNull() // not a model
    fireEvent.click(screen.getByRole("button", { name: "Añadir Nemotron-Q4_K_M" }))
    await waitFor(() => expect(last.recipes).toEqual(["new"]))
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "huggingface",
        model_ref: "ggml-org/Nemotron-GGUF",
        artifact: "Nemotron-Q4_K_M.gguf",
      }),
    )
  })

  it("an Ollama tag is added with its full name", async () => {
    vi.spyOn(api.federation.cloud, "replicableModels").mockResolvedValue([])
    vi.spyOn(api.federation.cloud, "recipes").mockResolvedValue([])
    vi.spyOn(api.registry, "searchOllama").mockResolvedValue([{ name: "qwen3", pulls: 5 } as never])
    vi.spyOn(api.registry, "getOllamaVariants").mockResolvedValue([
      { name: "qwen3:8b", tag: "8b", sizeGb: 5.2, parameterSize: "8B", quantization: "Q4_K_M" },
    ] as never)
    const create = vi.spyOn(api.federation.cloud, "createRecipe").mockResolvedValue({ ...nemo, id: "q" })
    render(<Harness gpuGb={24} onPick={() => {}} />)
    fireEvent.click(await screen.findByRole("button", { name: /Buscar en Hugging Face u Ollama/ }))
    fireEvent.change(screen.getByLabelText("Dónde buscar"), { target: { value: "ollama" } })
    fireEvent.change(screen.getByLabelText("Buscar modelo"), { target: { value: "qwen3" } })
    fireEvent.click(screen.getByRole("button", { name: /^Buscar$/ }))
    fireEvent.click(await screen.findByText("qwen3"))
    fireEvent.click(await screen.findByRole("button", { name: /Añadir qwen3:8b/ }))
    await waitFor(() =>
      expect(create).toHaveBeenCalledWith(expect.objectContaining({ source: "ollama", model_ref: "qwen3:8b", artifact: null })),
    )
  })

  it("a picked recipe's context and pin can be changed before launching", async () => {
    vi.spyOn(api.federation.cloud, "replicableModels").mockResolvedValue([])
    vi.spyOn(api.federation.cloud, "recipes").mockResolvedValue([nemo])
    const update = vi
      .spyOn(api.federation.cloud, "updateRecipe")
      .mockImplementation(async (_id, c) => ({ ...nemo, context: c.context ?? null, load_policy: c.load_policy }))
    render(<Harness gpuGb={96} onPick={() => {}} />)
    fireEvent.click(await screen.findByRole("checkbox", { name: /Nemotron 120B/ }))
    fireEvent.click(screen.getByRole("button", { name: "Configurar Nemotron 120B" }))
    fireEvent.change(screen.getByLabelText("Contexto"), { target: { value: "131072" } })
    fireEvent.click(screen.getByLabelText("Fijado (pin)"))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() =>
      expect(update).toHaveBeenCalledWith("r1", { context: 131072, parallel: 4, load_policy: "pin" }),
    )
    expect(await screen.findByText(/128k ctx/)).toBeTruthy()
  })

  it("relaunch repeats the node's setup", () => {
    const node = {
      name: "ocabra-pro-6000-63b924",
      gpu_type_id: "NVIDIA RTX PRO 6000",
      gpu_count: 1,
      cloud_type: "SECURE",
      network_volume_id: "vol1",
      privacy_mode: "zero_data",
      preload_models: [
        { model_id: "llamacpp_mks/gemma4:26b", status: "loaded" },
        { model_id: "llamacpp_mks/nemotron-120b", status: "ready", recipe_id: "r1", load: false },
      ],
    } as unknown as CloudNode
    const p = presetFromNode(node)
    expect(p).toMatchObject({ gpuId: "NVIDIA RTX PRO 6000", volumeId: "vol1", zeroData: true, cloudType: "SECURE" })
    expect(p.pick).toEqual({ models: ["llamacpp_mks/gemma4:26b"], recipes: ["r1"], downloadOnly: ["r1"] })
  })
})
