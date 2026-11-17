import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"
import type { ModelFederationStrategy } from "@/types"
import { FederationSection, nodeState } from "../sections/FederationSection"
import type { SectionProps } from "../sections/types"

const EMB = "llamacpp_mks/qwen3-embedding:8b"

const strategy = (over: Partial<ModelFederationStrategy> = {}): ModelFederationStrategy => ({
  model_id: EMB,
  strategy: "auto",
  nodes: [],
  fallback: true,
  local: true,
  peers: [
    { name: "Mapache", peer_id: "m-1", online: true, model: "cold" },
    { name: "pod", peer_id: "p-1", online: false, model: "loaded" },
  ],
  ...over,
})

const props = (readOnly = false) =>
  ({ editor: { model: { model_id: EMB } }, readOnly, isAdmin: true }) as unknown as SectionProps

afterEach(() => vi.restoreAllMocks())

describe("FederationSection", () => {
  it("labels how each node has the model", () => {
    const d = strategy()
    expect(nodeState("Mapache", d).label).toBe("en frío: lo carga")
    expect(nodeState("pod", d).label).toBe("desconectado")
    expect(nodeState("local", d).label).toBe("lo tiene")
    expect(nodeState("nadie", d).label).toBe("nodo desconocido")
  })

  it("prefer Mapache: picks the peer that has it first, then this node, and saves", async () => {
    vi.spyOn(api.federation, "getStrategy").mockResolvedValue(strategy())
    const save = vi
      .spyOn(api.federation, "setStrategy")
      .mockImplementation(async (_id, s) => strategy({ ...s }))
    render(<FederationSection {...props()} />)

    fireEvent.click(await screen.findByRole("button", { name: "Preferir nodos" }))
    expect(screen.getByText("1.")).toBeTruthy()
    expect(screen.getAllByText("Mapache").length).toBeGreaterThan(0)

    // The save shows up right under what changed (the editor's own
    // «Guardar» doesn't cover this section).
    expect(screen.getByTestId("federation-unsaved")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Guardar reparto" }))
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(EMB, { strategy: "prefer", nodes: ["Mapache", "local"], fallback: true }),
    )
    await waitFor(() => expect(screen.queryByTestId("federation-unsaved")).toBeNull())
  })

  it("nothing to save, no save bar; Descartar goes back to what is stored", async () => {
    vi.spyOn(api.federation, "getStrategy").mockResolvedValue(strategy())
    render(<FederationSection {...props()} />)
    fireEvent.click(await screen.findByRole("button", { name: "Solo este nodo" }))
    fireEvent.click(screen.getByRole("button", { name: "Descartar" }))
    expect(screen.queryByTestId("federation-unsaved")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Auto" }))
    expect(screen.queryByTestId("federation-unsaved")).toBeNull()
  })

  it("says so when the node is not in a federation", async () => {
    vi.spyOn(api.federation, "getStrategy").mockRejectedValue(new Error("Federation is not enabled"))
    render(<FederationSection {...props()} />)
    expect(await screen.findByText(/no está en una federación/)).toBeTruthy()
  })
})
