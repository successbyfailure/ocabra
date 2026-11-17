import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"
import type { FederationPeer, FederationRules } from "@/types"
import { PeerRulesDialog } from "../RulesEditor"

const NODE: FederationRules = {
  local_first: true,
  allow_loads: true,
  allow_evict: false,
  max_requests_per_hour: null,
  max_wh_per_month: null,
  allow_downloads: false,
  allow_admin: false,
  offered_models: null,
  hours: null,
}

const peer = (mode: "swarm" | "rules") => ({ peer_id: "p1", name: "Mapache", mode, rules: null }) as unknown as FederationPeer

afterEach(() => vi.restoreAllMocks())

describe("PeerRulesDialog", () => {
  it("switches a swarm peer to my rules in one save", async () => {
    vi.spyOn(api.federation, "getRules").mockResolvedValue({ base: NODE, defaults: {}, effective: NODE })
    vi.spyOn(api.federation, "rulesUsage").mockRejectedValue(new Error("x"))
    vi.spyOn(api.models, "list").mockResolvedValue([])
    const update = vi.spyOn(api.federation, "updatePeer").mockResolvedValue(peer("rules"))
    const onSaved = vi.fn()
    render(<PeerRulesDialog peer={peer("swarm")} onOpenChange={() => {}} onSaved={onSaved} />)

    fireEvent.click(screen.getByRole("radio", { name: /Mis reglas/ }))
    await screen.findByText(/aquí cambias solo lo que sea distinto/)
    await waitFor(() => expect(screen.getByRole("button", { name: "Guardar" }).hasAttribute("disabled")).toBe(false))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(update).toHaveBeenCalledWith("p1", { mode: "rules", rules: null }))
    expect(onSaved).toHaveBeenCalled()
  })

  it("back to the swarm sends only the mode", async () => {
    vi.spyOn(api.federation, "getRules").mockResolvedValue({ base: NODE, defaults: {}, effective: NODE })
    vi.spyOn(api.federation, "rulesUsage").mockRejectedValue(new Error("x"))
    vi.spyOn(api.models, "list").mockResolvedValue([])
    const update = vi.spyOn(api.federation, "updatePeer").mockResolvedValue(peer("swarm"))
    render(<PeerRulesDialog peer={peer("rules")} onOpenChange={() => {}} onSaved={() => {}} />)

    fireEvent.click(screen.getByRole("radio", { name: /Enjambre/ }))
    fireEvent.click(screen.getByRole("button", { name: "Guardar" }))
    await waitFor(() => expect(update).toHaveBeenCalledWith("p1", { mode: "swarm" }))
  })
})
