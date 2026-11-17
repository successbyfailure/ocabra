import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const request = vi.fn()
vi.mock("@/api/client", () => ({ request: (...a: unknown[]) => request(...a) }))

import { SelfUpdateSettings } from "@/components/settings/SelfUpdateSettings"

const base = {
  phase: "idle", current: "0.95.6", latest: "0.96.0", checked_at: 1, target: null, message: null, busy: [],
  update_available: true, policy: "notify", idle_minutes: 10,
  deployment: { kind: "compose", reason: "", can_apply: true, api_image: "ghcr.io/x/ocabra-api:0.95.6" },
  updater: null,
}

describe("SelfUpdateSettings", () => {
  beforeEach(() => request.mockReset())

  it("offers the newer release and applies it after confirming", async () => {
    request.mockResolvedValueOnce(base).mockResolvedValueOnce({ ...base, phase: "updating", target: "0.96.0" })
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<SelfUpdateSettings />)
    const button = await screen.findByTestId("self-update-apply")
    expect(button.textContent).toContain("0.96.0")
    fireEvent.click(button)
    await waitFor(() => expect(request).toHaveBeenCalledWith("POST", "/ocabra/system/update/apply", { version: "0.96.0" }))
    expect(await screen.findByTestId("self-update-running")).toBeTruthy()
  })

  it("a stack that builds its own images only gets told", async () => {
    request.mockResolvedValueOnce({
      ...base,
      deployment: { kind: "local_build", reason: "Las imágenes se construyen en local", can_apply: false, api_image: "ocabra-api:slim" },
    })
    render(<SelfUpdateSettings />)
    expect(((await screen.findByTestId("self-update-apply")) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByTestId("self-update").textContent).toContain("se construyen en local")
  })

  it("changing the policy saves it", async () => {
    request.mockResolvedValueOnce(base).mockResolvedValueOnce({ ...base, policy: "auto" })
    render(<SelfUpdateSettings />)
    fireEvent.click(await screen.findByLabelText(/Automático/))
    await waitFor(() => expect(request).toHaveBeenCalledWith("PATCH", "/ocabra/system/update/policy", { policy: "auto" }))
  })
})
