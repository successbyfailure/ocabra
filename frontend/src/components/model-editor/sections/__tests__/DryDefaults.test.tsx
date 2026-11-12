import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { LLMDefaults } from "../profiles/DefaultsForms"

// Portado del test del modal viejo (ProfileModal): mismos comportamientos.
function inputFor(label: string): HTMLInputElement {
  return screen.getByText(label).parentElement!.querySelector("input")!
}

describe("LLMDefaults DRY fields", () => {
  it("adds dry_multiplier without dropping existing defaults", () => {
    const onChange = vi.fn()
    render(<LLMDefaults defaults={{ temperature: 0, repeat_penalty: 1.15 }} onChange={onChange} />)
    fireEvent.change(inputFor("Multiplicador DRY"), { target: { value: "0.8" } })
    expect(onChange).toHaveBeenLastCalledWith({ temperature: 0, repeat_penalty: 1.15, dry_multiplier: 0.8 })
  })

  it("shows stored DRY values and clears them when emptied", () => {
    const onChange = vi.fn()
    render(<LLMDefaults defaults={{ dry_multiplier: 0.8, dry_penalty_last_n: 4096 }} onChange={onChange} />)
    const windowInput = inputFor("Ventana (tokens)")
    expect(windowInput.value).toBe("4096")
    fireEvent.change(windowInput, { target: { value: "" } })
    expect(onChange).toHaveBeenLastCalledWith({ dry_multiplier: 0.8 })
  })
})
