import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { STTDefaults } from "../profiles/DefaultsForms"
import { isDedicatedWorker, isValidSlug, overridesDiff, stripDiarizationOverrides } from "../profiles/logic"

describe("worker dedicado o compartido", () => {
  it("el interruptor de diarización no hace un worker aparte", () => {
    const overrides = { whisper: { diarization_enabled: true }, diarization_enabled: true }
    expect(stripDiarizationOverrides(overrides)).toEqual({})
    expect(isDedicatedWorker(overrides)).toBe(false)
    expect(isDedicatedWorker({ diarizationEnabled: true })).toBe(false)
  })

  it("cualquier otra sobrescritura sí", () => {
    expect(isDedicatedWorker({ whisper: { diarization_enabled: true, num_workers: 2 } })).toBe(true)
    expect(stripDiarizationOverrides({ whisper: { diarization_enabled: true, num_workers: 2 } })).toEqual({
      whisper: { num_workers: 2 },
    })
    expect(isDedicatedWorker({ llama_cpp: { ctx_size: 8192 } })).toBe(true)
    expect(isDedicatedWorker(null)).toBe(false)
    expect(isDedicatedWorker({})).toBe(false)
  })
})

describe("diff de sobrescrituras", () => {
  it("muestra el valor del despliegue frente al del perfil, hoja a hoja", () => {
    const base = { llama_cpp: { ctx_size: 32768, gpu_layers: 99 } }
    const rows = overridesDiff(base, { llama_cpp: { ctx_size: 8192, flash_attn: true } })
    expect(rows).toEqual([
      { key: "llama_cpp.ctx_size", base: 32768, override: 8192 },
      { key: "llama_cpp.flash_attn", base: undefined, override: true },
    ])
  })
})

describe("id de perfil", () => {
  it.each(["qwen3.8:27b", "mi-modelo", "a_b.c", "x1"])("acepta %s", (id) => expect(isValidSlug(id)).toBe(true))
  it.each(["", "Mayus", "con/barra", "-empieza", "con espacio"])("rechaza %s", (id) =>
    expect(isValidSlug(id)).toBe(false),
  )
})

describe("valores por defecto de transcripción", () => {
  it("Automático quita la clave diarize", () => {
    const onChange = vi.fn()
    render(<STTDefaults defaults={{ diarize: true, language: "es" }} onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Automático" }))
    expect(onChange).toHaveBeenLastCalledWith({ language: "es" })
  })

  it("Activar la pone a true", () => {
    const onChange = vi.fn()
    render(<STTDefaults defaults={{}} onChange={onChange} />)
    fireEvent.click(screen.getByRole("button", { name: "Activar" }))
    expect(onChange).toHaveBeenLastCalledWith({ diarize: true })
  })
})
