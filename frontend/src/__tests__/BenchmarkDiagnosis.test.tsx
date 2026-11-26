// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

const { list, get, create } = vi.hoisted(() => ({ list: vi.fn(), get: vi.fn(), create: vi.fn() }))

vi.mock("@/api/client", () => ({
  api: {
    benchmarks: {
      diagnoses: {
        list,
        get,
        create,
        fileUrl: (id: string, name: string) => `/files/${id}/${name}`,
      },
    },
  },
}))

import { BenchmarkDiagnoses } from "@/pages/benchmarks/BenchmarkDiagnosis"
import type { BenchmarkRun } from "@/types"

const run = {
  id: "r1",
  app: "bfcl",
  suite: "core",
  attempts: [{ id: "a1", model: "gemma4:12b", status: "completed" }],
} as unknown as BenchmarkRun

const diagnosis = {
  id: "d1",
  status: "completed",
  phase: "completed",
  progress: null,
  error: null,
  attemptId: "a1",
  model: "gemma4:12b",
  report: {
    findings: ["BFCL: 377 fallos son el arnés"],
    summary: {
      total: 3,
      counts: { servido: 2, modelo: 1, plantilla: 0, sampling: 0, limites: 0, no_reproduce: 0, sin_verificar: 0 },
      percent_of_failures: { servido: 66.7, modelo: 33.3, plantilla: 0, sampling: 0, limites: 0, no_reproduce: 0, sin_verificar: 0 },
    },
    cases: [{ case_id: "bfcl:x", category: "modelo", case_category: "simple", failure: "f", evidence: ["e"] }],
    candidates: { train: 1, eval: 0 },
  },
}

describe("BenchmarkDiagnoses", () => {
  it("muestra el resumen por categoría, los hallazgos y las descargas", async () => {
    list.mockResolvedValue([{ ...diagnosis, report: undefined }])
    get.mockResolvedValue(diagnosis)
    render(<BenchmarkDiagnoses run={run} />)
    await waitFor(() => expect(screen.getByText(/Cuantización \/ servido: 2/)).toBeTruthy())
    expect(screen.getByText(/377 fallos son el arnés/)).toBeTruthy()
    expect(screen.getByText("candidates.eval.jsonl").closest("a")?.getAttribute("href")).toBe("/files/d1/candidates.eval.jsonl")
  })

  it("encola un diagnóstico para el último intento completado de cada modelo", async () => {
    list.mockResolvedValue([])
    create.mockResolvedValue({ ...diagnosis, status: "queued" })
    render(<BenchmarkDiagnoses run={run} />)
    fireEvent.click(screen.getByText(/Diagnosticar gemma4:12b/))
    await waitFor(() => expect(create).toHaveBeenCalledWith("a1"))
  })

  it("no aparece en apps que aún no sabe leer", () => {
    const { container } = render(<BenchmarkDiagnoses run={{ ...run, app: "escriba" } as BenchmarkRun} />)
    expect(container.textContent).toBe("")
  })
})

describe("BenchmarkDiagnoses · cancelar y borrar", () => {
  it("cancela uno en curso y borra uno terminado", async () => {
    const cancel = vi.fn().mockResolvedValue({})
    const remove = vi.fn().mockResolvedValue(undefined)
    const { api } = await import("@/api/client")
    Object.assign(api.benchmarks.diagnoses, { cancel, remove })
    list.mockResolvedValue([
      { ...diagnosis, id: "d2", status: "running", phase: "replay", report: undefined, progress: { completed: 1, total: 4 } },
      { ...diagnosis, id: "d3", status: "failed", report: undefined },
    ])
    render(<BenchmarkDiagnoses run={run} />)
    await waitFor(() => expect(screen.getByText(/Cancelar/)).toBeTruthy())
    fireEvent.click(screen.getByText(/Cancelar/))
    await waitFor(() => expect(cancel).toHaveBeenCalledWith("d2"))
    fireEvent.click(screen.getByText(/Borrar/))
    await waitFor(() => expect(remove).toHaveBeenCalledWith("d3"))
  })
})
