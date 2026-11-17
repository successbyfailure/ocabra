import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ZeroDataNotice } from "../ZeroDataNotice"

describe("ZeroDataNotice", () => {
  it("says only the last 10 are kept and what they add up to", () => {
    render(
      <ZeroDataNotice
        summary={{
          kept: 10,
          requests: 7,
          errors: 1,
          avgDurationMs: 2500,
          inputTokens: 1200,
          outputTokens: 340,
          avgPredictedTokensPerSecond: 61.5,
        }}
      />,
    )
    expect(screen.getByText(/solo las últimas 10 peticiones/)).toBeTruthy()
    expect(screen.getByText(/7 de 10 · 1 con error · 2.5 s de media/)).toBeTruthy()
    expect(screen.getByText(/61.5 t\/s generando/)).toBeTruthy()
  })
})
