// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PoolingInterface } from "@/components/playground/PoolingInterface"

describe("PoolingInterface", () => {
  it("renders pooling task sections", () => {
    render(
      <PoolingInterface
        modelId="e5-base"
        scoreCapable={false}
        rerankCapable={false}
        classificationCapable={false}
      />,
    )

    expect(screen.getByText("Pooling")).toBeTruthy()
    expect(screen.getByText("Score")).toBeTruthy()
    expect(screen.getByText("Rerank")).toBeTruthy()
    expect(screen.getByText("Classification")).toBeTruthy()
    expect(screen.getByText(/no declara capacidad `score`/i)).toBeTruthy()
    expect(screen.getByText(/no declara capacidad `rerank`/i)).toBeTruthy()
    expect(screen.getByText(/no declara capacidad `classification`/i)).toBeTruthy()
  })
})
