// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import * as Tooltip from "@radix-ui/react-tooltip"
import { describe, expect, it, vi } from "vitest"
import { ModelGridCard } from "@/components/models/ModelGridCard"
import type { ModelState } from "@/types"

function model(overrides: Partial<ModelState> = {}): ModelState {
  return {
    modelId: "llama_cpp/chaos",
    displayName: "Chaos Agent",
    backendType: "llama_cpp",
    status: "loading",
    loadPolicy: "warm",
    pinPermanent: false,
    preferredGpu: null,
    currentGpu: [0, 1],
    vramUsedMb: 23_000,
    ramUsedMb: 0,
    diskSizeBytes: null,
    capabilities: {
      chat: true,
      completion: true,
      tools: true,
      vision: false,
      embeddings: false,
      pooling: false,
      rerank: false,
      classification: false,
      score: false,
      reasoning: true,
      imageGeneration: false,
      imageEditing: false,
      audioTranscription: false,
      musicGeneration: false,
      tts: false,
      audioInput: false,
      audioOutput: false,
      videoInput: false,
      streaming: true,
      contextLength: 131_072,
      nativeContextLength: 262_144,
    },
    lastRequestAt: null,
    loadedAt: null,
    runtimePhase: "upsizing",
    runtimeSlots: 4,
    canCancelLoad: true,
    canDownsize: false,
    ...overrides,
  }
}

function renderCard(item: ModelState) {
  const onCancelLoad = vi.fn()
  const onDownsize = vi.fn()
  render(
    <Tooltip.Provider>
      <ModelGridCard
        model={item}
        category="chat"
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onCancelLoad={onCancelLoad}
        onDownsize={onDownsize}
        onTogglePin={vi.fn()}
        onConfigure={vi.fn()}
        onDelete={vi.fn()}
        onEditProfile={vi.fn()}
        onToggleProfileEnabled={vi.fn()}
      />
    </Tooltip.Provider>,
  )
  return { onCancelLoad, onDownsize }
}

describe("ModelGridCard runtime controls", () => {
  it("shows upsize, slots, and a working load-cancel action", () => {
    const { onCancelLoad } = renderCard(model())

    expect(screen.getByText("Ampliando")).toBeTruthy()
    expect(screen.getByText((text) => text.includes("4 slots"))).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: /Cancelar carga/i }))
    expect(onCancelLoad).toHaveBeenCalledWith("llama_cpp/chaos")
  })

  it("offers a one-GPU downsize only when the backend marks it safe", () => {
    const { onDownsize } = renderCard(
      model({
        status: "loaded",
        runtimePhase: "loaded",
        canCancelLoad: false,
        canDownsize: true,
      }),
    )

    fireEvent.click(screen.getByRole("button", { name: /1 GPU/i }))
    expect(onDownsize).toHaveBeenCalledWith("llama_cpp/chaos")
  })
})
