// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { VideoInterface, type VideoJob } from "@/components/playground/VideoInterface"

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

function job(over: Partial<VideoJob> = {}): VideoJob {
  return {
    id: "video_abc",
    model: "ltx-2.5",
    status: "queued",
    progress: 0,
    created_at: 1_700_000_000,
    completed_at: null,
    expires_at: null,
    seconds: "5",
    size: "1280x704",
    seed: null,
    error: null,
    ...over,
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("VideoInterface", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("encola el vídeo con tamaño y duración y muestra el reproductor al terminar", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = []
    let polled = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      calls.push({ url, init })
      if (url.startsWith("/v1/videos?")) return json({ object: "list", data: [], has_more: false })
      if (url === "/v1/videos" && init?.method === "POST") return json(job())
      if (url === "/v1/videos/video_abc") {
        polled += 1
        return json(
          polled > 1
            ? job({ status: "completed", progress: 100, completed_at: 1_700_000_065, seed: 42 })
            : job({ status: "in_progress", progress: 57 }),
        )
      }
      return json({}, 404)
    })

    render(<VideoInterface modelId="ltx-2.5" />)
    fireEvent.change(screen.getByPlaceholderText(/Plano, sujeto/), {
      target: { value: "un taller" },
    })
    fireEvent.click(screen.getByText("Vertical"))
    fireEvent.click(screen.getByText("Generar vídeo"))

    await waitFor(() => expect(calls.some((c) => c.init?.method === "POST")).toBe(true))
    const post = calls.find((c) => c.init?.method === "POST")!
    expect(JSON.parse(String(post.init!.body))).toEqual({
      model: "ltx-2.5",
      prompt: "un taller",
      seconds: "5",
      size: "704x1280",
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600)
    })
    await waitFor(() => expect(screen.getByRole("progressbar")).toBeTruthy())

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600)
    })
    await waitFor(() => {
      const video = document.querySelector("video")
      expect(video?.getAttribute("src")).toBe("/v1/videos/video_abc/content")
    })
    expect(screen.getByText("seed 42")).toBeTruthy()
  })

  it("muestra el error de un vídeo fallido del historial", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      json({
        object: "list",
        data: [job({ status: "failed", error: { code: "insufficient_vram", message: "Sin VRAM" } })],
        has_more: false,
      }),
    )
    render(<VideoInterface modelId="ltx-2.5" />)
    await waitFor(() => expect(screen.getByText("Sin VRAM")).toBeTruthy())
  })
})

describe("VideoInterface con imágenes", () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("con imagen inicial y final envía multipart sin size (proporción de la imagen)", async () => {
    let sent: FormData | null = null
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.startsWith("/v1/videos?")) return json({ object: "list", data: [], has_more: false })
      if (url === "/v1/videos" && init?.method === "POST") {
        sent = init.body as FormData
        return json(job())
      }
      return json(job())
    })
    if (!URL.createObjectURL) {
      Object.defineProperty(URL, "createObjectURL", { value: () => "blob:x", configurable: true })
      Object.defineProperty(URL, "revokeObjectURL", { value: () => undefined, configurable: true })
    }

    render(<VideoInterface modelId="ltx-2.5" />)
    const png = new File([new Uint8Array([137, 80, 78, 71])], "a.png", { type: "image/png" })
    fireEvent.change(screen.getByTestId("video-start-input"), { target: { files: [png] } })
    fireEvent.change(screen.getByTestId("video-end-input"), { target: { files: [png] } })
    fireEvent.change(screen.getByPlaceholderText(/Plano, sujeto/), {
      target: { value: "transición" },
    })
    expect(screen.getByText("según la imagen")).toBeTruthy()
    fireEvent.click(screen.getByText("Generar transición"))

    await waitFor(() => expect(sent).not.toBeNull())
    const fd = sent as unknown as FormData
    expect(fd.get("model")).toBe("ltx-2.5")
    expect(fd.get("prompt")).toBe("transición")
    expect(fd.get("size")).toBeNull()
    expect(fd.get("input_reference")).toBeInstanceOf(File)
    expect(fd.get("last_frame")).toBeInstanceOf(File)
  })
})
