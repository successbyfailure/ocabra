import { describe, expect, it } from "vitest"
import { fmtWorkSpeed } from "@/components/stats/PerformanceTable"

describe("velocidad en la unidad de cada tipo", () => {
  it("formatea cada unidad", () => {
    expect(fmtWorkSpeed({ kinds: ["audio_transcription"], unit: "x_realtime", value: 22.05, samples: 166 })).toBe("×22.1 tiempo real")
    expect(fmtWorkSpeed({ kinds: ["tts"], unit: "chars_per_s", value: 138.06, samples: 33 })).toBe("138 car/s")
    expect(fmtWorkSpeed({ kinds: ["image_generation"], unit: "s_per_image", value: 6, samples: 2 })).toBe("6.0 s/imagen")
    expect(fmtWorkSpeed({ kinds: ["rerank"], unit: "documents_per_s", value: 200, samples: 1 })).toBe("200.0 docs/s")
  })
})
