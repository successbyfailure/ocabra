import { describe, expect, it } from "vitest"
import { modelSizes, suggestedDiskGb } from "../PodDiskSlider"

const GB = 1024 ** 3

describe("disco sugerido para un pod", () => {
  it("con borrado al terminar basta con los dos más grandes a la vez", () => {
    const sizes = [20 * GB, 18 * GB, 17 * GB, 10 * GB, 5 * GB]
    // 20 + 5 + (20 + 18) × 1,1 = 66,8 → 70
    expect(suggestedDiskGb(sizes, true)).toBe(70)
    // Sin borrar: todos (70 GB de modelos) → 20 + 5 + 77 = 102 → 110
    expect(suggestedDiskGb(sizes, false)).toBe(110)
    expect(suggestedDiskGb([1 * GB], true)).toBe(40)
  })

  it("un backend sin desplegar pesa lo mismo que el fichero que lee", () => {
    const catalog = [
      { model_id: "llamacpp_mks/gemma4:12b", display_name: "g", size_bytes: 8 * GB, replicable: true, loaded: false, reason: null },
    ]
    expect(modelSizes(["llamacpp_mks/gemma4:12b", "ollama/gemma4:12b", "otro/x"], catalog)).toEqual([8 * GB, 8 * GB, 0])
  })
})
