import { describe, expect, it } from "vitest"
import { assessPlacement } from "@/components/explore/DevicePlacementPicker"

const RTX = [
  { index: 0, name: "NVIDIA GeForce RTX 3060", totalVramMb: 12288 },
  { index: 1, name: "NVIDIA GeForce RTX 3090", totalVramMb: 24576 },
]
const M1660 = [{ index: 0, name: "NVIDIA GeForce GTX 1660", totalVramMb: 6144 }]

describe("assessPlacement", () => {
  it("cabe en la grande y con dos GPU anuncia el peldaño extendido", () => {
    const a = assessPlacement(17, RTX, [0, 1], false)
    expect(a.verdict).toBe("fits_one")
    expect(a.message).toContain("RTX 3090")
    expect(a.message).toContain("2 GPU")
    expect(a.canInstall).toBe(true)
  })

  it("solo cabe repartido entre las dos", () => {
    expect(assessPlacement(30, RTX, [0, 1], false).verdict).toBe("split")
  })

  it("no cabe sin la CPU: bloquea hasta marcarla", () => {
    const sin = assessPlacement(17, M1660, [0], false)
    expect(sin.verdict).toBe("needs_cpu")
    expect(sin.canInstall).toBe(false)
    const con = assessPlacement(17, M1660, [0], true)
    expect(con.canInstall).toBe(true)
  })

  it("sin dispositivos no deja instalar", () => {
    expect(assessPlacement(4, RTX, [], false).canInstall).toBe(false)
    expect(assessPlacement(4, RTX, [], true).verdict).toBe("cpu_only")
  })

  it("sin tamaño conocido deja decidir al servidor", () => {
    expect(assessPlacement(null, RTX, [1], false).verdict).toBe("unknown")
  })
})
