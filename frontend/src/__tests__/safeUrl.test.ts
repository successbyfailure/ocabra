import { describe, expect, it } from "vitest"
import { safeHttpUrl } from "@/lib/safeUrl"

describe("safeHttpUrl", () => {
  it("keeps http(s) URLs and same-site paths", () => {
    expect(safeHttpUrl("http://10.0.0.100:8188/")).toBe("http://10.0.0.100:8188/")
    expect(safeHttpUrl("https://comfy.example/ui")).toBe("https://comfy.example/ui")
    expect(safeHttpUrl("/services/comfyui/")).toBe("/services/comfyui/")
  })

  it("drops anything that could run code or leave the site oddly", () => {
    expect(safeHttpUrl("javascript:alert(1)")).toBeNull()
    expect(safeHttpUrl(" JavaScript:alert(1)")).toBeNull()
    expect(safeHttpUrl("data:text/html,<script>alert(1)</script>")).toBeNull()
    expect(safeHttpUrl("//evil.example/")).toBeNull()
    expect(safeHttpUrl("")).toBeNull()
    expect(safeHttpUrl(null)).toBeNull()
  })
})
