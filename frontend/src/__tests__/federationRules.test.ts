import { describe, expect, it } from "vitest"
import { rulesSummary } from "@/components/federation/RulesEditor"

describe("rulesSummary", () => {
  it("says everything is allowed for the swarm", () => {
    expect(rulesSummary(null)).toBe("todo permitido")
  })

  it("lists only what is restricted", () => {
    expect(
      rulesSummary({
        local_first: true,
        allow_loads: false,
        allow_evict: true,
        max_requests_per_hour: 100,
        max_wh_per_month: null,
        allow_downloads: true,
        allow_admin: false,
        offered_models: ["a", "b"],
        hours: [22, 8],
      }),
    ).toBe("yo primero · solo lo cargado · solo lectura · 100 pet/h · 2 modelos · de 22:00 a 8:00")
  })
})
