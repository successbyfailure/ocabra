import { describe, expect, it } from "vitest"
import {
  formatBytes,
  formatContext,
  formatDate,
  formatDateTime,
  formatDuration,
  formatDurationBetween,
  formatMb,
  formatTokens,
  mbToGb,
} from "@/lib/format"

describe("formatTokens (base 1000)", () => {
  it("formats small counts verbatim", () => {
    expect(formatTokens(0)).toBe("0")
    expect(formatTokens(999)).toBe("999")
  })
  it("uses decimal k and M", () => {
    expect(formatTokens(1234)).toBe("1.2k")
    expect(formatTokens(32768)).toBe("32.8k")
    expect(formatTokens(150000)).toBe("150k")
    expect(formatTokens(3_400_000)).toBe("3.4M")
  })
  it("returns a dash for nullish", () => {
    expect(formatTokens(null)).toBe("—")
    expect(formatTokens(undefined)).toBe("—")
  })
})

describe("formatContext (base 1024 for exact powers)", () => {
  it("shows clean k for exact multiples of 1024", () => {
    expect(formatContext(32768)).toBe("32k")
    expect(formatContext(98304)).toBe("96k")
    expect(formatContext(131072)).toBe("128k")
  })
  it("falls back to decimal tokens otherwise", () => {
    expect(formatContext(40000)).toBe("40.0k")
    expect(formatContext(500)).toBe("500")
  })
  it("returns a dash for non-positive", () => {
    expect(formatContext(0)).toBe("—")
    expect(formatContext(null)).toBe("—")
  })
  it("agrees with formatTokens on the base once (32768)", () => {
    // The BUG-54 symptom: the same context value rendered two ways.
    expect(formatContext(32768)).toBe("32k")
    expect(formatTokens(32768)).toBe("32.8k")
  })
})

describe("formatDuration", () => {
  it("formats ms, s, m, h", () => {
    expect(formatDuration(850)).toBe("850 ms")
    expect(formatDuration(12000)).toBe("12 s")
    expect(formatDuration(185000)).toBe("3m05s")
    expect(formatDuration(4320000)).toBe("1h12m")
  })
  it("returns a dash for nullish", () => {
    expect(formatDuration(null)).toBe("—")
  })
})

describe("formatDurationBetween", () => {
  it("measures the gap between two ISO instants", () => {
    expect(formatDurationBetween("2026-01-01T00:00:00Z", "2026-01-01T00:00:12Z")).toBe("12 s")
  })
  it("returns a dash without a start", () => {
    expect(formatDurationBetween(null, null)).toBe("—")
  })
})

describe("formatDate / formatDateTime", () => {
  it("returns the fallback for empty input", () => {
    expect(formatDate(null)).toBe("—")
    expect(formatDate("", "-")).toBe("-")
    expect(formatDateTime(null)).toBe("—")
  })
  it("renders a non-empty string for a valid date", () => {
    expect(formatDate("2026-11-15T10:00:00Z").length).toBeGreaterThan(0)
    expect(formatDateTime("2026-11-15T10:00:00Z").length).toBeGreaterThan(0)
  })
})

describe("MB / bytes helpers", () => {
  it("formatMb switches unit at 1 GB", () => {
    expect(formatMb(512)).toBe("512 MB")
    expect(formatMb(1536)).toBe("1.5 GB")
    expect(formatMb(20480)).toBe("20 GB")
    expect(formatMb(null)).toBe("—")
  })
  it("mbToGb returns a bare number string", () => {
    expect(mbToGb(1536)).toBe("1.5")
    expect(mbToGb(null)).toBe("—")
  })
  it("formatBytes picks a sensible unit", () => {
    expect(formatBytes(2 * 1024 ** 3)).toBe("2.0 GB")
    expect(formatBytes(5 * 1024 ** 2)).toBe("5 MB")
    expect(formatBytes(2048)).toBe("2 KB")
  })
})
