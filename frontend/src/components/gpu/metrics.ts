// Validated, CVD-safe metric hues (dataviz skill). Mid-tones that hold up on
// both light and dark card surfaces. Each metric keeps a fixed identity hue;
// status colors (below) are reserved for "is this a concern".
export const METRIC = {
  util: "#2a78d6", // utilization — accent blue (busy is normal, not alarming)
  mem: "#1baf7a", // memory in use — aqua
  memLocked: "#7a86f0", // locked / reserved / cache — violet, distinct from "used"
  power: "#eb6834", // electrical draw — heat orange
} as const

export function statusColor(pct: number, warn = 75, crit = 90): string {
  if (pct >= crit) return "#dc2626"
  if (pct >= warn) return "#eda100"
  return "#16a34a"
}

export function tempColor(tempC: number): string {
  if (tempC >= 88) return "#dc2626"
  if (tempC >= 78) return "#eda100"
  return "#16a34a"
}

export const pct = (value: number, total: number): number =>
  total > 0 ? Math.max(0, Math.min(100, (value / total) * 100)) : 0
