import type { CSSProperties } from "react"

export const BENCHMARK_TOOLTIP_STYLE: CSSProperties = {
  backgroundColor: "#09090b",
  border: "1px solid #52525b",
  borderRadius: 8,
  boxShadow: "0 12px 32px rgb(0 0 0 / 0.72)",
  color: "#fafafa",
  fontSize: 12,
  opacity: 1,
}

export const BENCHMARK_TOOLTIP_WRAPPER_STYLE: CSSProperties = {
  outline: "none",
  zIndex: 50,
}

export const BENCHMARK_TOOLTIP_LABEL_STYLE: CSSProperties = {
  color: "#fafafa",
  fontWeight: 600,
  marginBottom: 4,
}

export const BENCHMARK_TOOLTIP_ITEM_STYLE: CSSProperties = {
  color: "#e4e4e7",
}

export const QUALITY_AXIS_LABEL = "Calidad agregada (%)"
export const PERFORMANCE_AXIS_LABEL = "Rendimiento relativo (%)"
