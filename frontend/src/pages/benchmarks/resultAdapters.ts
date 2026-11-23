export type MetricFormat = "number" | "integer" | "percent" | "seconds" | "rate" | "boolean"

export interface BenchmarkMetric {
  key: string
  label: string
  value: number | string | boolean | null
  format: MetricFormat
  better?: "higher" | "lower"
}

export interface BenchmarkResultRow {
  model: string
  scope: string
  status: string | null
  metrics: BenchmarkMetric[]
  raw: Record<string, unknown>
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function metric(
  key: string,
  label: string,
  value: unknown,
  format: MetricFormat,
  better?: "higher" | "lower",
): BenchmarkMetric | null {
  if (typeof value !== "number" && typeof value !== "string" && typeof value !== "boolean") {
    return null
  }
  return { key, label, value, format, better }
}

function present(items: Array<BenchmarkMetric | null>): BenchmarkMetric[] {
  return items.filter((item): item is BenchmarkMetric => item !== null)
}

function parseCorrala(
  outerModel: string,
  result: Record<string, unknown>,
): BenchmarkResultRow[] {
  if (!Array.isArray(result.metrics)) return []
  const rows = result.metrics.flatMap((candidate, index) => {
    const row = asRecord(candidate)
    if (!row) return []
    const quality = asRecord(row.quality) ?? {}
    const speed = asRecord(row.speed) ?? {}
    return [{
      model: typeof row.model === "string" ? row.model : outerModel,
      scope: typeof row.suite === "string" ? row.suite : `resultado ${index + 1}`,
      status: typeof row.status === "string" ? row.status : null,
      metrics: present([
        metric("pass_rate", "Calidad", quality.pass_rate, "percent", "higher"),
        metric("primary_rate", "Primaria", quality.primary_rate, "percent", "higher"),
        metric("latency_s_p50", "Latencia p50", speed.latency_s_p50, "seconds", "lower"),
        metric("decode_tok_s_p50", "Decode", speed.decode_tok_s_p50, "rate", "higher"),
        metric("errors", "Errores", quality.errors, "integer", "lower"),
      ]),
      raw: row,
    }]
  })
  if (rows.length < 2) return rows

  const weighted = result.metrics.flatMap((candidate) => {
    const row = asRecord(candidate)
    const quality = asRecord(row?.quality)
    const runs = row?.runs
    return row && quality && typeof runs === "number" ? [{ row, quality, runs }] : []
  })
  const total = weighted.reduce((sum, item) => sum + item.runs, 0)
  if (total === 0) return rows
  const average = (key: string) => weighted.reduce((sum, item) => (
    sum + (typeof item.quality[key] === "number" ? item.quality[key] * item.runs : 0)
  ), 0) / total
  const errors = weighted.reduce((sum, item) => (
    sum + (typeof item.quality.errors === "number" ? item.quality.errors : 0)
  ), 0)
  return [{
    model: rows[0].model,
    scope: `Global · ${total} casos`,
    status: rows.every((row) => row.status === "completed") ? "completed" : null,
    metrics: present([
      metric("pass_rate", "Calidad", average("pass_rate"), "percent", "higher"),
      metric("primary_rate", "Primaria", average("primary_rate"), "percent", "higher"),
      metric("errors", "Errores", errors, "integer", "lower"),
    ]),
    raw: result,
  }, ...rows]
}

function parseEscriba(
  outerModel: string,
  result: Record<string, unknown>,
): BenchmarkResultRow[] {
  const rows: BenchmarkResultRow[] = []
  Object.entries(result).forEach(([innerModel, candidate]) => {
    const row = asRecord(candidate)
    if (!row || (!asRecord(row.speed) && !asRecord(row.quality))) return
    const speed = asRecord(row.speed) ?? {}
    const qualityGroups = asRecord(row.quality) ?? {}
    const groups = Object.entries(qualityGroups)
      .map(([name, value]) => [name, asRecord(value)] as const)
      .filter((entry): entry is readonly [string, Record<string, unknown>] => entry[1] !== null)
    const quality = groups[0]?.[1] ?? {}
    const transcript = typeof row.transcript_size === "string" ? row.transcript_size : null
    const qualityName = groups[0]?.[0]
    rows.push({
      model: innerModel || outerModel,
      scope: [transcript ? `Transcripción ${transcript}` : null, qualityName].filter(Boolean).join(" · ") || "resultado",
      status: typeof row.status === "string" ? row.status : null,
      metrics: present([
        metric("coverage", "Cobertura", quality.coverage, "percent", "higher"),
        metric("precision", "Precisión", quality.precision, "percent", "higher"),
        metric("claims", "Claims", quality.claims, "integer"),
        metric("valid_json", "JSON válido", quality.valid_json, "boolean", "higher"),
        metric("quality_tok_s", "Velocidad calidad", quality.tok_s, "rate", "higher"),
        metric("speed_tok_s", "Velocidad", speed.tok_s, "rate", "higher"),
        metric("elapsed", "Tiempo", speed.elapsed, "seconds", "lower"),
        metric("out_tokens", "Tokens salida", speed.out_tokens, "integer"),
      ]),
      raw: row,
    })
  })
  return rows
}

const PUBLIC_METRICS: Record<string, [string, MetricFormat, ("higher" | "lower")?]> = {
  strict_prompt_accuracy: ["Strict · prompts", "percent", "higher"],
  strict_instruction_accuracy: ["Strict · instrucciones", "percent", "higher"],
  loose_prompt_accuracy: ["Loose · prompts", "percent", "higher"],
  loose_instruction_accuracy: ["Loose · instrucciones", "percent", "higher"],
  non_live_accuracy: ["Non-live", "percent", "higher"],
  simple_accuracy: ["Simple", "percent", "higher"],
  multiple_accuracy: ["Múltiple", "percent", "higher"],
  parallel_accuracy: ["Paralela", "percent", "higher"],
  parallel_multiple_accuracy: ["Paralela múltiple", "percent", "higher"],
  irrelevance_accuracy: ["Irrelevancia", "percent", "higher"],
  latency_s_p50: ["Latencia p50", "seconds", "lower"],
  input_tokens: ["Tokens entrada", "integer"],
  output_tokens: ["Tokens salida", "integer"],
}

function parsePublic(
  outerModel: string,
  result: Record<string, unknown>,
): BenchmarkResultRow[] {
  if (typeof result.benchmark !== "string") return []
  const values = asRecord(result.metrics)
  if (!values) return []
  const metrics = Object.entries(values).flatMap(([key, value]) => {
    const definition = PUBLIC_METRICS[key]
    if (!definition) return []
    const parsed = metric(key, definition[0], value, definition[1], definition[2])
    return parsed ? [parsed] : []
  })
  return [{
    model: typeof result.model === "string" ? result.model : outerModel,
    scope: `${result.benchmark}${typeof result.cases === "number" ? ` · ${result.cases} casos` : ""}`,
    status: typeof result.status === "string" ? result.status : null,
    metrics,
    raw: result,
  }]
}

function flattenScalars(
  value: Record<string, unknown>,
  prefix = "",
  depth = 0,
): BenchmarkMetric[] {
  if (depth > 2) return []
  const metrics: BenchmarkMetric[] = []
  Object.entries(value).forEach(([key, candidate]) => {
    const path = prefix ? `${prefix}.${key}` : key
    const nested = asRecord(candidate)
    if (nested) {
      metrics.push(...flattenScalars(nested, path, depth + 1))
    } else if (
      typeof candidate === "number"
      || typeof candidate === "boolean"
      || (typeof candidate === "string" && candidate.length <= 80)
    ) {
      metrics.push({
        key: path,
        label: path.replace(/_/g, " "),
        value: candidate,
        format: typeof candidate === "boolean" ? "boolean" : "number",
      })
    }
  })
  return metrics.slice(0, 10)
}

export function parseBenchmarkResults(
  results: Record<string, Record<string, unknown>>,
): BenchmarkResultRow[] {
  return Object.entries(results).flatMap(([model, result]) => {
    const publicBenchmark = parsePublic(model, result)
    if (publicBenchmark.length > 0) return publicBenchmark
    const corrala = parseCorrala(model, result)
    if (corrala.length > 0) return corrala
    const escriba = parseEscriba(model, result)
    if (escriba.length > 0) return escriba
    return [{
      model,
      scope: "resultado",
      status: typeof result.status === "string" ? result.status : null,
      metrics: flattenScalars(result),
      raw: result,
    }]
  })
}

export function formatMetric(item: BenchmarkMetric): string {
  if (item.value === null) return "—"
  if (item.format === "boolean") return item.value ? "Sí" : "No"
  if (typeof item.value !== "number") return String(item.value)
  if (item.format === "percent") return `${(item.value * 100).toFixed(1)}%`
  if (item.format === "seconds") return `${item.value.toFixed(1)} s`
  if (item.format === "rate") return `${item.value.toFixed(1)} tok/s`
  if (item.format === "integer") return Math.round(item.value).toLocaleString()
  return Number.isInteger(item.value) ? item.value.toLocaleString() : item.value.toFixed(3)
}

export function bestMetricValues(rows: BenchmarkResultRow[]): Map<string, number> {
  const result = new Map<string, number>()
  const metrics = rows.flatMap((row) => row.metrics)
  new Set(metrics.map((item) => item.key)).forEach((key) => {
    const candidates = metrics.filter(
      (item): item is BenchmarkMetric & { value: number; better: "higher" | "lower" } => (
        item.key === key && typeof item.value === "number" && item.better !== undefined
      ),
    )
    if (candidates.length < 2) return
    const direction = candidates[0].better
    result.set(
      key,
      direction === "higher"
        ? Math.max(...candidates.map((item) => item.value))
        : Math.min(...candidates.map((item) => item.value)),
    )
  })
  return result
}
