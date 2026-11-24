/**
 * Formateadores compartidos de toda la UI. Antes había 7 copias de
 * ``formatDate``, 3 de ``formatDuration``, nueve conversiones MB→GB y ~28
 * divisiones ``/1024`` sueltas, y los tokens salían unas veces en base 1000
 * ("32.8k") y otras en base 1024 ("32k") según la pantalla (BUG-54 / DUP-09).
 *
 * Decisión de base para tokens:
 *  - ``formatTokens`` (recuentos arbitrarios: entrada/salida/total) usa **base
 *    1000** (k/M decimal), como cualquier recuento.
 *  - ``formatContext`` (tamaños de ventana de contexto, que se configuran en
 *    potencias de dos) usa **base 1024** cuando el valor es múltiplo exacto,
 *    para que 32768 se muestre como "32k" y no "32.8k"; si no, cae a decimal.
 *  Así todas las etiquetas de contexto coinciden y los recuentos de tokens
 *  son consistentes entre sí.
 */

const DASH = "—"

/** Recuento de tokens en base 1000: 1_234 → "1.2k", 3_400_000 → "3.4M". */
export function formatTokens(n: number | null | undefined): string {
  if (n == null) return DASH
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`
  return String(Math.round(n))
}

/** Tamaño de ventana de contexto: 32768 → "32k", 98304 → "96k", 40000 → "40k"
 *  aprox. Potencias de dos exactas en base 1024; el resto, decimal. */
export function formatContext(n: number | null | undefined): string {
  if (n == null || n <= 0) return DASH
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`
  return formatTokens(n)
}

/** Duración compacta a partir de milisegundos: 850 ms · 12 s · 3m05s · 1h12m. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return DASH
  if (ms < 1000) return `${Math.round(ms)} ms`
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m${String(s % 60).padStart(2, "0")}s`
  const h = Math.floor(m / 60)
  return `${h}h${String(m % 60).padStart(2, "0")}m`
}

/** Duración entre dos instantes ISO (o "en curso" si falta el fin). */
export function formatDurationBetween(
  start: string | null | undefined,
  end: string | null | undefined,
): string {
  if (!start) return DASH
  const a = Date.parse(start)
  if (Number.isNaN(a)) return DASH
  const b = end ? Date.parse(end) : Date.now()
  if (Number.isNaN(b)) return DASH
  return formatDuration(Math.max(0, b - a))
}

/** Fecha corta (es-ES, estilo medio): "15 nov 2026". */
export function formatDate(iso: string | null | undefined, fallback = DASH): string {
  if (!iso) return fallback
  try {
    return new Intl.DateTimeFormat("es-ES", { dateStyle: "medium" }).format(new Date(iso))
  } catch {
    return fallback
  }
}

/** Fecha y hora completas en el locale del navegador. */
export function formatDateTime(iso: string | null | undefined, fallback = DASH): string {
  if (!iso) return fallback
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return fallback
  }
}

/** Megabytes → cadena con unidad: 512 → "512 MB", 1536 → "1.5 GB".
 *  A partir de 10 GB se omiten los decimales. */
export function formatMb(mb: number | null | undefined): string {
  if (mb == null) return DASH
  if (mb >= 1024) return `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB`
  return `${Math.round(mb)} MB`
}

/** Megabytes → GB numérico (sin unidad), 1 decimal: para barras y títulos. */
export function mbToGb(mb: number | null | undefined, digits = 1): string {
  if (mb == null) return DASH
  return (mb / 1024).toFixed(digits)
}

/** Bytes → cadena legible: 1536 → "1.5 KB", … "GB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return DASH
  const gb = bytes / 1024 ** 3
  if (gb >= 1) return `${gb.toFixed(1)} GB`
  const mb = bytes / 1024 ** 2
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}
