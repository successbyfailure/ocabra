/**
 * A URL safe to put in an ``href``: http(s) or a same-site path, else null.
 *
 * Anything else (``javascript:``, ``data:``…) would run or render arbitrary
 * content when clicked. Service UI URLs come from server config, but the
 * dashboard should not trust them blindly.
 */
export function safeHttpUrl(url: string | null | undefined): string | null {
  if (!url) return null
  const trimmed = url.trim()
  if (trimmed.startsWith("/") && !trimmed.startsWith("//")) return trimmed
  try {
    const parsed = new URL(trimmed)
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null
  } catch {
    return null
  }
}
