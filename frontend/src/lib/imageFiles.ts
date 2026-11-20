/** Imágenes de entrada del Playground como ``File`` (URL, data URL o portapapeles). */

export async function fileFromUrl(url: string, name = "input.png"): Promise<File> {
  const res = await fetch(url)
  const blob = await res.blob()
  return new File([blob], name, { type: blob.type || "image/png" })
}

export async function fileFromClipboard(): Promise<File | null> {
  const nav = navigator as Navigator & {
    clipboard?: { read?: () => Promise<ClipboardItem[]> }
  }
  if (!nav.clipboard?.read) return null
  const items = await nav.clipboard.read()
  for (const item of items) {
    for (const type of item.types) {
      if (type.startsWith("image/")) {
        const blob = await item.getType(type)
        return new File([blob], "pasted.png", { type })
      }
    }
  }
  return null
}
