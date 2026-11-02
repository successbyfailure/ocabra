import { resolveModelIcon } from "@/lib/modelFamilyIcon"

const INITIAL_COLORS = [
  "bg-blue-500/20 text-blue-300",
  "bg-emerald-500/20 text-emerald-300",
  "bg-amber-500/20 text-amber-300",
  "bg-purple-500/20 text-purple-300",
  "bg-pink-500/20 text-pink-300",
  "bg-cyan-500/20 text-cyan-300",
]

function hashString(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0
  }
  return hash
}

function initialsFor(displayName: string): string {
  const words = displayName.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return "?"
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
  return (words[0][0] + words[1][0]).toUpperCase()
}

interface ModelIconProps {
  modelId: string
  displayName: string
  backendType: string
  size?: number
  className?: string
}

/** Brand icon for a recognized model family/backend, falling back to a
 * color-hashed initials avatar when nothing matches. See lib/modelFamilyIcon. */
export function ModelIcon({ modelId, displayName, backendType, size = 28, className }: ModelIconProps) {
  const resolved = resolveModelIcon(modelId, displayName, backendType)
  const wrapperClass = `flex shrink-0 items-center justify-center rounded-full ${className ?? ""}`

  if (resolved) {
    return (
      <span
        className={`${wrapperClass} bg-muted`}
        style={{ width: size, height: size }}
        title={resolved.icon.title}
      >
        <svg
          viewBox="0 0 24 24"
          width={size * 0.6}
          height={size * 0.6}
          role="img"
          aria-label={resolved.icon.title}
        >
          <path d={resolved.icon.path} fill={`#${resolved.hex}`} />
        </svg>
      </span>
    )
  }

  const colorClass = INITIAL_COLORS[hashString(displayName || modelId) % INITIAL_COLORS.length]
  return (
    <span
      className={`${wrapperClass} ${colorClass} text-xs font-semibold`}
      style={{ width: size, height: size }}
      title={displayName}
    >
      {initialsFor(displayName || modelId)}
    </span>
  )
}
