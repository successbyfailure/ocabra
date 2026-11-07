/**
 * snake_case / camelCase handling for model `extra_config`.
 *
 * Since migration 0034 the server stores `extra_config` in snake_case
 * (`ctx_size`, `diarization_enabled`) and normalises every write
 * (`backend/ocabra/core/config_keys.py`). The editor works with camelCase
 * fields internally, so it reads through {@link toCamelView} (snake wins when
 * both spellings are present — that is the value the server applies) and
 * writes through {@link toSnakeKeys}.
 *
 * Mirrors the backend rules, including the dicts whose keys belong to someone
 * else's schema (HF config, CLI flags, env...) and are never renamed.
 */

export const FOREIGN_KEYS: ReadonlySet<string> = new Set([
  "hf_overrides",
  "extra_args",
  "env",
  "chat_template_kwargs",
  "limit_mm_per_prompt",
  "speculative_config",
  "override_generation_config",
  "registry_metadata",
  "runtime_capabilities",
  "_runtime",
])

type AnyRecord = Record<string, unknown>

function isPlainRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** `draftPMin` → `draft_p_min`. Same regex as the backend's `camel_to_snake`. */
export function camelToSnake(key: string): string {
  return key.replace(/(?<!^)(?=[A-Z])/g, "_").toLowerCase()
}

/** `draft_p_min` → `draftPMin`. Leading underscores and `_<digit>` are kept. */
export function snakeToCamel(key: string): string {
  return key.replace(/(?<=[a-zA-Z0-9])_([a-z])/g, (_, ch: string) => ch.toUpperCase())
}

/**
 * Rename config keys to snake_case, recursing into sub-sections up to
 * `depth` (never into {@link FOREIGN_KEYS}). When both spellings of a key are
 * present, `prefer` decides which value survives: `"snake"` for stored data,
 * `"camel"` for an edit the user just made in the UI.
 */
export function toSnakeKeys<T>(value: T, depth = 3, prefer: "snake" | "camel" = "snake"): T {
  if (!isPlainRecord(value) || depth <= 0) return value
  const out: AnyRecord = {}
  const keys = Object.keys(value)
  for (const key of keys) {
    const snake = camelToSnake(key)
    const isCamel = snake !== key
    const hasTwin = keys.some((other) => other !== key && camelToSnake(other) === snake)
    if (hasTwin && (prefer === "camel") !== isCamel) continue
    const entry = value[key]
    out[snake] = FOREIGN_KEYS.has(snake) ? entry : toSnakeKeys(entry, depth - 1, prefer)
  }
  return out as T
}

/**
 * camelCase view of a config section for the editor. snake_case wins over a
 * camelCase twin, because snake is what the server persists and applies.
 */
export function toCamelView<T>(value: T, depth = 3): T {
  if (!isPlainRecord(value) || depth <= 0) return value
  const out: AnyRecord = {}
  for (const key of Object.keys(value)) {
    const snake = camelToSnake(key)
    const camel = snakeToCamel(snake)
    const isSnakeSpelling = key === snake
    const twinPresent = isSnakeSpelling ? camel !== key && camel in value : snake in value
    if (twinPresent && !isSnakeSpelling) continue
    const entry = value[key]
    out[camel] = FOREIGN_KEYS.has(snake) ? entry : toCamelView(entry, depth - 1)
  }
  return out as T
}
