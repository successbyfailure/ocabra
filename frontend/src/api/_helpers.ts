// Utilidades compartidas del cliente HTTP (REF-15).
// Antes vivían en api/client.ts; no dependen de ningún dominio.

export type AnyRecord = Record<string, unknown>;


export const BASE = "";


export function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null;
}


export function hasAnyKey(record: AnyRecord, keys: string[]): boolean {
  return keys.some((key) => key in record);
}


export function toNumberOrNull(
  record: AnyRecord,
  snakeKey: string,
  camelKey = snakeKey,
): number | null {
  const value = record[snakeKey] ?? record[camelKey];
  return value == null ? null : Number(value);
}


export function toBooleanOrNull(
  record: AnyRecord,
  snakeKey: string,
  camelKey = snakeKey,
): boolean | null {
  const value = record[snakeKey] ?? record[camelKey];
  return value == null ? null : Boolean(value);
}


export function buildQuery(
  params: Record<string, string | number | boolean | undefined>,
): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      search.set(key, String(value));
    }
  });
  const query = search.toString();
  return query ? `?${query}` : "";
}


export function numberMap(raw: unknown): Record<string, number> {
  return Object.fromEntries(
    Object.entries(isRecord(raw) ? raw : {}).map(([key, value]) => [key, Number(value)]),
  );
}

