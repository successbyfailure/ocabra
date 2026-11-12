import { useEffect, useState } from "react"
import { cn } from "@/lib/cn"
import { formatValue, getPath, isObject } from "./draft"
import type { Json, SchemaField } from "./types"
import type { ModelEditorState } from "./useModelEditor"
import { Hint, InheritBadge, Panel, Segmented, inputCls } from "./ui"

/** ¿Están puestas las dependencias del campo? Mira el borrador y, si no, el
 *  valor por defecto de la opción de la que depende. */
export function fieldEnabled(field: SchemaField, editor: ModelEditorState): { ok: boolean; reason?: string } {
  if (!field.requires || !editor.draft) return { ok: true }
  for (const [key, expected] of Object.entries(field.requires)) {
    const value = editor.valueOf(key)
    if (JSON.stringify(value ?? null) !== JSON.stringify(expected)) {
      const label = editor.fieldByKey.get(key)?.label ?? key
      return { ok: false, reason: `Requiere «${label}» = ${formatValue(expected)}` }
    }
  }
  return { ok: true }
}

/** Fila de un campo: etiqueta, indicador de herencia, control y ayuda. */
export function FieldRow({
  field,
  editor,
  readOnly,
  children,
  compact,
}: {
  field: SchemaField
  editor: ModelEditorState
  readOnly?: boolean
  children?: React.ReactNode
  compact?: boolean
}) {
  const set = editor.isSet(field.key)
  const dep = fieldEnabled(field, editor)
  const error = editor.previewErrors?.find((e) => e.key === field.key)
  return (
    <div className={cn("grid gap-1", compact ? "" : "py-1.5")} data-field={field.key}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-foreground">{field.label}</span>
        <InheritBadge set={set} disabled={readOnly} onReset={() => editor.setField(field.path, undefined)} />
        {field.reload && set && <span className="text-[10px] text-muted-foreground">· requiere recarga</span>}
      </div>
      <div className={cn(!dep.ok && "pointer-events-none opacity-40")}>
        {children ?? <FieldControl field={field} editor={editor} readOnly={readOnly} />}
      </div>
      {!dep.ok && <Hint tone="warn">{dep.reason}</Hint>}
      {error && <Hint tone="error">{error.message}</Hint>}
      {field.help && <Hint>{field.help}</Hint>}
    </div>
  )
}

function defaultLabel(field: SchemaField): string {
  const d = field.default
  if (d === null || d === undefined) return "por defecto"
  if (field.options) {
    const opt = field.options.find((o) => o.value === d)
    if (opt) return `por defecto (${opt.label})`
  }
  return `por defecto (${formatValue(d)}${field.unit ? ` ${field.unit}` : ""})`
}

/** Control según el tipo del campo. */
export function FieldControl({
  field,
  editor,
  readOnly,
}: {
  field: SchemaField
  editor: ModelEditorState
  readOnly?: boolean
}) {
  const raw = editor.draft ? getPath(editor.draft.extraConfig, field.path) : undefined
  const set = (value: Json | undefined) => editor.setField(field.path, value)

  if (field.type === "bool") {
    const value = raw === undefined ? "default" : raw ? "on" : "off"
    return (
      <Segmented
        size="sm"
        value={value}
        disabled={readOnly}
        options={[
          { value: "default", label: defaultLabel(field) },
          { value: "on", label: "Sí" },
          { value: "off", label: "No" },
        ]}
        onChange={(v) => set(v === "default" ? undefined : v === "on")}
      />
    )
  }

  if (field.type === "enum" && field.options) {
    const opts = field.options
    if (opts.length <= 5) {
      return (
        <Segmented
          size="sm"
          value={raw === undefined ? "__default" : String(raw)}
          disabled={readOnly}
          options={[
            { value: "__default", label: defaultLabel(field) },
            ...opts.map((o) => ({ value: String(o.value), label: o.label })),
          ]}
          onChange={(v) => set(v === "__default" ? undefined : (opts.find((o) => String(o.value) === v)?.value ?? v))}
        />
      )
    }
    return (
      <select
        className={inputCls}
        disabled={readOnly}
        value={raw === undefined ? "" : String(raw)}
        onChange={(e) => set(e.target.value === "" ? undefined : (opts.find((o) => String(o.value) === e.target.value)?.value ?? e.target.value))}
      >
        <option value="">{defaultLabel(field)}</option>
        {opts.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    )
  }

  if (field.type === "int" || field.type === "float") {
    return <NumberInput field={field} value={raw} onChange={set} readOnly={readOnly} />
  }

  if (field.type === "text") {
    return (
      <textarea
        className={cn(inputCls, "min-h-[80px] font-mono text-xs")}
        disabled={readOnly}
        placeholder={defaultLabel(field)}
        value={typeof raw === "string" ? raw : ""}
        onChange={(e) => set(e.target.value === "" ? undefined : e.target.value)}
      />
    )
  }

  if (field.type === "int_list" || field.type === "float_list" || field.type === "string_list") {
    return <ListInput field={field} value={raw} onChange={set} readOnly={readOnly} />
  }

  if (field.type === "json") {
    return <JsonInput value={raw} onChange={set} readOnly={readOnly} placeholder={defaultLabel(field)} />
  }

  return (
    <input
      className={inputCls}
      disabled={readOnly}
      placeholder={defaultLabel(field)}
      value={typeof raw === "string" ? raw : raw == null ? "" : String(raw)}
      onChange={(e) => set(e.target.value === "" ? undefined : e.target.value)}
    />
  )
}

export function NumberInput({
  field,
  value,
  onChange,
  readOnly,
  className,
}: {
  field: Pick<SchemaField, "type" | "min" | "max" | "step" | "unit" | "default" | "label" | "options" | "key">
  value: Json | undefined
  onChange: (value: Json | undefined) => void
  readOnly?: boolean
  className?: string
}) {
  const [text, setText] = useState(value == null ? "" : String(value))
  useEffect(() => setText(value == null ? "" : String(value)), [value])
  const commit = (t: string) => {
    if (t.trim() === "") return onChange(undefined)
    const n = field.type === "int" ? Number.parseInt(t, 10) : Number.parseFloat(t)
    if (Number.isNaN(n)) return
    onChange(n)
  }
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <input
        type="number"
        inputMode={field.type === "int" ? "numeric" : "decimal"}
        className={cn(inputCls, "max-w-[11rem] tabular-nums")}
        disabled={readOnly}
        min={field.min}
        max={field.max}
        step={field.step ?? (field.type === "int" ? 1 : "any")}
        placeholder={field.default == null ? "por defecto" : `por defecto: ${formatValue(field.default)}`}
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          commit(e.target.value)
        }}
      />
      {field.unit && <span className="text-xs text-muted-foreground">{field.unit}</span>}
    </div>
  )
}

function ListInput({
  field,
  value,
  onChange,
  readOnly,
}: {
  field: SchemaField
  value: Json | undefined
  onChange: (value: Json | undefined) => void
  readOnly?: boolean
}) {
  const toText = (v: Json | undefined) => (Array.isArray(v) ? v.join(field.type === "string_list" ? " " : ", ") : "")
  const [text, setText] = useState(toText(value))
  useEffect(() => setText(toText(value)), [value]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <input
      className={cn(inputCls, "font-mono text-xs")}
      disabled={readOnly}
      placeholder={field.type === "string_list" ? "--flag valor --otro" : "1, 2"}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        const t = e.target.value.trim()
        if (!t) return onChange(undefined)
        if (field.type === "string_list") return onChange(t.split(/\s+/))
        const nums = t.split(/[,\s]+/).filter(Boolean).map(Number)
        if (nums.every((n) => !Number.isNaN(n))) onChange(nums)
      }}
    />
  )
}

export function JsonInput({
  value,
  onChange,
  readOnly,
  placeholder,
  minHeight = 90,
}: {
  value: Json | undefined
  onChange: (value: Json | undefined) => void
  readOnly?: boolean
  placeholder?: string
  minHeight?: number
}) {
  const toText = (v: Json | undefined) => (v === undefined ? "" : isObject(v) || Array.isArray(v) ? JSON.stringify(v, null, 2) : String(v))
  const [text, setText] = useState(toText(value))
  const [bad, setBad] = useState(false)
  useEffect(() => {
    setText(toText(value))
    setBad(false)
  }, [JSON.stringify(value ?? null)]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div>
      <textarea
        className={cn(inputCls, "font-mono text-xs", bad && "border-red-500")}
        style={{ minHeight }}
        disabled={readOnly}
        placeholder={placeholder}
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          const t = e.target.value.trim()
          if (!t) {
            setBad(false)
            return onChange(undefined)
          }
          try {
            onChange(JSON.parse(t) as Json)
            setBad(false)
          } catch {
            setBad(true)
          }
        }}
      />
      {bad && <Hint tone="error">JSON no válido: no se aplica hasta que se corrija.</Hint>}
    </div>
  )
}

/** Todos los campos de un grupo que no tienen control a medida. */
export function SchemaFields({
  editor,
  group,
  exclude = [],
  readOnly,
  showAdvanced,
  only,
  title,
  subtitle,
}: {
  editor: ModelEditorState
  group?: string
  exclude?: string[]
  readOnly?: boolean
  showAdvanced?: boolean
  only?: string[]
  /** Con título se envuelve en un Panel, que no se pinta si no hay campos. */
  title?: string
  subtitle?: string
}) {
  const fields = editor.fields.filter(
    (f) =>
      (only ? only.includes(f.key) : f.group === group) &&
      !exclude.includes(f.key) &&
      (showAdvanced || !f.advanced || editor.isSet(f.key)),
  )
  if (!fields.length) return null
  const grid = (
    <div className="grid gap-x-6 gap-y-1 md:grid-cols-2">
      {fields.map((f) => (
        <div key={f.key} className={cn((f.type === "text" || f.type === "json") && "md:col-span-2")}>
          <FieldRow field={f} editor={editor} readOnly={readOnly} />
        </div>
      ))}
    </div>
  )
  return title ? (
    <Panel title={title} subtitle={subtitle}>
      {grid}
    </Panel>
  ) : (
    grid
  )
}
