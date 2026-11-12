import type { ReactNode } from "react"
import type { ProfileCategory } from "@/types"
import { Segmented, inputCls } from "../../ui"
import { type Defaults, extractBool, extractField, mergeBool, mergeField } from "./logic"

type FormProps = { defaults: Defaults; onChange: (d: Defaults) => void; readOnly?: boolean }

function Field({ label, hint, children, wide }: { label: string; hint?: string; children: ReactNode; wide?: boolean }) {
  return (
    <div className={wide ? "md:col-span-2" : undefined}>
      <label className="mb-0.5 block text-xs font-medium text-muted-foreground">{label}</label>
      {children}
      {hint && <p className="mt-0.5 text-[10.5px] text-muted-foreground/80">{hint}</p>}
    </div>
  )
}

function Text({
  defaults,
  onChange,
  name,
  placeholder,
  number,
  readOnly,
  testId,
}: FormProps & { name: string; placeholder?: string; number?: boolean; testId?: string }) {
  return (
    <input
      className={inputCls}
      type={number ? "number" : "text"}
      step="any"
      disabled={readOnly}
      data-testid={testId}
      placeholder={placeholder}
      value={extractField(defaults, name)}
      onChange={(e) => onChange(mergeField(defaults, name, e.target.value, !number))}
    />
  )
}

function Select({
  defaults,
  onChange,
  name,
  options,
  readOnly,
}: FormProps & { name: string; options: [string, string][] }) {
  return (
    <select
      className={inputCls}
      disabled={readOnly}
      value={extractField(defaults, name)}
      onChange={(e) => onChange(mergeField(defaults, name, e.target.value, true))}
    >
      <option value="">Sin preferencia</option>
      {options.map(([v, l]) => (
        <option key={v} value={v}>
          {l}
        </option>
      ))}
    </select>
  )
}

/** Automático / Sí / No. Automático quita la clave. */
export function TriState({
  value,
  onChange,
  labelOn = "Sí",
  labelOff = "No",
  readOnly,
  testId,
}: {
  value: boolean | null
  onChange: (v: boolean | null) => void
  labelOn?: string
  labelOff?: string
  readOnly?: boolean
  testId?: string
}) {
  return (
    <Segmented<string>
      size="sm"
      testId={testId}
      disabled={readOnly}
      value={value === null ? "auto" : value ? "on" : "off"}
      options={[
        { value: "auto", label: "Automático" },
        { value: "on", label: labelOn },
        { value: "off", label: labelOff },
      ]}
      onChange={(v) => onChange(v === "auto" ? null : v === "on")}
    />
  )
}

export function LLMDefaults(p: FormProps) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Field label="Temperatura" hint="0 – 2; más alto = más creativo">
        <Text {...p} name="temperature" number placeholder="0.7" />
      </Field>
      <Field label="Máximo de tokens de salida">
        <Text {...p} name="max_tokens" number />
      </Field>
      <Field label="Top P" hint="0 – 1">
        <Text {...p} name="top_p" number placeholder="1.0" />
      </Field>
      <Field label="Top K">
        <Text {...p} name="top_k" number placeholder="40" />
      </Field>
      <Field label="Penalización de presencia" hint="-2 – 2">
        <Text {...p} name="presence_penalty" number placeholder="0" />
      </Field>
      <Field label="Penalización de frecuencia" hint="-2 – 2">
        <Text {...p} name="frequency_penalty" number placeholder="0" />
      </Field>
      <Field label="Prompt de sistema" hint="Se añade si el cliente no manda uno" wide>
        <textarea
          className={`${inputCls} min-h-[60px]`}
          disabled={p.readOnly}
          value={extractField(p.defaults, "system_prompt")}
          onChange={(e) => p.onChange(mergeField(p.defaults, "system_prompt", e.target.value, true))}
        />
      </Field>
      <div className="mt-1 border-t border-border/50 pt-2 md:col-span-2">
        <p className="text-xs font-medium text-foreground">Anti-repetición (DRY)</p>
        <p className="text-[10.5px] text-muted-foreground">
          Penaliza continuar una secuencia que ya apareció: corta bucles sin castigar repeticiones
          legítimas como claves de JSON o código. Solo llama.cpp.
        </p>
      </div>
      <Field label="Multiplicador DRY" hint="0 = desactivado; 0,8 es el valor típico">
        <Text {...p} name="dry_multiplier" number placeholder="0" />
      </Field>
      <Field label="Longitud permitida" hint="Tokens que puede repetir sin penalización; 3">
        <Text {...p} name="dry_allowed_length" number placeholder="2" />
      </Field>
      <Field label="Ventana (tokens)" hint="Cuánto mira hacia atrás; 4096. -1 la desactiva">
        <Text {...p} name="dry_penalty_last_n" number placeholder="64" />
      </Field>
      <Field label="Base" hint="Crecimiento de la penalización; 1,75">
        <Text {...p} name="dry_base" number placeholder="1.75" />
      </Field>
    </div>
  )
}

export function TTSDefaults(p: FormProps) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Field label="Voz" hint="alloy, echo, nova… o el nombre del hablante">
        <Text {...p} name="voice" placeholder="alloy" />
      </Field>
      <Field label="Velocidad" hint="0,25 – 4">
        <Text {...p} name="speed" number placeholder="1.0" />
      </Field>
      <Field label="Idioma">
        <Text {...p} name="language" placeholder="Automático" />
      </Field>
      <Field label="Hablante" hint="Modelos CustomVoice (ryan, vivian…)">
        <Text {...p} name="speaker" />
      </Field>
      <Field label="Formato de audio">
        <Select {...p} name="response_format" options={[["mp3", "MP3"], ["wav", "WAV"], ["opus", "Opus"], ["flac", "FLAC"]]} />
      </Field>
      <Field label="Instrucción de estilo" hint="Para modelos que admiten instrucciones">
        <Text {...p} name="instruct" placeholder="Habla despacio y con calma" />
      </Field>
    </div>
  )
}

export function STTDefaults(p: FormProps) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Field label="Idioma" hint="es, en, fr… o vacío para detectarlo">
        <Text {...p} name="language" placeholder="Automático" />
      </Field>
      <Field label="Diarizar (quién habla)" hint="Automático = lo decide la petición">
        <TriState
          testId="stt-diarize"
          readOnly={p.readOnly}
          value={extractBool(p.defaults, "diarize")}
          onChange={(v) => p.onChange(mergeBool(p.defaults, "diarize", v))}
          labelOn="Activar"
          labelOff="Desactivar"
        />
      </Field>
      <Field label="Temperatura" hint="0 – 1; más bajo = más literal">
        <Text {...p} name="temperature" number placeholder="0.0" />
      </Field>
      <Field label="Formato de respuesta">
        <Select
          {...p}
          name="response_format"
          options={[["json", "JSON"], ["verbose_json", "JSON detallado"], ["text", "Texto"], ["srt", "SRT"], ["vtt", "VTT"]]}
        />
      </Field>
    </div>
  )
}

export function ImageDefaults(p: FormProps) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Field label="Tamaño">
        <Select {...p} name="size" options={[["256x256", "256×256"], ["512x512", "512×512"], ["1024x1024", "1024×1024"]]} />
      </Field>
      <Field label="Pasos" hint="Pasos de inferencia">
        <Text {...p} name="num_inference_steps" number placeholder="30" />
      </Field>
      <Field label="Guidance scale">
        <Text {...p} name="guidance_scale" number placeholder="7.5" />
      </Field>
      <Field label="Prompt negativo">
        <Text {...p} name="negative_prompt" />
      </Field>
    </div>
  )
}

/** Formulario de valores por defecto según la categoría del perfil. */
export function DefaultsForm({ category, ...p }: FormProps & { category: ProfileCategory }) {
  if (category === "llm") return <LLMDefaults {...p} />
  if (category === "tts") return <TTSDefaults {...p} />
  if (category === "stt") return <STTDefaults {...p} />
  if (category === "image") return <ImageDefaults {...p} />
  return (
    <p className="text-xs text-muted-foreground">
      Esta categoría no tiene formulario propio: usa el JSON para fijar parámetros de la petición.
    </p>
  )
}
