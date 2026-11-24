import { useState } from "react"
import { Loader2, Plus, Search } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import { formatContext } from "@/lib/format"
import type { CloudRecipe, HFModelCard, HFModelVariant, OllamaModelCard, OllamaModelVariant } from "@/types"

const GB = 1024 ** 3
const errMsg = (err: unknown) => (err instanceof Error ? err.message : String(err))
const fieldCls = "rounded border border-border bg-background px-1.5 py-0.5 text-xs"
const btnCls =
  "inline-flex items-center gap-1 rounded-md border border-border px-2 py-0.5 text-xs hover:bg-muted disabled:opacity-50"

const CONTEXTS = [0, 8192, 16384, 32768, 65536, 131072, 262144]
const ctxLabel = (n: number) => (n === 0 ? "por defecto" : formatContext(n))

type Variant = {
  key: string
  label: string
  artifact: string | null
  sizeBytes: number | null
  /** Backend the node will serve it with. */
  backend: string
}

const LLAMA = ["llamacpp_mks", "llama_cpp", "bitnet"]

/** Which of the node's *backends* serves this HF variant (null = none of them). */
export function backendForVariant(v: HFModelVariant, backends: string[]): string | null {
  if (/\.gguf$/i.test(v.artifact ?? "")) {
    return backends.find((b) => LLAMA.includes(b)) ?? null
  }
  const candidates = [v.backendType, ...(v.compatibleBackends ?? []).map((c) => c.backend)]
  return candidates.find((b) => b && backends.includes(b) && !LLAMA.includes(b) && b !== "ollama") ?? null
}

/**
 * Add a model to the cloud catalogue from Hugging Face (a .gguf file) or
 * Ollama (a tag): becomes a recipe (core/cloud/recipes.py) and is picked.
 * llama.cpp only — what a node can download and run without our config.
 */
export function RecipeSearch({
  onAdded,
  backends,
  nameOf = (b) => b,
}: {
  onAdded: (r: CloudRecipe) => void
  /** Backends the node will have: results are what they can serve. */
  backends: string[]
  nameOf?: (backend: string) => string
}) {
  const onlyLlama = backends.every((b) => LLAMA.includes(b))
  const [source, setSource] = useState<"huggingface" | "ollama">("huggingface")
  const [q, setQ] = useState("")
  const [busy, setBusy] = useState(false)
  const [hf, setHf] = useState<HFModelCard[] | null>(null)
  const [ol, setOl] = useState<OllamaModelCard[] | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [variants, setVariants] = useState<Variant[] | null>(null)
  const [adding, setAdding] = useState<string | null>(null)

  const search = async () => {
    if (!q.trim()) return
    setBusy(true)
    setOpen(null)
    setVariants(null)
    try {
      if (source === "huggingface") {
        setOl(null)
        // Only llama.cpp on the node: GGUF repos. Other backends chosen: any repo.
        const term = q.trim()
        setHf(await api.registry.searchHF(onlyLlama && !term.toLowerCase().includes("gguf") ? `${term} gguf` : term))
      } else {
        setHf(null)
        setOl(await api.registry.searchOllama(q.trim()))
      }
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  const openModel = async (name: string) => {
    if (open === name) {
      setOpen(null)
      return
    }
    setOpen(name)
    setVariants(null)
    try {
      if (source === "huggingface") {
        const vs: HFModelVariant[] = await api.registry.getHFVariants(name)
        setVariants(
          vs
            // Model files only: not the vision projector or the imatrix.
            .filter((v) => !/^(mmproj|imatrix)/i.test(v.artifact ?? ""))
            .map((v) => ({ v, backend: backendForVariant(v, backends) }))
            .filter((x): x is { v: HFModelVariant; backend: string } => x.backend !== null)
            .map(({ v, backend }) => ({
              key: v.variantId,
              label: (v.artifact ?? v.label).replace(/\.gguf$/i, ""),
              artifact: v.artifact,
              sizeBytes: v.sizeGb != null ? v.sizeGb * GB : null,
              backend,
            })),
        )
      } else {
        const vs: OllamaModelVariant[] = await api.registry.getOllamaVariants(name)
        setVariants(
          vs.map((v) => ({
            key: v.name, // the full tag: qwen3:8b
            label: [v.name, v.parameterSize, v.quantization].filter(Boolean).join(" · "),
            artifact: null,
            sizeBytes: v.sizeGb != null ? v.sizeGb * GB : null,
            backend: backends.find((b) => LLAMA.includes(b)) ?? "llamacpp_mks",
          })),
        )
      }
    } catch (err) {
      toast.error(errMsg(err))
      setVariants([])
    }
  }

  const add = async (name: string, v: Variant) => {
    setAdding(v.key)
    try {
      const tag = source === "ollama" ? v.key : name
      const r = await api.federation.cloud.createRecipe({
        source,
        model_ref: tag,
        artifact: v.artifact,
        size_bytes: v.sizeBytes != null ? Math.round(v.sizeBytes) : null,
        display_name: source === "huggingface" ? `${name.split("/").pop()} · ${v.label}` : tag,
        backend_type: v.backend,
      })
      onAdded(r)
      toast.success(`${r.display_name} añadido`)
    } catch (err) {
      toast.error(errMsg(err))
    } finally {
      setAdding(null)
    }
  }

  const results: { name: string; detail: string }[] =
    source === "huggingface"
      ? (hf ?? []).map((m) => ({
          name: m.repoId,
          detail: [`${m.downloads.toLocaleString()} descargas`, m.gated ? "con acceso restringido" : null].filter(Boolean).join(" · "),
        }))
      : (ol ?? []).map((m) => ({ name: m.name, detail: `${m.pulls.toLocaleString()} descargas` }))

  return (
    <div className="space-y-2 px-3 py-2" data-testid="recipe-search">
      <div className="flex flex-wrap items-center gap-2">
        <select className={fieldCls} value={source} onChange={(e) => setSource(e.target.value as "huggingface" | "ollama")} aria-label="Dónde buscar">
          <option value="huggingface">Hugging Face (GGUF)</option>
          <option value="ollama">Ollama</option>
        </select>
        <input
          className={`${fieldCls} min-w-0 flex-1`}
          placeholder={source === "huggingface" ? "p. ej. nemotron, qwen3 coder…" : "p. ej. qwen3, gemma4…"}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void search()}
          aria-label="Buscar modelo"
        />
        <button type="button" className={btnCls} disabled={busy || !q.trim()} onClick={() => void search()}>
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />} Buscar
        </button>
      </div>
      {(hf || ol) && results.length === 0 && <p className="text-xs text-muted-foreground">Sin resultados.</p>}
      <div className="max-h-48 space-y-1 overflow-y-auto">
        {results.map((r) => (
          <div key={r.name} className="rounded border border-border/60">
            <button type="button" className="flex w-full items-center gap-2 px-2 py-1 text-left text-xs hover:bg-muted" onClick={() => void openModel(r.name)}>
              <span className="truncate font-medium">{r.name}</span>
              <span className="ml-auto shrink-0 text-muted-foreground">{r.detail}</span>
            </button>
            {open === r.name && (
              <div className="border-t border-border/60 px-2 py-1">
                {variants == null ? (
                  <span className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Loader2 size={12} className="animate-spin" /> Variantes…
                  </span>
                ) : variants.length === 0 ? (
                  <span className="text-xs text-muted-foreground">
                    {source === "huggingface"
                      ? `Nada en este repo que puedan servir los backends del nodo (${backends.map(nameOf).join(", ")}).`
                      : "Sin variantes."}
                  </span>
                ) : (
                  variants.map((v) => (
                    <div key={v.key} className="flex items-center gap-2 py-0.5 text-xs">
                      <span className="truncate">{v.label}</span>
                      <span className="shrink-0 text-[10px] text-muted-foreground">{nameOf(v.backend)}</span>
                      <span className="ml-auto shrink-0 tabular-nums text-muted-foreground">
                        {v.sizeBytes ? `${(v.sizeBytes / GB).toFixed(1)} GB` : ""}
                      </span>
                      <button
                        type="button"
                        className={btnCls}
                        disabled={adding !== null}
                        aria-label={`Añadir ${v.label}`}
                        onClick={() => void add(r.name, v)}
                      >
                        {adding === v.key ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />} Añadir
                      </button>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

/** Context, slots and load policy a recipe's model gets on the next node. */
export function RecipeConfig({ recipe, onSaved }: { recipe: CloudRecipe; onSaved: (r: CloudRecipe) => void }) {
  const [context, setContext] = useState(recipe.context ?? 0)
  const [parallel, setParallel] = useState(recipe.parallel ?? 0)
  const [pin, setPin] = useState(recipe.load_policy === "pin")
  const [busy, setBusy] = useState(false)
  const contexts = CONTEXTS.includes(context) ? CONTEXTS : [...CONTEXTS, context].sort((a, b) => a - b)
  return (
    <div className="flex flex-wrap items-center gap-2 bg-muted/30 px-3 py-1.5 text-xs" data-testid={`recipe-config-${recipe.id}`}>
      <label className="flex items-center gap-1">
        Contexto
        <select className={fieldCls} value={context} onChange={(e) => setContext(Number(e.target.value))} aria-label="Contexto">
          {contexts.map((c) => (
            <option key={c} value={c}>
              {ctxLabel(c)}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1">
        Slots
        <input
          type="number"
          min={0}
          max={64}
          className={`${fieldCls} w-14`}
          value={parallel}
          onChange={(e) => setParallel(Math.max(0, Number(e.target.value) || 0))}
          aria-label="Slots"
          title="0 = por defecto"
        />
      </label>
      <label className="flex items-center gap-1">
        <input type="checkbox" checked={pin} onChange={(e) => setPin(e.target.checked)} />
        Fijado (pin)
      </label>
      <button
        type="button"
        className={`${btnCls} ml-auto`}
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          try {
            onSaved(
              await api.federation.cloud.updateRecipe(recipe.id, {
                context,
                parallel,
                load_policy: pin ? "pin" : "on_demand",
              }),
            )
            toast.success("Configuración guardada")
          } catch (err) {
            toast.error(errMsg(err))
          } finally {
            setBusy(false)
          }
        }}
      >
        {busy && <Loader2 size={12} className="animate-spin" />} Guardar
      </button>
    </div>
  )
}
