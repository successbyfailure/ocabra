import { useEffect, useMemo, useState } from "react"
import { History, Loader2, Search, Server, Settings2, X } from "lucide-react"
import { toast } from "sonner"
import { api } from "@/api/client"
import type { CloudPreloadPick, CloudRecipe, CloudReplicableModel } from "@/types"
import { RecipeConfig, RecipeSearch } from "./RecipeTools"

export const EMPTY_PICK: CloudPreloadPick = { models: [], recipes: [], downloadOnly: [] }

export const pickCount = (p: CloudPreloadPick) => p.models.length + p.recipes.length

const GB = 1024 ** 3
const gb = (bytes: number | null | undefined) =>
  bytes ? `${(bytes / GB).toFixed(bytes > 10 * GB ? 0 : 1)} GB` : "—"

/** Memory a model takes loaded: its own estimate if it has one, else its size. */
function recipeMemory(r: CloudRecipe): number {
  return r.vram_estimate_mb ? r.vram_estimate_mb * 1024 ** 2 : (r.size_bytes ?? 0)
}

function ago(iso: string | null): string {
  if (!iso) return ""
  const min = Math.round((Date.now() - new Date(iso).getTime()) / 60000)
  if (min < 60) return `hace ${Math.max(1, min)} min`
  if (min < 48 * 60) return `hace ${Math.round(min / 60)} h`
  return `hace ${Math.round(min / 1440)} días`
}

type Row = {
  key: string
  kind: "model" | "recipe"
  name: string
  memory: number
  detail: string
  badge?: string
  disabledReason?: string | null
  onForget?: () => void
  recipe?: CloudRecipe
}

/**
 * What a cloud node takes along: models other pods had («Recientes en pods»,
 * captured from them, core/cloud/recipes.py) and ours («De este nodo»). Each
 * chosen one is either loaded at start or only downloaded (loaded when asked
 * for). The memory warning counts only what will be loaded, by its loaded
 * size — it used to add up the disk size of everything picked and say they
 * would not fit, even when the point was just to have them downloaded.
 */
export function ModelPicker({
  value,
  onChange,
  exclude = [],
  gpuMemoryGb,
  autoRecipesFrom,
}: {
  value: CloudPreloadPick
  onChange: (v: CloudPreloadPick) => void
  exclude?: string[]
  gpuMemoryGb?: number | null
  /** Relaunch: also pick what this (earlier) node had. */
  autoRecipesFrom?: string | null
}) {
  const [models, setModels] = useState<CloudReplicableModel[] | null>(null)
  const [recipes, setRecipes] = useState<CloudRecipe[] | null>(null)
  const [filter, setFilter] = useState("")
  const [searching, setSearching] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)

  useEffect(() => {
    api.federation.cloud
      .replicableModels()
      .then(setModels)
      .catch(() => setModels([]))
    api.federation.cloud
      .recipes()
      .then(setRecipes)
      .catch(() => setRecipes([]))
  }, [])

  // Relaunch: add what that node had, once its recipes are known.
  useEffect(() => {
    if (!autoRecipesFrom || !recipes) return
    const fromIt = recipes
      .filter((r) => r.last_node_name === autoRecipesFrom && !value.recipes.includes(r.id))
      .map((r) => r.id)
    if (fromIt.length) onChange({ ...value, recipes: [...value.recipes, ...fromIt] })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRecipesFrom, recipes])

  const memoryOf = useMemo(() => {
    const m = new Map<string, number>()
    for (const x of models ?? []) m.set(x.model_id, x.size_bytes ?? 0)
    for (const r of recipes ?? []) m.set(r.id, recipeMemory(r))
    return m
  }, [models, recipes])

  const picked = [...value.models, ...value.recipes]
  const loading = picked.filter((k) => !value.downloadOnly.includes(k))
  const loadBytes = loading.reduce((acc, k) => acc + (memoryOf.get(k) ?? 0), 0)
  const diskBytes = picked.reduce((acc, k) => acc + (memoryOf.get(k) ?? 0), 0)
  const budget = gpuMemoryGb != null ? gpuMemoryGb * GB : null
  const tooBig = budget != null && loadBytes > budget

  const toggle = (row: Row) => {
    const list = row.kind === "model" ? "models" : "recipes"
    const on = value[list].includes(row.key)
    if (on) {
      onChange({
        ...value,
        [list]: value[list].filter((x) => x !== row.key),
        downloadOnly: value.downloadOnly.filter((x) => x !== row.key),
      })
      return
    }
    // Load it if it still fits next to what is already being loaded;
    // otherwise only download it (it loads when someone asks for it).
    const fits = budget == null || loadBytes + row.memory <= budget
    onChange({
      ...value,
      [list]: [...value[list], row.key],
      downloadOnly: fits ? value.downloadOnly : [...value.downloadOnly, row.key],
    })
  }
  const setLoad = (key: string, load: boolean) =>
    onChange({
      ...value,
      downloadOnly: load ? value.downloadOnly.filter((x) => x !== key) : [...value.downloadOnly, key],
    })

  const q = filter.toLowerCase()
  const match = (text: string) => !q || text.toLowerCase().includes(q)
  const ourIds = new Set((models ?? []).map((m) => m.model_id))
  const recipeRows: Row[] = (recipes ?? [])
    .filter((r) => !exclude.includes(r.node_model_id) && !ourIds.has(r.node_model_id))
    .filter((r) => match(`${r.display_name} ${r.node_model_id} ${r.source ?? ""}`))
    .map((r) => ({
      key: r.id,
      kind: "recipe",
      recipe: r,
      name: r.display_name,
      memory: recipeMemory(r),
      detail: [
        gb(r.size_bytes),
        r.context ? `${Math.round(r.context / 1024)}k ctx` : null,
        r.parallel ? `${r.parallel} slots` : null,
        r.last_node_name ? `en ${r.last_node_name.replace(/^ocabra-/, "")}` : null,
        ago(r.last_used_at ?? r.captured_at),
      ]
        .filter(Boolean)
        .join(" · "),
      onForget: async () => {
        if (!window.confirm(`¿Olvidar ${r.display_name}? Volverá si un pod en marcha lo tiene.`)) return
        try {
          await api.federation.cloud.deleteRecipe(r.id)
          setRecipes((rs) => (rs ?? []).filter((x) => x.id !== r.id))
          onChange({
            ...value,
            recipes: value.recipes.filter((x) => x !== r.id),
            downloadOnly: value.downloadOnly.filter((x) => x !== r.id),
          })
        } catch (err) {
          toast.error(err instanceof Error ? err.message : String(err))
        }
      },
    }))
  const modelRows: Row[] = (models ?? [])
    .filter((m) => !exclude.includes(m.model_id))
    .filter((m) => match(`${m.display_name} ${m.model_id} ${m.backend_type === "diffusers" ? "imagen" : ""}`))
    .map((m) => ({
      key: m.model_id,
      kind: "model",
      name: m.display_name,
      memory: m.size_bytes ?? 0,
      detail: m.replicable ? gb(m.size_bytes) : "no replicable",
      badge: m.backend_type === "diffusers" ? "imagen" : m.loaded ? "cargado aquí" : undefined,
      disabledReason: m.replicable ? null : (m.reason ?? "no replicable"),
    }))

  const renderRow = (row: Row) => {
    const list = row.kind === "model" ? value.models : value.recipes
    const on = list.includes(row.key)
    const load = !value.downloadOnly.includes(row.key)
    const recipe = row.recipe
    const config =
      on && recipe && editing === row.key ? (
        <RecipeConfig
          key={`${row.key}-config`}
          recipe={recipe}
          onSaved={(r) => {
            setRecipes((rs) => (rs ?? []).map((x) => (x.id === r.id ? r : x)))
            setEditing(null)
          }}
        />
      ) : null
    return [
      <div
        key={row.key}
        title={row.disabledReason ?? undefined}
        className={`flex items-center gap-2 border-b border-border/50 px-3 py-1.5 text-sm last:border-b-0 ${
          row.disabledReason ? "opacity-50" : "hover:bg-muted"
        }`}
      >
        <label className={`flex min-w-0 flex-1 items-center gap-2 ${row.disabledReason ? "cursor-not-allowed" : "cursor-pointer"}`}>
          <input type="checkbox" disabled={Boolean(row.disabledReason)} checked={on} onChange={() => toggle(row)} />
          <span className="truncate">{row.name}</span>
          {row.badge && (
            <span className="shrink-0 rounded-full border border-border px-1.5 text-[10px] text-muted-foreground">{row.badge}</span>
          )}
          <span className="ml-auto shrink-0 truncate text-xs tabular-nums text-muted-foreground">{row.detail}</span>
        </label>
        {on && (
          <select
            aria-label={`Qué hacer con ${row.name}`}
            className="shrink-0 rounded border border-border bg-background px-1 py-0.5 text-[11px]"
            value={load ? "load" : "download"}
            onChange={(e) => setLoad(row.key, e.target.value === "load")}
          >
            <option value="load">cargar al arrancar</option>
            <option value="download">solo descargar</option>
          </select>
        )}
        {on && row.recipe && (
          <button
            type="button"
            aria-label={`Configurar ${row.name}`}
            title="Contexto, slots y fijado en el nodo"
            className={`shrink-0 rounded p-0.5 hover:bg-muted ${editing === row.key ? "text-primary" : "text-muted-foreground"}`}
            onClick={() => setEditing(editing === row.key ? null : row.key)}
          >
            <Settings2 size={12} />
          </button>
        )}
        {row.onForget && !on && (
          <button type="button" aria-label={`Olvidar ${row.name}`} title="Olvidar" className="shrink-0 rounded p-0.5 text-muted-foreground hover:bg-muted" onClick={row.onForget}>
            <X size={12} />
          </button>
        )}
      </div>,
      config,
    ]
  }

  const addRecipe = (r: CloudRecipe) => {
    setRecipes((rs) => [r, ...(rs ?? []).filter((x) => x.id !== r.id)])
    if (!value.recipes.includes(r.id)) {
      toggle({ key: r.id, kind: "recipe", name: r.display_name, memory: recipeMemory(r), detail: "" })
    }
  }

  const ready = models != null && recipes != null
  return (
    <div className="rounded-md border border-border">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <span className="text-sm font-medium">Modelos a llevar</span>
        <span className="text-xs text-muted-foreground">
          {picked.length === 0
            ? "ninguno: el nodo arranca vacío"
            : `${picked.length} elegidos · ${gb(diskBytes)} · ${loading.length} se cargan al arrancar`}
        </span>
        <input
          className="ml-auto w-40 rounded-md border border-border bg-background px-2 py-1 text-xs"
          placeholder="Filtrar…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      {tooBig && (
        <div className="border-b border-border bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-300" data-testid="preload-too-big">
          Los que se cargan al arrancar ocupan unos {gb(loadBytes)} y la GPU tiene {gpuMemoryGb} GB: no caben juntos.
          Pasa alguno a «solo descargar» (se carga cuando se pida); si no, se irán cargando y sacando de uno en uno.
        </div>
      )}
      <div className="max-h-72 overflow-y-auto">
        {!ready ? (
          <div className="flex items-center gap-2 px-3 py-3 text-sm text-muted-foreground">
            <Loader2 size={14} className="animate-spin" /> Cargando modelos…
          </div>
        ) : (
          <>
            {recipeRows.length > 0 && (
              <>
                <div className="flex items-center gap-1.5 bg-muted/40 px-3 py-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  <History size={11} /> Recientes en pods
                </div>
                {recipeRows.map(renderRow)}
              </>
            )}
            <button
              type="button"
              className="flex w-full items-center gap-1.5 bg-muted/40 px-3 py-1 text-left text-[11px] font-medium uppercase tracking-wide text-muted-foreground hover:bg-muted"
              onClick={() => setSearching((v) => !v)}
              aria-expanded={searching}
            >
              <Search size={11} /> Buscar en Hugging Face u Ollama
            </button>
            {searching && <RecipeSearch onAdded={addRecipe} />}
            <div className="flex items-center gap-1.5 bg-muted/40 px-3 py-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              <Server size={11} /> De este nodo
            </div>
            {modelRows.length === 0 ? (
              <div className="px-3 py-2 text-sm text-muted-foreground">No hay modelos que llevar.</div>
            ) : (
              modelRows.map(renderRow)
            )}
          </>
        )}
      </div>
    </div>
  )
}
