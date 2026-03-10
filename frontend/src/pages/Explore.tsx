import { useEffect, useMemo, useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import * as Tabs from "@radix-ui/react-tabs"
import { toast } from "sonner"
import { api } from "@/api/client"
import { DownloadQueue } from "@/components/downloads/DownloadQueue"
import { HFModelCard } from "@/components/explore/HFModelCard"
import { OllamaModelCard } from "@/components/explore/OllamaModelCard"
import { SearchFilters } from "@/components/explore/SearchFilters"
import { useDownloadStore } from "@/stores/downloadStore"
import type { DownloadJob, DownloadSource, HFModelCard as HFCardType, OllamaModelCard as OllamaCardType } from "@/types"

interface InstallTarget {
  source: DownloadSource
  modelRef: string
  title: string
}

export function Explore() {
  const [activeTab, setActiveTab] = useState<"hf" | "ollama">("hf")
  const [query, setQuery] = useState("mistral")
  const [debouncedQuery, setDebouncedQuery] = useState(query)
  const [taskFilter, setTaskFilter] = useState("")
  const [sizeFilter, setSizeFilter] = useState("")
  const [gatedFilter, setGatedFilter] = useState("")
  const [hfResults, setHfResults] = useState<HFCardType[]>([])
  const [ollamaResults, setOllamaResults] = useState<OllamaCardType[]>([])
  const [loading, setLoading] = useState(false)
  const [installTarget, setInstallTarget] = useState<InstallTarget | null>(null)
  const [targetDir, setTargetDir] = useState("/models")
  const [loadPolicy, setLoadPolicy] = useState("on_demand")

  const jobs = useDownloadStore((state) => state.jobs)
  const setJobs = useDownloadStore((state) => state.setJobs)
  const addJob = useDownloadStore((state) => state.addJob)

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(query)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [query])

  const refreshJobs = async () => {
    const list = await api.downloads.list()
    setJobs(list)
  }

  useEffect(() => {
    void refreshJobs().catch(() => {
      // silent bootstrap failure
    })
    const timer = window.setInterval(() => {
      void refreshJobs().catch(() => {
        // silent background refresh failure
      })
    }, 30_000)
    return () => window.clearInterval(timer)
  }, [setJobs])

  useEffect(() => {
    const q = debouncedQuery.trim()
    if (!q) {
      setHfResults([])
      setOllamaResults([])
      return
    }

    let active = true
    setLoading(true)

    const run = async () => {
      try {
        if (activeTab === "hf") {
          const data = await api.registry.searchHF(q, taskFilter || undefined, 30)
          if (!active) return
          let next = data
          if (sizeFilter === "small") next = next.filter((item) => (item.sizeGb ?? 0) < 4)
          if (sizeFilter === "medium") next = next.filter((item) => (item.sizeGb ?? 0) >= 4 && (item.sizeGb ?? 0) <= 12)
          if (sizeFilter === "large") next = next.filter((item) => (item.sizeGb ?? 0) > 12)
          if (gatedFilter !== "") next = next.filter((item) => String(item.gated) === gatedFilter)
          setHfResults(next)
        } else {
          const data = await api.registry.searchOllama(q)
          if (!active) return
          setOllamaResults(data)
        }
      } catch (err) {
        if (active) {
          toast.error(err instanceof Error ? err.message : "No se pudo buscar")
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    void run()

    return () => {
      active = false
    }
  }, [activeTab, debouncedQuery, gatedFilter, sizeFilter, taskFilter])

  const activeDownloads = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "downloading"),
    [jobs],
  )

  const install = async () => {
    if (!installTarget) return
    try {
      const job = await api.downloads.enqueue(installTarget.source, installTarget.modelRef)
      addJob(job)
      toast.success(`Descarga iniciada en ${targetDir} (${loadPolicy})`)
      setInstallTarget(null)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo iniciar descarga")
    }
  }

  const cancelJob = async (job: DownloadJob) => {
    try {
      await api.downloads.cancel(job.jobId)
      setJobs(jobs.map((item) => (item.jobId === job.jobId ? { ...item, status: "cancelled" } : item)))
      toast.success("Descarga cancelada")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "No se pudo cancelar")
    }
  }

  return (
    <div className="space-y-5 pb-32">
      <div>
        <h1 className="text-2xl font-semibold">Explore</h1>
        <p className="text-muted-foreground">Buscar modelos en HuggingFace y Ollama.</p>
      </div>

      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Buscar modelo"
        className="w-full rounded-lg border border-border bg-card px-3 py-2"
      />

      <Tabs.Root value={activeTab} onValueChange={(value) => setActiveTab(value as "hf" | "ollama")}>
        <Tabs.List className="inline-flex rounded-lg border border-border bg-card p-1">
          <Tabs.Trigger
            value="hf"
            className="rounded-md px-3 py-1.5 text-sm text-muted-foreground data-[state=active]:bg-muted data-[state=active]:text-foreground"
          >
            HuggingFace
          </Tabs.Trigger>
          <Tabs.Trigger
            value="ollama"
            className="rounded-md px-3 py-1.5 text-sm text-muted-foreground data-[state=active]:bg-muted data-[state=active]:text-foreground"
          >
            Ollama
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="hf" className="mt-3 space-y-3">
          <SearchFilters
            task={taskFilter}
            size={sizeFilter}
            gated={gatedFilter}
            onTaskChange={setTaskFilter}
            onSizeChange={setSizeFilter}
            onGatedChange={setGatedFilter}
          />
          {loading ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={`hf-skeleton-${idx}`} className="h-32 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {hfResults.map((model) => (
                <HFModelCard
                  key={model.repoId}
                  model={model}
                  onInstall={(item) =>
                    setInstallTarget({
                      source: "huggingface",
                      modelRef: item.repoId,
                      title: item.modelName,
                    })
                  }
                />
              ))}
            </div>
          )}
        </Tabs.Content>

        <Tabs.Content value="ollama" className="mt-3">
          {loading ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Array.from({ length: 6 }).map((_, idx) => (
                <div key={`ollama-skeleton-${idx}`} className="h-32 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {ollamaResults.map((model) => (
                <OllamaModelCard
                  key={model.name}
                  model={model}
                  onInstall={(item) =>
                    setInstallTarget({
                      source: "ollama",
                      modelRef: item.name,
                      title: item.name,
                    })
                  }
                />
              ))}
            </div>
          )}
        </Tabs.Content>
      </Tabs.Root>

      {activeDownloads.length > 0 && <p className="text-xs text-muted-foreground">Descargas activas: {activeDownloads.length}</p>}

      <Dialog.Root open={Boolean(installTarget)} onOpenChange={(next) => !next && setInstallTarget(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5">
            <Dialog.Title className="text-lg font-semibold">Instalar modelo</Dialog.Title>
            <Dialog.Description className="mt-1 text-sm text-muted-foreground">
              {installTarget?.title}
            </Dialog.Description>

            <div className="mt-4 space-y-3">
              <label className="block text-sm text-muted-foreground">
                Carpeta destino
                <input
                  value={targetDir}
                  onChange={(event) => setTargetDir(event.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                />
              </label>
              <label className="block text-sm text-muted-foreground">
                load_policy
                <select
                  value={loadPolicy}
                  onChange={(event) => setLoadPolicy(event.target.value)}
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2"
                >
                  <option value="on_demand">on_demand</option>
                  <option value="warm">warm</option>
                  <option value="pin">pin</option>
                </select>
              </label>
            </div>

            <div className="mt-4 flex justify-end gap-2">
              <Dialog.Close className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted">
                Cancelar
              </Dialog.Close>
              <button
                type="button"
                onClick={() => void install()}
                className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
              >
                Iniciar descarga
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <DownloadQueue jobs={jobs} onCancel={cancelJob} />
    </div>
  )
}
