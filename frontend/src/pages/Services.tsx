import { useEffect, useMemo } from "react"
import { RefreshCw, Wand2 } from "lucide-react"
import { api } from "@/api/client"
import { EmptyState } from "@/components/common/EmptyState"
import { ServiceCard } from "@/components/services/ServiceCard"
import { useServiceStore } from "@/stores/serviceStore"

/* Página dedicada para servicios de generación interactivos (ComfyUI,
 * A1111, ACE-Step, TRELLIS.2, …). Antes vivían dentro del Dashboard mezclados
 * con GPUs y modelos LLM, lo que ensuciaba la pantalla principal. */
export function Services() {
  const services = useServiceStore((s) => s.services)
  const setServices = useServiceStore((s) => s.setServices)

  const serviceList = useMemo(() => Object.values(services), [services])
  const activeCount = useMemo(
    () => serviceList.filter((s) => s.enabled && s.serviceAlive).length,
    [serviceList],
  )
  const generatingCount = useMemo(
    () => serviceList.filter((s) => s.isGenerating).length,
    [serviceList],
  )

  const refresh = async () => {
    try {
      const list = await api.services.list()
      setServices(list)
    } catch (err) {
      // keep stale state; the badge just shows the old count until next poll
      console.warn("services refresh failed", err)
    }
  }

  useEffect(() => {
    void refresh()
    const t = window.setInterval(() => void refresh(), 10_000)
    return () => window.clearInterval(t)
  }, [])

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h1 className="flex items-center gap-2 text-xl font-semibold">
            <Wand2 size={18} />
            Services
          </h1>
          <p className="text-sm text-muted-foreground">
            Servicios de generación interactivos (ComfyUI, A1111, ACE-Step, TRELLIS.2, …).
            Gestionan su propio runtime y VRAM fuera del scheduler de modelos LLM.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
        >
          <RefreshCw size={14} />
          Actualizar
        </button>
      </div>

      {serviceList.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full border border-border bg-muted/40 px-2 py-0.5">
            {serviceList.length} total
          </span>
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-200">
            {activeCount} online
          </span>
          {generatingCount > 0 && (
            <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-amber-200">
              {generatingCount} generando
            </span>
          )}
        </div>
      )}

      <div className="space-y-3">
        {serviceList.map((service) => (
          <ServiceCard key={service.serviceId} service={service} />
        ))}
        {serviceList.length === 0 && (
          <EmptyState
            title="Sin servicios configurados"
            description="Los servicios se definen en docker-compose.yml y se registran al arrancar oCabra."
          />
        )}
      </div>
    </div>
  )
}
