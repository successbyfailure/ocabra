import { create } from "zustand"
import type { ServiceState } from "@/types"
import { api } from "@/api/client"

interface ServiceStore {
  services: Record<string, ServiceState>
  setServices: (services: ServiceState[]) => void
  updateService: (serviceId: string, patch: Partial<ServiceState>) => void
  unloadService: (serviceId: string) => Promise<void>
  startService: (serviceId: string) => Promise<void>
  refreshService: (serviceId: string) => Promise<void>
}

export const useServiceStore = create<ServiceStore>((set) => ({
  services: {},
  setServices: (services) =>
    set({
      services: Object.fromEntries(services.map((s) => [s.serviceId, s])),
    }),
  updateService: (serviceId, patch) =>
    set((state) => ({
      services: {
        ...state.services,
        [serviceId]: state.services[serviceId]
          ? { ...state.services[serviceId], ...patch }
          : (patch as ServiceState),
      },
    })),
  unloadService: async (serviceId) => {
    const updated = await api.services.unload(serviceId)
    set((state) => ({
      services: { ...state.services, [serviceId]: updated },
    }))
  },
  startService: async (serviceId) => {
    const updated = await api.services.start(serviceId)
    set((state) => ({
      services: { ...state.services, [serviceId]: updated },
    }))
  },
  refreshService: async (serviceId) => {
    const updated = await api.services.refresh(serviceId)
    set((state) => ({
      services: { ...state.services, [serviceId]: updated },
    }))
  },
}))
