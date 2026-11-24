// Tipos del dominio «services» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

export type ServiceStatus =
  | "unknown"
  | "idle"
  | "active"
  | "unreachable"
  | "disabled"
  | "building"
  | "starting";

export interface ServiceState {
  serviceId: string;
  serviceType: string;
  displayName: string;
  uiUrl: string;
  preferredGpu: number | null;
  idleUnloadAfterSeconds: number;
  enabled: boolean;
  serviceAlive: boolean;
  runtimeLoaded: boolean;
  status: ServiceStatus;
  activeModelRef: string | null;
  lastActivityAt: string | null;
  lastHealthCheckAt: string | null;
  detail: string | null;
  // Generation metrics
  isGenerating: boolean;
  queueDepth: number;
  vramUsedMb: number | null;
  gpuUtilPct: number | null;
  // Container resource metrics
  cpuPct: number | null;
  memUsedMb: number | null;
  memLimitMb: number | null;
}
