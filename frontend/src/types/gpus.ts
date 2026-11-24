// Tipos del dominio «gpus» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

export interface GPUProcessInfo {
  pid: number;
  processName: string | null;
  processType: "compute" | "graphics";
  usedVramMb: number;
}

export interface GPUState {
  index: number;
  name: string;
  totalVramMb: number;
  freeVramMb: number;
  usedVramMb: number;
  utilizationPct: number;
  temperatureC: number;
  powerDrawW: number;
  powerLimitW: number;
  lockedVramMb: number;
  processes: GPUProcessInfo[];
  /** Solo en el editor de modelos: el hueco de esta GPU para el modelo que se
   *  edita, contado por el servidor como lo cuenta el planificador. */
  editorCapacity?: GpuCapacity;
}

export interface GpuCapacity {
  /** Lo que tendría ahora mismo (su propia copia liberada si está cargado). */
  availableMb: number;
  /** Lo que tendría desalojando todos los demás modelos de oCabra. */
  budgetMb: number;
  /** Tarjeta menos budget: driver, escritorio, margen del planificador. */
  systemMb: number;
}
