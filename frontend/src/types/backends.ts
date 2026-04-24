// Types for modular backend management (Bloque 15).
// See docs/tasks/modular-backends-plan.md for the API contract.

export type BackendInstallStatus =
  | "not_installed"
  | "installing"
  | "installed"
  | "uninstalling"
  | "error"
  | "built-in"

export type BackendInstallMethod = "oci" | "source"

export type BackendInstallSource = "oci" | "source" | "built-in" | null

export interface BackendModuleState {
  backendType: string
  displayName: string
  description: string
  tags: string[]
  installStatus: BackendInstallStatus
  installedVersion: string | null
  installedAt: string | null
  installSource: BackendInstallSource
  estimatedSizeMb: number
  actualSizeMb: number | null
  modelsLoaded: number
  hasUpdate: boolean
  installProgress: number | null
  installDetail: string | null
  error: string | null
  alwaysAvailable: boolean
}
