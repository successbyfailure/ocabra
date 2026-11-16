// Types for modular backend management (Bloque 15).
// See docs/tasks/modular-backends-plan.md for the API contract.

export type BackendInstallStatus =
  | "not_installed"
  | "installing"
  | "installed"
  | "uninstalling"
  | "error"
  | "built-in"

export type BackendInstallMethod = "auto" | "oci" | "source"

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
  /** Prebuilt installs: image and digest it came from. */
  ociRef: string | null
  ociDigest: string | null
  /** Installed versions of the packages the spec names. */
  packages: Record<string, string>
  declaredPackages: string[]
  gitRef: string | null
}

/** What the OCI repository publishes for a backend. */
export interface BackendPublished {
  ref: string
  digest: string
  sizeBytes: number
  version: string | null
  packages: Record<string, string>
  gpuArchs: string | null
  created: string | null
}

export interface BackendPublishedInfo {
  published: BackendPublished | null
  updateAvailable: boolean
}
