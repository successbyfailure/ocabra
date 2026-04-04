import { useEffect } from "react"
import { api } from "@/api/client"
import { useAuthStore } from "@/stores/authStore"
import type { AuthUser, UserRole } from "@/types"

/**
 * Bootstraps auth state by calling GET /ocabra/auth/me on mount.
 * Should be called once at the application root.
 */
export function useAuth(): void {
  const setUser = useAuthStore((s) => s.setUser)
  const setLoading = useAuthStore((s) => s.setLoading)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api.auth
      .me()
      .then((user) => {
        if (!cancelled) setUser(user)
      })
      .catch(() => {
        if (!cancelled) setUser(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [setUser, setLoading])
}

/** Returns the current authenticated user or null. */
export function useCurrentUser(): AuthUser | null {
  return useAuthStore((s) => s.user)
}

/**
 * Returns true if the current user has at least `minRole`.
 * Returns false if the user is not authenticated.
 */
export function useRequireRole(minRole: UserRole): boolean {
  return useAuthStore((s) => s.hasRole(minRole))
}

/** Returns true if the current user is a system_admin. */
export function useIsAdmin(): boolean {
  return useAuthStore((s) => s.hasRole("system_admin"))
}

/** Returns true if the current user has model_manager role or higher. */
export function useIsModelManager(): boolean {
  return useAuthStore((s) => s.hasRole("model_manager"))
}
