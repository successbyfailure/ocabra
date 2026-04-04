import { Navigate } from "react-router-dom"
import { useAuthStore } from "@/stores/authStore"
import type { UserRole } from "@/types"
import type { ReactNode } from "react"

interface Props {
  minRole?: UserRole
  children: ReactNode
}

export function ProtectedRoute({ minRole, children }: Props) {
  const user = useAuthStore((s) => s.user)
  const isLoading = useAuthStore((s) => s.isLoading)
  const hasRole = useAuthStore((s) => s.hasRole)

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  if (minRole && !hasRole(minRole)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-background">
        <div className="text-4xl font-bold text-destructive">403</div>
        <p className="text-lg text-muted-foreground">Acceso denegado</p>
        <p className="text-sm text-muted-foreground">
          Necesitas el rol <span className="font-semibold text-foreground">{minRole}</span> para acceder a esta
          sección.
        </p>
        <a href="/dashboard" className="mt-2 text-sm text-primary underline underline-offset-4">
          Volver al dashboard
        </a>
      </div>
    )
  }

  return <>{children}</>
}
