import { create } from "zustand"
import type { AuthUser, UserRole } from "@/types"

const ROLE_HIERARCHY: Record<UserRole, number> = {
  user: 0,
  model_manager: 1,
  system_admin: 2,
}

interface AuthState {
  user: AuthUser | null
  isLoading: boolean
  setUser: (user: AuthUser | null) => void
  setLoading: (loading: boolean) => void
  logout: () => void
  hasRole: (minRole: UserRole) => boolean
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  isLoading: true,

  setUser: (user) => set({ user }),

  setLoading: (loading) => set({ isLoading: loading }),

  logout: () => set({ user: null }),

  hasRole: (minRole) => {
    const { user } = get()
    if (!user) return false
    return ROLE_HIERARCHY[user.role] >= ROLE_HIERARCHY[minRole]
  },
}))
