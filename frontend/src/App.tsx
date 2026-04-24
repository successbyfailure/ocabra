import { Routes, Route, Navigate } from "react-router-dom"
import { useAuth } from "@/hooks/useAuth"
import { Layout } from "@/components/layout/Layout"
import { ProtectedRoute } from "@/components/auth/ProtectedRoute"
import { Login } from "@/pages/Login"
import { Dashboard } from "@/pages/Dashboard"
import { Models } from "@/pages/Models"
import { Explore } from "@/pages/Explore"
import { Playground } from "@/pages/Playground"
import { Stats } from "@/pages/Stats"
import { Settings } from "@/pages/Settings"
import { TrtllmEngines } from "@/pages/TrtllmEngines"
import { Backends } from "@/pages/Backends"
import { Users } from "@/pages/Users"
import { Groups } from "@/pages/Groups"
import { ApiKeys } from "@/pages/ApiKeys"
import { Logs } from "@/pages/Logs"
import { Agents } from "@/pages/Agents"
import { MCPServers } from "@/pages/MCPServers"

// ROUTES — Each stream adds its page component here. Do not remove this comment.
export default function App() {
  // Bootstrap auth state once at the application root
  useAuth()

  return (
    <Routes>
      {/* Public route */}
      <Route path="/login" element={<Login />} />

      {/* All protected routes share the app layout */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <Layout>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/models" element={<Models />} />
                <Route path="/playground" element={<Playground />} />
                <Route path="/stats" element={<Stats />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/api-keys" element={<ApiKeys />} />
                <Route path="/logs" element={<Logs />} />

                {/* model_manager+ only */}
                <Route
                  path="/explore"
                  element={
                    <ProtectedRoute minRole="model_manager">
                      <Explore />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/engines"
                  element={
                    <ProtectedRoute minRole="model_manager">
                      <TrtllmEngines />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/backends"
                  element={
                    <ProtectedRoute minRole="system_admin">
                      <Backends />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/agents"
                  element={
                    <ProtectedRoute minRole="model_manager">
                      <Agents />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/mcp-servers"
                  element={
                    <ProtectedRoute minRole="model_manager">
                      <MCPServers />
                    </ProtectedRoute>
                  }
                />

                {/* system_admin only */}
                <Route
                  path="/users"
                  element={
                    <ProtectedRoute minRole="system_admin">
                      <Users />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/groups"
                  element={
                    <ProtectedRoute minRole="system_admin">
                      <Groups />
                    </ProtectedRoute>
                  }
                />
              </Routes>
            </Layout>
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}
