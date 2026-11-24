import { lazy, Suspense } from "react"
import { Routes, Route, Navigate } from "react-router"
import { Loader2 } from "lucide-react"
import { useAuth } from "@/hooks/useAuth"
import { Layout } from "@/components/layout/Layout"
import { ProtectedRoute } from "@/components/auth/ProtectedRoute"
import { Login } from "@/pages/Login"
import { Dashboard } from "@/pages/Dashboard"

// REF-18 — code splitting por ruta. Login y Dashboard (la primera pantalla
// tras entrar) van en el bundle inicial; el resto se carga bajo demanda con
// React.lazy, así el arranque no arrastra el editor de modelos, el playground,
// estadísticas, federación, etc. Las páginas exportan con nombre, de ahí el
// ``.then(m => ({ default: ... }))``.
const Models = lazy(() => import("@/pages/Models").then((m) => ({ default: m.Models })))
const ModelEditor = lazy(() => import("@/pages/ModelEditor").then((m) => ({ default: m.ModelEditor })))
const Explore = lazy(() => import("@/pages/Explore").then((m) => ({ default: m.Explore })))
const Playground = lazy(() => import("@/pages/Playground").then((m) => ({ default: m.Playground })))
const Stats = lazy(() => import("@/pages/Stats").then((m) => ({ default: m.Stats })))
const Settings = lazy(() => import("@/pages/Settings").then((m) => ({ default: m.Settings })))
const TrtllmEngines = lazy(() => import("@/pages/TrtllmEngines").then((m) => ({ default: m.TrtllmEngines })))
const Backends = lazy(() => import("@/pages/Backends").then((m) => ({ default: m.Backends })))
const Users = lazy(() => import("@/pages/Users").then((m) => ({ default: m.Users })))
const Groups = lazy(() => import("@/pages/Groups").then((m) => ({ default: m.Groups })))
const ApiKeys = lazy(() => import("@/pages/ApiKeys").then((m) => ({ default: m.ApiKeys })))
const Logs = lazy(() => import("@/pages/Logs").then((m) => ({ default: m.Logs })))
const Agents = lazy(() => import("@/pages/Agents").then((m) => ({ default: m.Agents })))
const MCPServers = lazy(() => import("@/pages/MCPServers").then((m) => ({ default: m.MCPServers })))
const Sessions = lazy(() => import("@/pages/Sessions").then((m) => ({ default: m.Sessions })))
const Routers = lazy(() => import("@/pages/Routers").then((m) => ({ default: m.Routers })))
const OpenAICompat = lazy(() => import("@/pages/OpenAICompat").then((m) => ({ default: m.OpenAICompat })))
const Profiles = lazy(() => import("@/pages/Profiles").then((m) => ({ default: m.Profiles })))
const Services = lazy(() => import("@/pages/Services").then((m) => ({ default: m.Services })))
const Batches = lazy(() => import("@/pages/Batches").then((m) => ({ default: m.Batches })))
const Benchmarks = lazy(() => import("@/pages/Benchmarks").then((m) => ({ default: m.Benchmarks })))
const Federation = lazy(() => import("@/pages/Federation").then((m) => ({ default: m.Federation })))

function RouteFallback() {
  return (
    <div className="flex items-center gap-2 p-6 text-muted-foreground">
      <Loader2 size={16} className="animate-spin" /> Cargando…
    </div>
  )
}

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
              <Suspense fallback={<RouteFallback />}>
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/models" element={<Models />} />
                  <Route path="/models/edit/*" element={<ModelEditor />} />
                  <Route path="/services" element={<Services />} />
                  <Route path="/playground" element={<Playground />} />
                  <Route path="/stats" element={<Stats />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="/api-keys" element={<ApiKeys />} />
                  <Route path="/logs" element={<Logs />} />

                  {/* model_manager+ only */}
                  <Route
                    path="/batches"
                    element={
                      <ProtectedRoute minRole="model_manager">
                        <Batches />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/benchmarks"
                    element={
                      <ProtectedRoute minRole="model_manager">
                        <Benchmarks />
                      </ProtectedRoute>
                    }
                  />
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

                  <Route
                    path="/sessions"
                    element={
                      <ProtectedRoute minRole="system_admin">
                        <Sessions />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/profiles"
                    element={
                      <ProtectedRoute minRole="model_manager">
                        <Profiles />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/routers"
                    element={
                      <ProtectedRoute minRole="model_manager">
                        <Routers />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/openai-compat"
                    element={
                      <ProtectedRoute minRole="model_manager">
                        <OpenAICompat />
                      </ProtectedRoute>
                    }
                  />

                  {/* system_admin only */}
                  <Route
                    path="/federation"
                    element={
                      <ProtectedRoute minRole="system_admin">
                        <Federation />
                      </ProtectedRoute>
                    }
                  />
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
              </Suspense>
            </Layout>
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}
