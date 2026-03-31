import { Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/layout/Layout"
import { Dashboard } from "@/pages/Dashboard"
import { Models } from "@/pages/Models"
import { Explore } from "@/pages/Explore"
import { Playground } from "@/pages/Playground"
import { Stats } from "@/pages/Stats"
import { Settings } from "@/pages/Settings"
import { TrtllmEngines } from "@/pages/TrtllmEngines"

// ROUTES — Each stream adds its page component here. Do not remove this comment.
export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/models" element={<Models />} />
        <Route path="/engines" element={<TrtllmEngines />} />
        <Route path="/explore" element={<Explore />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/stats" element={<Stats />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}
