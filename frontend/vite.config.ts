import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"
import { readFileSync } from "node:fs"

const pkg = JSON.parse(readFileSync(path.resolve(__dirname, "package.json"), "utf-8"))

export default defineConfig({
  plugins: [react()],
  cacheDir: ".vite",
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://api:8000",
      "/api": "http://api:8000",
      "/ocabra": { target: "http://api:8000", ws: true },
      "/health": "http://api:8000",
    },
  },
})
