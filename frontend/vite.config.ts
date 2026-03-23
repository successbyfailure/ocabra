import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig({
  plugins: [react()],
  cacheDir: ".vite",
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
