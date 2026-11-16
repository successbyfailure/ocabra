import React from "react"
import ReactDOM from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import { Toaster } from "sonner"
import App from "./App"
import { installRemoteRouting } from "./lib/remoteNode"
import "./index.css"

// Before anything renders: in remote mode every request goes to that node.
installRemoteRouting()

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <Toaster theme="dark" position="bottom-right" richColors />
    </BrowserRouter>
  </React.StrictMode>,
)
