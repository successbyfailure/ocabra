import { ApiKeyManager } from "@/components/auth/ApiKeyManager"

export function ApiKeys() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">API Keys</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Gestiona tus claves de acceso para la API OpenAI y Ollama compatibles.
          Las claves tienen el formato <code className="text-xs bg-muted px-1 py-0.5 rounded">sk-ocabra-…</code>
        </p>
      </div>
      <ApiKeyManager />
    </div>
  )
}
