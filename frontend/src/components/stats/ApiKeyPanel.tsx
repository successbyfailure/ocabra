import type { ByApiKeyStats } from "@/types"

interface ApiKeyPanelProps {
  data: ByApiKeyStats
}

export function ApiKeyPanel({ data }: ApiKeyPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Uso por API Key</h3>
      {data.byApiKey.length === 0 ? (
        <p className="text-sm text-muted-foreground py-4 text-center">Sin datos de API keys</p>
      ) : (
        <div className="overflow-auto">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">API Key</th>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground">Usuario</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Requests</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Errores</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens in</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Tokens out</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Energia (Wh)</th>
                <th className="px-3 py-2 text-right font-medium text-muted-foreground">Coste ref.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {data.byApiKey.map((row) => (
                <tr key={row.apiKeyName} className="hover:bg-muted/30">
                  <td className="px-3 py-2 font-mono text-xs">{row.apiKeyName}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.username ?? "—"}</td>
                  <td className="px-3 py-2 text-right">{row.totalRequests}</td>
                  <td className="px-3 py-2 text-right">{row.totalErrors}</td>
                  <td className="px-3 py-2 text-right">{row.totalInputTokens.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">{row.totalOutputTokens.toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">{row.totalEnergyWh.toFixed(1)}</td>
                  <td className="px-3 py-2 text-right text-orange-400">${row.estimatedCostUsd.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
