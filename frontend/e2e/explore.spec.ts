import type { Page } from "@playwright/test"
import { expect, test } from "@playwright/test"

async function mockDownloads(page: Page) {
  await page.route("**/ocabra/downloads", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: [] })
      return
    }
    if (route.request().method() === "POST") {
      await route.fulfill({
        json: {
          job_id: "job-1",
          source: "huggingface",
          model_ref: "Qwen/Qwen3-8B-Instruct",
          status: "queued",
          progress_pct: 0,
          speed_mb_s: null,
          eta_seconds: null,
          error: null,
          started_at: "2026-03-14T00:00:00Z",
          completed_at: null,
        },
      })
      return
    }
    await route.fallback()
  })
}

async function mockHFSearch(
  page: Page,
  payload: unknown,
) {
  await page.route("**/ocabra/registry/hf/search**", async (route) => {
    await route.fulfill({ json: payload })
  })
}

async function mockHFVariants(
  page: Page,
  repoId: string,
  payload: unknown,
) {
  await page.route(`**/ocabra/registry/hf/${encodeURIComponent(repoId)}/variants`, async (route) => {
    await route.fulfill({ json: payload })
  })
}

test("explore install modal shows recipe and probe guidance", async ({ page }) => {
  await mockDownloads(page)
  await mockHFSearch(page, [
    {
      repo_id: "Qwen/Qwen3-8B-Instruct",
      model_name: "Qwen3-8B-Instruct",
      task: "text-generation",
      downloads: 123,
      likes: 45,
      size_gb: 15.2,
      tags: ["chat"],
      gated: false,
      suggested_backend: "vllm",
      compatibility: "warning",
      compatibility_reason: "Compatibilidad prevista via Transformers backend.",
      vllm_support: {
        classification: "transformers_backend",
        label: "transformers backend",
        model_impl: "transformers",
        runner: "generate",
        task_mode: "generate",
        required_overrides: ["chat_template", "tool_call_parser"],
        recipe_id: "qwen3",
        recipe_notes: ["Qwen3 necesita parser de tools y reasoning."],
        recipe_model_impl: "vllm",
        recipe_runner: "generate",
        suggested_config: { tool_call_parser: "qwen3_json", reasoning_parser: "qwen3" },
        suggested_tuning: { gpu_memory_utilization: 0.9 },
        runtime_probe: {
          status: "supported_transformers_backend",
          reason: "Compatibilidad verificada via Transformers backend.",
          recommended_model_impl: "transformers",
          recommended_runner: "generate",
          tokenizer_load: true,
          config_load: true,
          observed_at: "2026-03-14T12:00:00Z",
        },
      },
    },
  ])

  await mockHFVariants(page, "Qwen/Qwen3-8B-Instruct", [
    {
      variant_id: "standard",
      label: "standard",
      artifact: null,
      size_gb: 15.2,
      format: "safetensors",
      quantization: null,
      backend_type: "vllm",
      is_default: true,
      installable: true,
      compatibility: "warning",
      compatibility_reason: "Compatibilidad prevista via Transformers backend.",
      vllm_support: {
        classification: "transformers_backend",
        label: "transformers backend",
        model_impl: "transformers",
        runner: "generate",
        task_mode: "generate",
        required_overrides: ["chat_template", "tool_call_parser"],
        recipe_id: "qwen3",
        recipe_notes: ["Qwen3 necesita parser de tools y reasoning."],
        recipe_model_impl: "vllm",
        recipe_runner: "generate",
        suggested_config: { tool_call_parser: "qwen3_json", reasoning_parser: "qwen3" },
        suggested_tuning: { gpu_memory_utilization: 0.9 },
        runtime_probe: {
          status: "supported_transformers_backend",
          reason: "Compatibilidad verificada via Transformers backend.",
          recommended_model_impl: "transformers",
          recommended_runner: "generate",
          tokenizer_load: true,
          config_load: true,
          observed_at: "2026-03-14T12:00:00Z",
        },
      },
    },
  ])

  await page.goto("/explore")
  await page.getByPlaceholder("Buscar modelo").fill("Qwen3")
  await expect(page.getByRole("heading", { name: "Qwen3-8B-Instruct" })).toBeVisible()

  await page.getByRole("button", { name: "Instalar" }).click()
  await expect(page.getByText("Instalar modelo")).toBeVisible()
  await expect(page.getByText(/Recomendacion verificada por probe/)).toBeVisible()
  await expect(page.getByText(/La preconfiguracion automatica usara la recomendacion verificada por probe/)).toBeVisible()
  await expect(page.getByText(/La recomendacion final del probe difiere de la recipe base/)).toBeVisible()
  await expect(page.getByText(/recipe_model_impl=vllm/)).toBeVisible()

  await page.getByRole("button", { name: "Iniciar descarga" }).click()
  await expect(page.getByText(/Descarga iniciada/i)).toBeVisible()
})

test("explore install modal shows actionable probe override hints", async ({ page }) => {
  await mockDownloads(page)
  await mockHFSearch(page, [
    {
      repo_id: "Qwen/Qwen3-Tools-8B",
      model_name: "Qwen3-Tools-8B",
      task: "text-generation",
      downloads: 55,
      likes: 12,
      size_gb: 14.8,
      tags: ["chat", "tools"],
      gated: false,
      suggested_backend: "vllm",
      compatibility: "warning",
      compatibility_reason: "Falta parser para tool calling.",
      vllm_support: {
        classification: "native_vllm",
        label: "native vllm",
        model_impl: "vllm",
        runner: "generate",
        task_mode: "generate",
        required_overrides: ["tool_call_parser"],
        recipe_id: "qwen3",
        recipe_notes: ["Qwen3 necesita parser específico para tools."],
        recipe_model_impl: "vllm",
        recipe_runner: "generate",
        suggested_config: { tool_call_parser: "qwen3_json" },
        suggested_tuning: {},
        runtime_probe: {
          status: "missing_tool_parser",
          reason: "Automatic tool choice requires parser.",
          recommended_model_impl: "vllm",
          recommended_runner: "generate",
          tokenizer_load: true,
          config_load: true,
          observed_at: "2026-03-14T12:10:00Z",
        },
      },
    },
  ])

  await mockHFVariants(page, "Qwen/Qwen3-Tools-8B", [
    {
      variant_id: "standard",
      label: "standard",
      artifact: null,
      size_gb: 14.8,
      format: "safetensors",
      quantization: null,
      backend_type: "vllm",
      is_default: true,
      installable: true,
      compatibility: "warning",
      compatibility_reason: "Falta parser para tool calling.",
      vllm_support: {
        classification: "native_vllm",
        label: "native vllm",
        model_impl: "vllm",
        runner: "generate",
        task_mode: "generate",
        required_overrides: ["tool_call_parser"],
        recipe_id: "qwen3",
        recipe_notes: ["Qwen3 necesita parser específico para tools."],
        recipe_model_impl: "vllm",
        recipe_runner: "generate",
        suggested_config: { tool_call_parser: "qwen3_json" },
        suggested_tuning: {},
        runtime_probe: {
          status: "missing_tool_parser",
          reason: "Automatic tool choice requires parser.",
          recommended_model_impl: "vllm",
          recommended_runner: "generate",
          tokenizer_load: true,
          config_load: true,
          observed_at: "2026-03-14T12:10:00Z",
        },
      },
    },
  ])

  await page.goto("/explore")
  await page.getByPlaceholder("Buscar modelo").fill("Qwen3 Tools")
  await expect(page.getByRole("heading", { name: "Qwen3-Tools-8B" })).toBeVisible()

  await page.getByRole("button", { name: "Instalar" }).click()
  await expect(page.getByText("Instalar modelo")).toBeVisible()
  await expect.poll(async () => await page.locator("body").textContent()).toContain(
    "override sugerido: tool_call_parser",
  )
  await expect.poll(async () => await page.locator("body").textContent()).toContain(
    "Automatic tool choice requires parser.",
  )
})
