import js from "@eslint/js"
import globals from "globals"

export default [
  {
    ignores: [
      "dist",
      "node_modules",
      "test-results",
      "playwright-report",
      ".vite",
      ".vitest-cache",
      // Until TS ESLint parser/plugins can be installed in this env,
      // avoid failing lint for TS sources.
      "src/**/*.{ts,tsx}",
      "e2e/**/*.ts",
      "playwright.config.ts",
      "vitest.config.ts",
    ],
  },
  js.configs.recommended,
  {
    files: ["**/*.{js,mjs,cjs}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
  },
]
