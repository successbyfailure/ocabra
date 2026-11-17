import js from "@eslint/js"
import reactHooks from "eslint-plugin-react-hooks"
import globals from "globals"
import tseslint from "typescript-eslint"

// Until 2026-11-17 this ignored every .ts/.tsx file ("until TS ESLint can be
// installed"), so `npm run lint` passed while checking nothing. tsc covers the
// types; this covers what tsc cannot see, chiefly the rules of hooks.
export default tseslint.config(
  {
    ignores: [
      "dist",
      "node_modules",
      "test-results",
      "playwright-report",
      ".vite",
      ".vitest-cache",
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
  {
    files: ["**/*.{ts,tsx}"],
    extends: [tseslint.configs.recommended],
    plugins: { "react-hooks": reactHooks },
    languageOptions: {
      ecmaVersion: 2022,
      globals: { ...globals.browser },
    },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      // `_name` marks an argument kept on purpose (a callback's signature).
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
      ],
    },
  },
)
