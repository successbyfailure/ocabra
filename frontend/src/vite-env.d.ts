/// <reference types="vite/client" />

// Injected by vite.config.ts via `define`, sourced from package.json's
// "version" so the sidebar footer can never drift from the real build
// version again (it used to be a hardcoded "v0.5.0" string).
declare const __APP_VERSION__: string
