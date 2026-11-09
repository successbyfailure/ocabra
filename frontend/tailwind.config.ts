import type { Config } from "tailwindcss"

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      // Lenguaje de estados del dashboard (components/dashboard/format.ts):
      // parpadeo lento = esperando, pulso lento = procesando sin señal,
      // flujo = barra que avanza.
      keyframes: {
        "blink-slow": { "0%, 100%": { opacity: "1" }, "50%": { opacity: "0.3" } },
        flow: { from: { backgroundPosition: "0 0" }, to: { backgroundPosition: "24px 0" } },
      },
      animation: {
        "blink-slow": "blink-slow 1.8s ease-in-out infinite",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        flow: "flow 0.9s linear infinite",
      },
    },
  },
  plugins: [],
} satisfies Config
