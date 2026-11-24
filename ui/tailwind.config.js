/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: "#07070A",
          900: "#0C0C12",
          850: "#12121B",
          800: "#181824",
          700: "#222232",
          600: "#2E2E42",
        },
        violet: {
          400: "#A78BFA",
          500: "#8B5CF6",
          600: "#7C3AED",
          700: "#6D28D9",
        },
        gold: {
          400: "#FBBF24",
          500: "#F59E0B",
          600: "#D97706",
        },
        cyan: {
          400: "#22D3EE",
          500: "#06B6D4",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
}
