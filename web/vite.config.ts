import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: env.SPAREPARTS_API_TARGET || "http://127.0.0.1:8000",
          ws: true,
        },
      },
    },
    build: {
      // ECharts et zrender forment un lot distinct pour réduire le poids initial.
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes("node_modules")) return undefined;
            if (id.includes("echarts") || id.includes("zrender")) return "echarts";
            return "vendor";
          },
        },
      },
      // ECharts élagué (Bar, Line, Scatter et rendu canvas) pèse ≈ 560 kB, juste au-dessus du
      // seuil d'alerte par défaut : ce lot est isolé et chargé à la demande, le seuil est relevé.
      chunkSizeWarningLimit: 600,
    },
    test: {
      environment: "jsdom",
      exclude: ["e2e/**", "node_modules/**"],
    },
  };
});
