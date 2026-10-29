import { defineConfig } from "@playwright/test";

// Parcours principal de l'application (ENF-06). L'API et l'interface doivent tourner :
// uv run uvicorn spareparts.api.app:app et npm run dev. Les tests utilisent Microsoft Edge
// installé sur le poste, aucun navigateur supplémentaire n'est téléchargé.
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:5173",
    channel: "msedge",
    viewport: { width: 1440, height: 1000 },
  },
});
