import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Evaluation } from "./Evaluation";

// Le rendu ECharts exige un canevas et un ResizeObserver absents de jsdom : la figure est
// neutralisée, les tests portent sur les tableaux et les verdicts.
vi.mock("../lib/chart", () => ({
  Chart: () => null,
  baseOption: () => ({}),
  axisStyle: {},
}));

/** Rapport réduit à trois politiques, dont P1 prise pour référence des tests. */
const REPORT = {
  available: true,
  scenario: "s1_nominal",
  replications: 30,
  days: 365,
  policies: ["P0", "P1", "P3"],
  tost_margin: 0.02,
  nominal: {
    P0: { cout_total: { mean: 1200, low: 1100, high: 1300 }, trs: { mean: 0.6, low: 0.58, high: 0.62 } },
    P1: { cout_total: { mean: 1000, low: 950, high: 1050 }, trs: { mean: 0.65, low: 0.63, high: 0.67 } },
    P3: { cout_total: { mean: 900, low: 860, high: 940 }, trs: { mean: 0.67, low: 0.65, high: 0.69 } },
  },
  wilcoxon_cout: { P0: 0.001, P3: 0.004 },
  tost_cout: {
    P0: {
      ecart_moyen: 200,
      marge: 20,
      marge_relative: 0.02,
      p_pas_pire: 0.9,
      p_pas_meilleur: 0.001,
      verdict: "moins bon",
    },
    P3: {
      ecart_moyen: -100,
      marge: 20,
      marge_relative: 0.02,
      p_pas_pire: 0.002,
      p_pas_meilleur: 0.98,
      verdict: "meilleur",
    },
  },
  robustesse: {
    "délai fournisseur doublé": {
      P0: { cout_total: { mean: 1500, low: 1400, high: 1600 } },
      P1: { cout_total: { mean: 1050, low: 1000, high: 1100 } },
      P3: { cout_total: { mean: 960, low: 920, high: 1000 } },
    },
    "flotte homogène S0": {
      P0: { cout_total: { mean: 1100, low: 1000, high: 1200 } },
      P1: { cout_total: { mean: 900, low: 850, high: 950 } },
    },
  },
  robustesse_wilcoxon: { "délai fournisseur doublé": { P0: 0.001 } },
  robustesse_tost: {
    "délai fournisseur doublé": {
      P0: {
        ecart_moyen: 450,
        marge: 21,
        marge_relative: 0.02,
        p_pas_pire: 1,
        p_pas_meilleur: 0.001,
        verdict: "moins bon",
      },
      P3: {
        ecart_moyen: -90,
        marge: 21,
        marge_relative: 0.02,
        p_pas_pire: 0.003,
        p_pas_meilleur: 0.97,
        verdict: "meilleur",
      },
    },
    "flotte homogène S0": {
      P0: {
        ecart_moyen: 200,
        marge: 18,
        marge_relative: 0.02,
        p_pas_pire: 1,
        p_pas_meilleur: 0.001,
        verdict: "moins bon",
      },
    },
  },
};

/** Campagne de généralisation réduite : la composition nominale et deux compositions tirées. */
const GENERALISATION = {
  available: true,
  base: "s1_nominal",
  replications: 30,
  days: 365,
  policies: ["P0", "P1", "P2", "P3", "DQN", "P4", "P5", "BDQ", "Oracle"],
  transfert_agents: true,
  cases: [
    {
      nom: "nominal",
      techniciens: 2,
      part_U: null,
      kendall_vs_nominal: 1,
      ordre_cdc: { conserve: true, controles: { "P2<P3": true, "P3<P1": true, "P1<agents": true } },
      cout_moyen: { P1: 52136, P2: 47135, P4: 52799, P5: 67049, DQN: 54932, BDQ: 74825 },
    },
    {
      nom: "composition_00",
      techniciens: 1,
      part_U: 0.42,
      kendall_vs_nominal: 0.667,
      ordre_cdc: { conserve: false, controles: { "P2<P3": false, "P3<P1": false, "P1<agents": true } },
      cout_moyen: { P1: 52642, P2: 53061, P4: 52954, P5: 63525, DQN: 54446, BDQ: 92820 },
    },
    {
      nom: "composition_02",
      techniciens: 2,
      part_U: 0.48,
      kendall_vs_nominal: 1,
      ordre_cdc: { conserve: true, controles: { "P2<P3": true, "P3<P1": true, "P1<agents": true } },
      cout_moyen: { P1: 50033, P2: 45997, P4: 50672, P5: 65072, DQN: 53042, BDQ: 71734 },
    },
  ],
};

/** Plancher de représentation : les deux agents à espace d'action paramétrique. */
const PLANCHER = {
  available: true,
  scenario: "s1_nominal",
  replications: 30,
  days: 365,
  seeds: [9000, 9001],
  reference: "P1",
  bornes: { P1: 52136, P2: 47135, P3: 49220, Oracle: 42408 },
  espaces: {
    reduced: { methode: "énumération exhaustive des 36 actions", n_actions: 36, exhaustif: true, plancher: { cout_total: 54279 } },
    n1: { methode: "montée de coordonnée", n_actions: 1679616, exhaustif: false, plancher: { cout_total: 50185 } },
  },
  decomposition: {
    DQN: {
      espace: "reduced",
      cout_agent: 54932,
      cout_plancher: 54279,
      cout_p1: 52136,
      ecart_observe: 2796,
      ecart_apprentissage: 652,
      ecart_representation: 2143,
      part_apprentissage: 0.233,
    },
    P4: {
      espace: "n1",
      cout_agent: 52799,
      cout_plancher: 50185,
      cout_p1: 52136,
      ecart_observe: 662,
      ecart_apprentissage: 2614,
      ecart_representation: -1952,
      part_apprentissage: 3.946,
    },
  },
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

async function renderPage() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const target = String(url);
      const body = target.includes("/generalisation")
        ? GENERALISATION
        : target.includes("/plancher")
          ? PLANCHER
          : REPORT;
      return { ok: true, json: async () => body };
    }),
  );
  render(<Evaluation />, { wrapper });
  await screen.findByRole("heading", { name: "Scénario nominal" });
}

/** Table des situations dégradées : la seconde du document. */
function degradedTable() {
  return screen.getAllByRole("table")[1];
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Evaluation", () => {
  it("présente le scénario nominal et signale P1 comme référence", async () => {
    await renderPage();
    expect(screen.getByRole("heading", { name: "Évaluation et robustesse" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Situations dégradées" })).toBeTruthy();
    // P1 n'est jamais comparée à elle-même : ni écart, ni test d'équivalence.
    expect(screen.getAllByText("référence").length).toBeGreaterThanOrEqual(1);
  });

  it("affiche les verdicts d'équivalence lus dans le rapport", async () => {
    await renderPage();
    expect(screen.getAllByText("moins bon que P1").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("meilleur que P1").length).toBeGreaterThanOrEqual(1);
  });

  it("change de situation dégradée depuis le sélecteur", async () => {
    await renderPage();
    const table = degradedTable();
    expect(within(table).getByText("+25,0 %")).toBeTruthy();

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "flotte homogène S0" } });
    // P3 n'est pas évaluable sur une flotte à famille unique : ses cellules se vident.
    await waitFor(() => expect(within(degradedTable()).getByText("non évaluée")).toBeTruthy());
    expect(within(degradedTable()).getAllByText("n.d.").length).toBeGreaterThanOrEqual(2);
  });

  it("restitue la généralisation du classement par composition", async () => {
    await renderPage();
    expect(screen.getByRole("heading", { name: "Généralisation du classement" })).toBeTruthy();
    // La composition nominale n'a pas de part de famille U : sa cellule reste vide.
    expect(screen.getAllByText("---").length).toBeGreaterThanOrEqual(1);
    // Une seule composition renverse l'ordre attendu, les deux autres le conservent.
    expect(screen.getByText("renversé")).toBeTruthy();
    expect(screen.getAllByText("conservé").length).toBe(2);
    // Le meilleur agent de chaque composition est nommé avec son coût annuel.
    expect(screen.getByText("P4 (50 672)")).toBeTruthy();
  });

  it("décompose l'écart des agents à P1 par espace d'action", async () => {
    await renderPage();
    expect(
      screen.getByRole("heading", { name: "Plancher de représentation des espaces d'action" }),
    ).toBeTruthy();
    // Chaque agent est rattaché à son espace d'action.
    expect(screen.getByText("Réduit (36 actions)")).toBeTruthy();
    expect(screen.getByText("N1 (6⁸)")).toBeTruthy();
    // L'espace N1 contient un réglage fixe sous P1 : la part de représentation de P4 est négative.
    expect(screen.getByText(/−1 952 UR/)).toBeTruthy();
  });
});
