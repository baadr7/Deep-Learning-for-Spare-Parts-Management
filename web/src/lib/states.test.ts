import { describe, expect, it } from "vitest";
import { COST_LABELS, PALETTE, POLICY_COLORS, STATES, STATE_HEX } from "./states";

describe("STATES", () => {
  it("décrit les sept états machine dans l'ordre attendu", () => {
    expect(STATES.map((s) => s.label)).toEqual([
      "Opérationnelle",
      "En panne",
      "Attente pièce",
      "Attente technicien",
      "Réparation",
      "Maintenance préventive",
      "Test",
    ]);
  });

  it("fournit un style complet (fond, bordure, texte) pour chaque état", () => {
    for (const s of STATES) {
      expect(s.fill).toBeTruthy();
      expect(s.stroke).toBeTruthy();
      expect(s.text).toBeTruthy();
    }
  });

  it("associe une couleur ECharts hexadécimale à chaque état", () => {
    expect(STATE_HEX).toHaveLength(STATES.length);
    for (const hex of STATE_HEX) expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe("palettes et libellés", () => {
  it("colore les politiques connues", () => {
    expect(POLICY_COLORS.P4).toBe(PALETTE.plum);
    expect(Object.keys(POLICY_COLORS)).toEqual(expect.arrayContaining(["P1", "DQN", "Oracle"]));
  });

  it("traduit les postes de coûts en français", () => {
    expect(COST_LABELS.stockage).toBe("Stockage");
    expect(COST_LABELS.main_oeuvre).toBe("Main-d'œuvre");
    expect(COST_LABELS.arret).toBe("Arrêts machine");
  });
});
