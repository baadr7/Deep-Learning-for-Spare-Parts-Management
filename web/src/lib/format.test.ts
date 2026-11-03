import { describe, expect, it } from "vitest";
import { dayLabel, eventSentence, fmtHours, fmtInt, fmtNum, fmtPct, fmtUR, rolling } from "./format";

// Le séparateur de milliers fr-FR est une espace fine insécable (\u202f) ; on la tolère via \s.
const MILLIERS = /^1\s234$/;

describe("formatage des nombres", () => {
  it("renvoie « n.d. » pour les valeurs manquantes ou non finies", () => {
    expect(fmtInt(null)).toBe("n.d.");
    expect(fmtInt(undefined)).toBe("n.d.");
    expect(fmtInt(NaN)).toBe("n.d.");
    expect(fmtInt(Infinity)).toBe("n.d.");
    expect(fmtNum(undefined)).toBe("n.d.");
    expect(fmtPct(NaN)).toBe("n.d.");
    expect(fmtUR(null)).toBe("n.d.");
    expect(fmtHours(undefined)).toBe("n.d.");
  });

  it("formate les entiers sans décimale", () => {
    expect(fmtInt(0)).toBe("0");
    expect(fmtInt(1234)).toMatch(MILLIERS);
    expect(fmtInt(999.6)).toMatch(/^1\s000$/);
  });

  it("fixe le nombre de décimales demandé (virgule fr-FR)", () => {
    expect(fmtNum(3.14159, 2)).toBe("3,14");
    expect(fmtNum(2, 1)).toBe("2,0");
  });

  it("exprime un pourcentage à partir d'une fraction", () => {
    expect(fmtPct(0.1234)).toBe("12,3 %");
    expect(fmtPct(1)).toBe("100,0 %");
    expect(fmtPct(0, 0)).toBe("0 %");
  });

  it("ajoute l'unité UR aux entiers", () => {
    expect(fmtUR(7)).toBe("7 UR");
    expect(fmtUR(1234)).toMatch(/^1\s234 UR$/);
  });
});

describe("fmtHours", () => {
  it("affiche des heures en dessous de 48 h", () => {
    expect(fmtHours(0)).toBe("0,0 h");
    expect(fmtHours(47.9)).toBe("47,9 h");
  });

  it("bascule en jours à partir de 48 h", () => {
    expect(fmtHours(48)).toBe("2,0 j");
    expect(fmtHours(72)).toBe("3,0 j");
  });
});

describe("rolling", () => {
  it("calcule une moyenne glissante centrée sur la fenêtre par défaut", () => {
    const out = rolling([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(out).toHaveLength(9);
    // i = 0 : fenêtre 7 → indices 0..3 → (1+2+3+4)/4
    expect(out[0]).toBeCloseTo(2.5);
    // i = 4 : indices 1..7 → (2+…+8)/7
    expect(out[4]).toBe(5);
  });

  it("ignore les valeurs manquantes sans décaler la fenêtre", () => {
    const out = rolling([null, 2, null, 4], 3);
    expect(out).toEqual([2, 2, 3, 4]);
  });

  it("renvoie null lorsque la fenêtre ne contient aucune valeur exploitable", () => {
    expect(rolling([null, NaN, null], 3)).toEqual([null, null, null]);
  });
});

describe("dayLabel", () => {
  it("numérote les jours à partir de J1", () => {
    expect(dayLabel(0)).toBe("J1");
    expect(dayLabel(364)).toBe("J365");
  });
});

describe("eventSentence", () => {
  const components = ["Pompe", "Moteur"];
  const parts = [
    { code: "P-01", name: "Filtre" },
    { code: "P-02", name: "Roulement" },
  ];

  it("décrit une panne avec la machine et le composant en minuscules", () => {
    expect(eventSentence({ type: "panne", machine: 0, component: 0 }, components, parts)).toBe(
      "Machine 1 : panne, pompe",
    );
  });

  it("décrit une rupture de stock", () => {
    expect(eventSentence({ type: "rupture", machine: 1, component: null, part: 1 }, components, parts)).toBe(
      "Machine 2 attend une pièce P-02 (stock vide)",
    );
  });

  it("distingue une commande urgente d'une commande normale", () => {
    expect(
      eventSentence({ type: "commande", machine: null, component: null, part: 0, qty: 5, urgent: true }, components, parts),
    ).toBe("Commande urgente de 5 P-01");
    expect(
      eventSentence({ type: "commande", machine: null, component: null, part: 0, qty: 3 }, components, parts),
    ).toBe("Commande de 3 P-01");
  });

  it("décrit une remise en service", () => {
    expect(eventSentence({ type: "remise_en_service", machine: 2, component: null }, components, parts)).toBe(
      "Machine 3 remise en service",
    );
  });

  it("retombe sur le type brut pour un événement inconnu", () => {
    expect(eventSentence({ type: "inattendu", machine: null, component: null }, components, parts)).toBe("inattendu");
  });
});
