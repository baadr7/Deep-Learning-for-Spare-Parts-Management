/** États des machines et couleurs associées (DESIGN.md, « États des machines »). */
export type StateStyle = { label: string; fill: string; stroke: string; text: string };

export const STATES: StateStyle[] = [
  { label: "Opérationnelle", fill: "var(--color-plum-100)", stroke: "var(--color-plum-100)", text: "var(--color-plum-900)" },
  { label: "En panne", fill: "var(--color-danger)", stroke: "var(--color-danger)", text: "white" },
  { label: "Attente pièce", fill: "var(--color-sun-400)", stroke: "var(--color-sun-400)", text: "var(--color-sun-800)" },
  { label: "Attente technicien", fill: "var(--color-sun-100)", stroke: "var(--color-sun-600)", text: "var(--color-sun-800)" },
  { label: "Réparation", fill: "var(--color-plum-700)", stroke: "var(--color-plum-700)", text: "white" },
  { label: "Maintenance préventive", fill: "var(--color-plum-400)", stroke: "var(--color-plum-400)", text: "white" },
  { label: "Test", fill: "var(--color-plum-200)", stroke: "var(--color-plum-200)", text: "var(--color-plum-900)" },
];

/** Valeurs résolues pour ECharts, qui ne lit pas les variables CSS. */
export const STATE_HEX = ["#e7defa", "#c8372d", "#f2c94c", "#fbf1c7", "#4e2683", "#9d7bd8", "#d6c6f2"];

export const PALETTE = {
  plum: "#5b2ca0",
  plumDark: "#3d1f6b",
  plumLight: "#b9a3e6",
  plumWash: "#efe9fb",
  sun: "#f2c94c",
  sunDark: "#a97c12",
  sunText: "#6b5014",
  ink: "#2d2438",
  inkSoft: "#655b73",
  line: "#e3dee9",
  danger: "#c8372d",
  grey: "#a69eb1",
};

export const COST_LABELS: Record<string, string> = {
  stockage: "Stockage",
  commandes: "Achats de pièces",
  urgences: "Surcoût urgences",
  main_oeuvre: "Main-d'œuvre",
  corrective: "Surcoût correctif",
  arret: "Arrêts machine",
  rebut: "Rebut",
  retouche: "Retouche",
  revision: "Révisions générales",
};

export const POLICY_COLORS: Record<string, string> = {
  P0: PALETTE.grey,
  P1: PALETTE.plumLight,
  P2: PALETTE.sunDark,
  P3: PALETTE.sun,
  P4: PALETTE.plum,
  P5: PALETTE.plumDark,
  Oracle: PALETTE.ink,
};
