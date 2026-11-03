import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Reproductibilite } from "./Reproductibilite";

afterEach(cleanup);

describe("Reproductibilite", () => {
  it("liste les campagnes dans l'ordre où elles s'enchaînent", () => {
    render(<Reproductibilite />);
    expect(screen.getByRole("heading", { name: "Reproductibilité" })).toBeTruthy();
    expect(screen.getByText("uv run python scripts/optimize_p1.py")).toBeTruthy();
    expect(screen.getByText("uv run python scripts/evaluate_policies.py --robustness")).toBeTruthy();
    // La compilation du rapport clôt la chaîne : sans elle, les résultats ne sont pas livrés.
    expect(screen.getByText("latexmk -xelatex rapport.tex")).toBeTruthy();
  });

  it("détaille les plages de graines des campagnes", () => {
    render(<Reproductibilite />);
    expect(screen.getByText("Évaluation finale (30 réplications) et robustesse")).toBeTruthy();
    expect(screen.getByText("9 000 à 9 029")).toBeTruthy();
  });
});
