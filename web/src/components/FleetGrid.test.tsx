import { cleanup, render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { FleetGrid } from "./FleetGrid";

afterEach(cleanup);

/** FleetGrid s'appuie sur useNavigate : il faut donc un routeur autour du composant. */
function renderGrid(ui: ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe("FleetGrid", () => {
  it("groupe les machines par famille et compte les arrêts du jour", () => {
    renderGrid(
      <FleetGrid
        states={[0, 1, 0, 1]}
        families={["a", "a", "b", "b"]}
        familyLabels={{ a: "Famille A", b: "Famille B" }}
      />,
    );

    expect(screen.getByText("Famille A")).toBeTruthy();
    expect(screen.getByText("Famille B")).toBeTruthy();
    // Deux machines par famille.
    expect(screen.getAllByText("2 machines")).toHaveLength(2);
    // Une machine arrêtée dans chaque famille (état 1 = « En panne »).
    expect(screen.getAllByText("1 arrêtée ce jour")).toHaveLength(2);
  });

  it("labellise chaque case par son numéro de machine et son état", () => {
    renderGrid(<FleetGrid states={[0, 1]} families={["a", "a"]} familyLabels={{ a: "Famille A" }} />);

    expect(screen.getByRole("button", { name: "Machine 1, opérationnelle" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Machine 2, en panne" })).toBeTruthy();
  });

  it("signale l'absence d'arrêt lorsque toutes les machines tournent", () => {
    renderGrid(<FleetGrid states={[0, 0]} families={["a", "a"]} familyLabels={{ a: "Famille A" }} />);
    expect(screen.getByText("aucun arrêt ce jour")).toBeTruthy();
  });

  it("désactive les cases tant que la grille n'est pas cliquable", () => {
    renderGrid(<FleetGrid states={[0]} families={["a"]} familyLabels={{ a: "Famille A" }} />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
  });

  it("active les cases lorsque la grille est cliquable", () => {
    renderGrid(<FleetGrid states={[0]} families={["a"]} familyLabels={{ a: "Famille A" }} linkable />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.disabled).toBe(false);
  });
});
