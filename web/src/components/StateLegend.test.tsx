import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { STATES } from "../lib/states";
import { StateLegend } from "./StateLegend";

afterEach(cleanup);

describe("StateLegend", () => {
  it("affiche une entrée par état machine", () => {
    render(<StateLegend />);
    expect(screen.getAllByRole("listitem")).toHaveLength(STATES.length);
  });

  it("reprend les libellés d'états de référence", () => {
    render(<StateLegend />);
    expect(screen.getByText("Opérationnelle")).toBeTruthy();
    expect(screen.getByText("Attente pièce")).toBeTruthy();
    expect(screen.getByText("Maintenance préventive")).toBeTruthy();
  });
});
