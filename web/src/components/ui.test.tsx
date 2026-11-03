import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Button, KpiList, Loading, Notice, Page, Section } from "./ui";

afterEach(cleanup);

describe("Page", () => {
  it("affiche le titre, la description, les actions et le contenu", () => {
    render(
      <Page title="Bilan de l'année" description="Indicateurs de référence" actions={<Button>Simuler</Button>}>
        <p>Contenu de la page</p>
      </Page>,
    );
    expect(screen.getByRole("heading", { name: "Bilan de l'année" })).toBeTruthy();
    expect(screen.getByText("Indicateurs de référence")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Simuler" })).toBeTruthy();
    expect(screen.getByText("Contenu de la page")).toBeTruthy();
  });
});

describe("Section", () => {
  it("affiche le titre et sa description", () => {
    render(
      <Section title="Coûts" description="Par poste">
        <span>corps</span>
      </Section>,
    );
    expect(screen.getByRole("heading", { name: "Coûts" })).toBeTruthy();
    expect(screen.getByText("Par poste")).toBeTruthy();
    expect(screen.getByText("corps")).toBeTruthy();
  });
});

describe("Notice", () => {
  it("expose le rôle status pour un message d'information", () => {
    render(<Notice>Information neutre</Notice>);
    expect(screen.getByRole("status").textContent).toBe("Information neutre");
  });

  it("expose le rôle alert pour une erreur", () => {
    render(<Notice tone="error">Échec de la simulation</Notice>);
    expect(screen.getByRole("alert").textContent).toBe("Échec de la simulation");
  });
});

describe("KpiList", () => {
  it("rend chaque indicateur avec son libellé, son indice et sa valeur", () => {
    render(
      <KpiList
        items={[
          { label: "TRS", value: "82,4 %" },
          { label: "Coût total", value: "1 250 UR", hint: "sur 365 jours", emphasis: true },
        ]}
      />,
    );
    expect(screen.getByText("TRS")).toBeTruthy();
    expect(screen.getByText("82,4 %")).toBeTruthy();
    expect(screen.getByText("1 250 UR")).toBeTruthy();
    expect(screen.getByText("sur 365 jours")).toBeTruthy();
  });
});

describe("Button", () => {
  it("propage les attributs et reflète l'état désactivé", () => {
    render(
      <Button type="submit" disabled>
        Simuler 365 jours
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Simuler 365 jours" }) as HTMLButtonElement;
    expect(button.type).toBe("submit");
    expect(button.disabled).toBe(true);
  });
});

describe("Loading", () => {
  it("affiche le message passé en libellé", () => {
    render(<Loading label="Calcul en cours" />);
    expect(screen.getByText("Calcul en cours")).toBeTruthy();
  });
});
