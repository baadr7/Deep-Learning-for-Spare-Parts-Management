import { expect, test } from "@playwright/test";

const SHOTS = process.env.SHOTS_DIR;

test("le tableau de bord affiche le bilan de l'année", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Bilan de l'année" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("TRS", { exact: true }).first()).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-dashboard.png`, fullPage: true });
});

test("la flotte en direct avance jour après jour", async ({ page }) => {
  await page.goto("/direct?lancer=1");
  await expect(page.getByText(/Jour \d+ sur 365/)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("button", { name: /Machine 1,/ })).toBeVisible();
  await page.waitForTimeout(5_000);
  await expect(page.getByRole("heading", { name: "Journal" })).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-live.png`, fullPage: true });
});

test("une ligne du tableau des machines ouvre la fiche", async ({ page }) => {
  await page.goto("/machines");
  const firstRow = page.locator("tbody tr").first();
  await expect(firstRow).toBeVisible({ timeout: 30_000 });
  await firstRow.click();
  await expect(page.getByRole("heading", { name: "Télémétrie" })).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-machine.png`, fullPage: true });
});

test("la comparaison produit un tableau d'indicateurs", async ({ page }) => {
  await page.goto("/comparaison");
  // Trois réplications au moins : le champ refuse moins, l'intervalle de confiance n'aurait pas de sens.
  await page.getByLabel("Réplications").fill("3");
  await page.getByRole("button", { name: "Comparer" }).click();
  await expect(page.getByRole("heading", { name: "Indicateurs" })).toBeVisible({ timeout: 60_000 });
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-compare.png`, fullPage: true });
});

test("le tableau des machines liste la flotte", async ({ page }) => {
  await page.goto("/machines");
  await expect(page.locator("tbody tr").first()).toBeVisible({ timeout: 30_000 });
  // Capture de la fenêtre seule : le tableau de 100 machines est trop long pour une page du
  // rapport, tandis que l'en-tête et les premières lignes suffisent à illustrer la liste.
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-machines.png` });
});

test("le magasin affiche niveaux, commandes et seuils", async ({ page }) => {
  await page.goto("/stock");
  // La page attend la simulation de référence : le tableau n'apparaît qu'ensuite.
  await expect(page.getByRole("columnheader", { name: "Taux de service" })).toBeVisible({ timeout: 60_000 });
  // Laisser les graphiques de stock se dessiner avant la capture.
  await page.waitForTimeout(1_500);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-stock.png`, fullPage: true });
});

test("la page des scénarios interroge le modèle de substitution", async ({ page }) => {
  await page.goto("/scenarios");
  await expect(page.getByRole("heading", { name: "Scénarios" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Estimation instantanée" })).toBeVisible({ timeout: 30_000 });
  await page.waitForTimeout(1_500);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-scenarios.png`, fullPage: true });
});

test("la page d'évaluation restitue les tests d'équivalence", async ({ page }) => {
  await page.goto("/evaluation");
  // Le tableau n'apparaît qu'une fois le rapport d'évaluation chargé.
  await expect(page.getByRole("heading", { name: "Évaluation et robustesse" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Scénario nominal" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Situations dégradées" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Généralisation du classement" })).toBeVisible();
  await page.waitForTimeout(1_000);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/e2e-evaluation.png`, fullPage: true });
});

test("la page de reproductibilité liste les campagnes", async ({ page }) => {
  await page.goto("/reproductibilite");
  await expect(page.getByRole("heading", { name: "Reproductibilité" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Commandes des campagnes" })).toBeVisible();
  await expect(page.getByText("latexmk -xelatex rapport.tex")).toBeVisible();
});
