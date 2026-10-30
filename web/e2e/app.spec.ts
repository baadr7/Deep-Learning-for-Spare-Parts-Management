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
