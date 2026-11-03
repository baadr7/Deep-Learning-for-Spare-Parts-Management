import { Page, Section } from "../components/ui";

/** Campagnes d'entraînement et d'évaluation, dans l'ordre où elles s'enchaînent. */
const COMMANDS: { command: string; effect: string }[] = [
  { command: "uv sync --extra ml", effect: "Installation des dépendances, PyTorch compris" },
  {
    command: "uv run python scripts/calibrate_azure.py",
    effect: "Lois de durée de vie et familles, depuis le jeu Azure",
  },
  { command: "uv run python scripts/optimize_p1.py", effect: "Seuils de la politique P1" },
  { command: "uv run python scripts/train_predictor.py", effect: "Modèles de prédiction des pannes" },
  { command: "uv run python scripts/tune_p3.py", effect: "Seuils de la politique prédictive P3" },
  {
    command: "uv run python scripts/train_rl.py --level n1 --risk --drift --curriculum --seeds 0 1 2",
    effect: "Agent P4 ; --level n2 pour P5, --level reduced --algo dqn pour le DQN",
  },
  {
    command: "uv run python scripts/train_bdq.py --risk --drift --curriculum --seeds 0 1 2",
    effect: "Branching Dueling Q-Network, au niveau N2",
  },
  { command: "uv run python scripts/choose_agent_mode.py", effect: "Mode de jeu de P4 et P5" },
  { command: "uv run python scripts/train_surrogate.py", effect: "Modèle de substitution de l'application" },
  { command: "uv run python scripts/sensitivity.py", effect: "Indices de Sobol sur le modèle de substitution" },
  {
    command: "uv run python scripts/evaluate_policies.py --robustness",
    effect: "Évaluation sur trente années et quatre situations dégradées",
  },
  {
    command: "uv run python scripts/generate_fleets.py --n 5",
    effect: "Compositions de flotte tirées, par plan de Dirichlet centré sur le nominal",
  },
  {
    command: "uv run python scripts/generalize_ranking.py --replications 30",
    effect: "Stabilité du classement selon la composition de la flotte",
  },
  { command: "uv run python scripts/export_report_data.py", effect: "Données des graphiques du rapport" },
  { command: "latexmk -xelatex rapport.tex", effect: "Rapport en PDF, depuis docs/rapport" },
];

const SEEDS: { usage: string; seeds: string }[] = [
  { usage: "Calibration des lois de durée de vie", seeds: "0 à 3" },
  { usage: "Jeu d'apprentissage de la prédiction (cinq années simulées)", seeds: "1 à 5" },
  { usage: "Réglage de P1, de P3 et du mode de jeu des agents", seeds: "101, 102, 103" },
  { usage: "Sélection du meilleur modèle des agents appris", seeds: "8 000 à 8 004" },
  { usage: "Évaluation finale (30 réplications) et robustesse", seeds: "9 000 à 9 029" },
  { usage: "Entraînement des agents (trois graines par agent)", seeds: "0, 1, 2" },
];

export function Reproductibilite() {
  return (
    <Page
      title="Reproductibilité"
      description="L'ordre dans lequel les campagnes s'enchaînent et les graines qui les rendent déterministes. Chaque commande rejoue les mêmes années et réécrit les mêmes fichiers ; la liste complète figure en annexe du rapport."
    >
      <div className="space-y-12">
        <Section
          title="Commandes des campagnes"
          description="L'API (uv run uvicorn spareparts.api.app:app) et l'interface (npm run dev) doivent tourner pour consulter ces résultats dans l'application."
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-[14px]">
              <thead>
                <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                  <th className="py-2.5 pr-4 font-medium">Commande</th>
                  <th className="py-2.5 pr-4 font-medium">Résultat</th>
                </tr>
              </thead>
              <tbody>
                {COMMANDS.map(({ command, effect }) => (
                  <tr key={command} className="border-b border-line align-top">
                    <td className="py-2.5 pr-4 font-mono text-[13px] break-words">{command}</td>
                    <td className="py-2.5 pr-4">{effect}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section
          title="Graines aléatoires"
          description="Aucune graine n'est choisie au hasard : chaque étape dispose de sa plage, et les plages restent disjointes."
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-[14px]">
              <thead>
                <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                  <th className="py-2.5 pr-4 font-medium">Étape</th>
                  <th className="py-2.5 pr-4 text-right font-medium">Graines</th>
                </tr>
              </thead>
              <tbody>
                {SEEDS.map(({ usage, seeds }) => (
                  <tr key={usage} className="border-b border-line">
                    <td className="py-2.5 pr-4">{usage}</td>
                    <td className="py-2.5 pr-4 text-right tabular">{seeds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[13px] text-ink-soft">
            Une même graine produit les mêmes pannes pour toutes les politiques : à l'intérieur d'une
            campagne, celles-ci affrontent donc les mêmes années, et l'écart de coût qui les sépare
            tient à la politique et non au tirage. Les graines d'évaluation sont postérieures à
            toutes celles du réglage et de la sélection, ce qui écarte un biais optimiste.
          </p>
        </Section>
      </div>
    </Page>
  );
}
