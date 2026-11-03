import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { useState } from "react";
import { Loading, Notice, Page, Section, Select } from "../components/ui";
import { api, type Interval, type TostResult, type TostVerdict } from "../lib/api";
import { Chart, axisStyle, baseOption } from "../lib/chart";
import { fmtInt, fmtNum, fmtPct, fmtUR } from "../lib/format";
import { PALETTE } from "../lib/states";

const SITUATIONS = [
  "délai fournisseur doublé",
  "taux de panne + 30 %",
  "un technicien de moins",
  "flotte homogène S0",
];

const SITUATION_LABELS: Record<string, string> = {
  "délai fournisseur doublé": "Délais fournisseurs doublés",
  "taux de panne + 30 %": "Taux de panne augmenté de 30 %",
  "un technicien de moins": "Équipe réduite à un technicien",
  "flotte homogène S0": "Flotte homogène S0",
};

const VERDICT_LABELS: Record<TostVerdict, string> = {
  équivalent: "équivalent à P1",
  meilleur: "meilleur que P1",
  "moins bon": "moins bon que P1",
  "non concluant": "non concluant",
};

/** Espaces d'action couverts par le plancher de représentation (décomposition de l'objectif O4). */
const ESPACE_LABELS: Record<string, string> = {
  reduced: "Réduit (36 actions)",
  n1: "N1 (6⁸)",
};

/** Agents appris comparés dans la campagne de généralisation du classement (item 12). */
const AGENTS = ["P4", "P5", "DQN", "BDQ"];

/** « Nominale » pour la composition de référence, « C0 », « C1 », … pour les compositions tirées. */
function compositionLabel(name: string): string {
  return name === "nominal" ? "Nominale" : `C${name.split("_").pop()}`;
}

/** Meilleur agent d'une composition : coût annuel moyen le plus faible. */
function bestAgent(costs: Record<string, number>): string | null {
  const ranked = AGENTS.filter((a) => costs[a] != null);
  if (!ranked.length) return null;
  return ranked.reduce((best, a) => (costs[a] < costs[best] ? a : best));
}

/** Valeur p : les très petites probabilités sont résumées plutôt qu'affichées en notation scientifique. */
function fmtP(p: number | undefined): string {
  if (p == null || !Number.isFinite(p)) return "n.d.";
  return p < 1e-4 ? "< 10⁻⁴" : fmtNum(p, 3);
}

function fmtSigned(v: number, unit: string): string {
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${fmtInt(Math.abs(v))} ${unit}`;
}

function CostCell({ interval }: { interval?: Interval }) {
  if (!interval || interval.mean == null) return <span className="text-ink-soft">n.d.</span>;
  return (
    <span>
      <span className="tabular">{fmtUR(interval.mean)}</span>
      <span className="block text-[12px] text-ink-soft tabular">
        {fmtUR(interval.low)} à {fmtUR(interval.high)}
      </span>
    </span>
  );
}

function VerdictBadge({ tost }: { tost?: TostResult }) {
  if (!tost) return <span className="text-ink-soft">référence</span>;
  return (
    <span className="inline-flex flex-col gap-0.5">
      <span
        className={clsx(
          "inline-block w-fit rounded-sm px-2 py-0.5 text-[13px]",
          tost.verdict === "meilleur" && "bg-sun-200 font-medium text-sun-800",
          tost.verdict === "moins bon" && "bg-danger-soft text-danger",
          (tost.verdict === "équivalent" || tost.verdict === "non concluant") && "bg-mist text-ink",
        )}
      >
        {VERDICT_LABELS[tost.verdict]}
      </span>
      <span className="text-[12px] text-ink-soft tabular">p = {fmtP(Math.min(tost.p_pas_pire, tost.p_pas_meilleur))}</span>
    </span>
  );
}

export function Evaluation() {
  const query = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluation });
  const [situation, setSituation] = useState(SITUATIONS[0]);
  const data = query.data;
  const generalisation = useQuery({ queryKey: ["generalisation"], queryFn: api.generalisation }).data;
  const plancher = useQuery({ queryKey: ["plancher"], queryFn: api.plancher }).data;
  const drawn = generalisation?.available
    ? generalisation.cases.filter((c) => c.nom !== "nominal")
    : [];

  return (
    <Page
      title="Évaluation et robustesse"
      description="Chaque politique est simulée sur les mêmes trente années (graines 9000 à 9029). Les écarts sont testés par Wilcoxon apparié et par équivalence TOST au seuil de 2 % du coût de P1."
    >
      {query.isLoading && <Loading label="Chargement de l'évaluation…" />}
      {query.error && <Notice tone="error">{(query.error as Error).message}</Notice>}
      {data && !data.available && (
        <Notice>
          L'évaluation n'a pas encore été calculée. Lancez
          <span className="tabular"> scripts/evaluate_policies.py --robustness</span>.
        </Notice>
      )}

      {data?.available && (
        <div className="space-y-12">
          <Section
            title="Scénario nominal"
            description={`Coût total et TRS sur ${data.replications} années simulées par politique, avec leur intervalle de confiance à 95 %. La référence des tests est P1.`}
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 pr-4 font-medium">Politique</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Coût total</th>
                    <th className="py-2.5 pr-4 text-right font-medium">TRS</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Écart de coût à P1</th>
                    <th className="py-2.5 pr-4 font-medium">Test d'équivalence</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Wilcoxon</th>
                  </tr>
                </thead>
                <tbody>
                  {data.policies.map((code) => {
                    const summary = data.nominal[code] ?? {};
                    const tost = data.tost_cout[code];
                    const reference = data.nominal.P1?.cout_total?.mean;
                    const variation =
                      tost && reference ? tost.ecart_moyen / reference : null;
                    return (
                      <tr key={code} className="border-b border-line">
                        <td className="py-2.5 pr-4 font-medium">{code}</td>
                        <td className="py-2.5 pr-4 text-right">
                          <CostCell interval={summary.cout_total} />
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtPct(summary.trs?.mean)}</td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {tost && variation != null ? (
                            <span className={clsx(variation < 0 ? "text-sun-800" : "text-ink")}>
                              {fmtSigned(tost.ecart_moyen, "UR")}
                              <span className="block text-[12px] text-ink-soft">{fmtNum(100 * variation, 1)} %</span>
                            </span>
                          ) : (
                            <span className="text-ink-soft">référence</span>
                          )}
                        </td>
                        <td className="py-2.5 pr-4">
                          <VerdictBadge tost={tost} />
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular text-ink-soft">{fmtP(data.wilcoxon_cout[code])}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Section>

          <Section
            title="Situations dégradées"
            description="Politiques réglées ou apprises sur le scénario nominal, confrontées sans nouveau réglage à quatre situations qui s'en écartent. La variation est celle du coût annuel moyen par rapport au scénario nominal."
            aside={
              <label className="flex items-center gap-2 text-[13px] text-ink-soft">
                Situation
                <Select value={situation} onChange={(e) => setSituation(e.target.value)} className="w-64">
                  {SITUATIONS.map((s) => (
                    <option key={s} value={s}>{SITUATION_LABELS[s]}</option>
                  ))}
                </Select>
              </label>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 pr-4 font-medium">Politique</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Coût total</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Variation / nominal</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Écart de coût à P1</th>
                    <th className="py-2.5 pr-4 font-medium">Test d'équivalence</th>
                  </tr>
                </thead>
                <tbody>
                  {data.policies.map((code) => {
                    const summary = data.robustesse[situation]?.[code];
                    const evaluated = summary != null;
                    const tost = data.robustesse_tost[situation]?.[code];
                    const nominal = data.nominal[code]?.cout_total?.mean;
                    const degraded = summary?.cout_total?.mean;
                    const variation =
                      degraded != null && nominal ? degraded / nominal - 1 : null;
                    return (
                      <tr key={code} className="border-b border-line">
                        <td className="py-2.5 pr-4 font-medium">{code}</td>
                        <td className="py-2.5 pr-4 text-right">
                          <CostCell interval={summary?.cout_total} />
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {variation == null ? (
                            <span className="text-ink-soft">n.d.</span>
                          ) : (
                            <span className={clsx(variation > 0 ? "text-ink" : "text-sun-800")}>
                              {variation > 0 ? "+" : "−"}
                              {fmtNum(Math.abs(100 * variation), 1)} %
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {tost ? (
                            fmtSigned(tost.ecart_moyen, "UR")
                          ) : evaluated ? (
                            <span className="text-ink-soft">référence</span>
                          ) : (
                            <span className="text-ink-soft">n.d.</span>
                          )}
                        </td>
                        <td className="py-2.5 pr-4">
                          {tost ? (
                            <VerdictBadge tost={tost} />
                          ) : evaluated ? (
                            <span className="text-ink-soft">référence</span>
                          ) : (
                            <span className="text-ink-soft">non évaluée</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[13px] text-ink-soft">
              La flotte S0 ne compte qu'une famille de machines, alors que P3 et les quatre agents
              appris observent un risque prédit par un réseau de survie entraîné pour deux
              familles : leurs cellules restent vides. Seules P0, P1, P2 et l'oracle y restent
              évaluables.
            </p>
          </Section>
        </div>
      )}

      {plancher?.available && (
        <div className="mt-12">
          <Section
            title="Plancher de représentation des espaces d'action"
            description={`Un agent ne peut pas descendre sous le meilleur coût de son propre répertoire d'actions. Le plancher mesure, sur les mêmes ${plancher.replications} années, le coût de la meilleure politique constante de chaque espace. L'écart de l'agent à ce plancher est ce qu'il n'a pas su tirer d'un répertoire qui le lui permettait ; l'écart du plancher à P1 est ce que ce répertoire, dans sa forme la plus simple, laisse au-dessus de la référence.`}
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 pr-4 font-medium">Politique</th>
                    <th className="py-2.5 pr-4 font-medium">Espace d'action</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Coût agent</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Plancher</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Écart d'apprentissage</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Écart de représentation</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(plancher.decomposition).map(([code, dec]) => {
                    const reference = plancher.bornes[plancher.reference];
                    const share = (v: number) =>
                      reference ? ` (${fmtNum(100 * v / reference, 1)} %)` : "";
                    return (
                      <tr key={code} className="border-b border-line">
                        <td className="py-2.5 pr-4 font-medium">{code}</td>
                        <td className="py-2.5 pr-4">{ESPACE_LABELS[dec.espace] ?? dec.espace}</td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtUR(dec.cout_agent)}</td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtUR(dec.cout_plancher)}</td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {fmtSigned(dec.ecart_apprentissage, "UR")}
                          <span className="block text-[12px] text-ink-soft tabular">
                            {share(dec.ecart_apprentissage)}
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {fmtSigned(dec.ecart_representation, "UR")}
                          <span className="block text-[12px] text-ink-soft tabular">
                            {share(dec.ecart_representation)}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                  <tr className="border-b border-line">
                    <td className="py-2.5 pr-4 font-medium">{plancher.reference}</td>
                    <td className="py-2.5 pr-4 text-ink-soft">---</td>
                    <td className="py-2.5 pr-4 text-right tabular">{fmtUR(plancher.bornes[plancher.reference])}</td>
                    <td className="py-2.5 pr-4 text-right text-ink-soft">---</td>
                    <td className="py-2.5 pr-4 text-right text-ink-soft">---</td>
                    <td className="py-2.5 pr-4 text-right text-ink-soft">---</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[13px] text-ink-soft">
              L'écart à P1 se lit comme la somme des deux colonnes. P2, qui coûte{" "}
              {fmtUR(plancher.bornes.P2)} et bat P1 sans apprentissage, est une politique à décision
              machine par machine : le vocabulaire du niveau N2 permet d'exprimer ce type de
              décision, si bien que le critère O4 est atteignable dans l'espace de décision complet.
              Ce qui manque aux agents n'est donc pas l'existence d'une solution, mais leur capacité
              à la retrouver avec la représentation et le budget d'entraînement dont ils disposent.
            </p>
          </Section>
        </div>
      )}

      {generalisation?.available && (
        <div className="mt-12">
          <Section
            title="Généralisation du classement"
            description={`Le classement nominal vaut pour une composition précise de la flotte. Cinq compositions tirées par un plan de Dirichlet centré sur le nominal sont rejouées sur les mêmes ${generalisation.replications} années ; P1 et P3 sont réglés composition par composition, les autres politiques transférées. τ est le tau de Kendall du classement de la composition contre le nominal.`}
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[860px] text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 pr-4 font-medium">Flotte</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Techniciens</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Part U</th>
                    <th className="py-2.5 pr-4 text-right font-medium">τ</th>
                    <th className="py-2.5 pr-4 font-medium">Ordre attendu</th>
                    <th className="py-2.5 pr-4 text-right font-medium">P1 (UR)</th>
                    <th className="py-2.5 pr-4 text-right font-medium">P2 (UR)</th>
                    <th className="py-2.5 pr-4 font-medium">Meilleur agent</th>
                  </tr>
                </thead>
                <tbody>
                  {generalisation.cases.map((c) => {
                    const best = bestAgent(c.cout_moyen);
                    return (
                      <tr key={c.nom} className="border-b border-line">
                        <td className="py-2.5 pr-4 font-medium">{compositionLabel(c.nom)}</td>
                        <td className="py-2.5 pr-4 text-right tabular">{c.techniciens}</td>
                        <td className="py-2.5 pr-4 text-right tabular">
                          {c.part_U == null ? <span className="text-ink-soft">---</span> : fmtNum(c.part_U, 2)}
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtNum(c.kendall_vs_nominal, 3)}</td>
                        <td className="py-2.5 pr-4">
                          <span
                            className={clsx(
                              "inline-block rounded-sm px-2 py-0.5 text-[13px]",
                              c.ordre_cdc.conserve ? "bg-mist text-ink" : "bg-danger-soft text-danger",
                            )}
                          >
                            {c.ordre_cdc.conserve ? "conservé" : "renversé"}
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtInt(c.cout_moyen.P1)}</td>
                        <td className="py-2.5 pr-4 text-right tabular">{fmtInt(c.cout_moyen.P2)}</td>
                        <td className="py-2.5 pr-4 tabular">
                          {best ? (
                            `${best} (${fmtInt(c.cout_moyen[best])})`
                          ) : (
                            <span className="text-ink-soft">---</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-8">
              <Chart
                ariaLabel="Concordance du classement de chaque composition tirée avec le classement nominal"
                height={280}
                option={{
                  ...baseOption(),
                  legend: { show: false },
                  grid: { left: 52, right: 16, top: 28, bottom: 36 },
                  tooltip: {
                    ...baseOption().tooltip,
                    formatter: (p: { name: string; value: number }) =>
                      `${p.name}<br/>τ = ${fmtNum(p.value, 3)}`,
                  },
                  xAxis: { type: "category", data: drawn.map((c) => compositionLabel(c.nom)), ...axisStyle },
                  yAxis: {
                    type: "value",
                    min: 0,
                    max: 1.05,
                    ...axisStyle,
                    axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => fmtNum(v, 2) },
                  },
                  series: [
                    {
                      type: "bar",
                      barMaxWidth: 34,
                      data: drawn.map((c) => c.kendall_vs_nominal),
                      itemStyle: { color: PALETTE.plum, borderRadius: [3, 3, 0, 0] },
                      markLine: {
                        silent: true,
                        symbol: "none",
                        lineStyle: { color: PALETTE.inkSoft, type: "dashed" },
                        label: { formatter: "τ = 1", position: "insideEndTop" },
                        data: [{ yAxis: 1 }],
                      },
                    },
                  ],
                }}
              />
            </div>

            <p className="mt-3 text-[13px] text-ink-soft">
              La ligne en pointillés marque un classement identique au nominal (τ = 1). L'ordre attendu
              du cahier des charges — P2 &lt; P3 &lt; P1 &lt; agents — est conservé sur quatre compositions
              sur six. Il est renversé en C0, à un seul technicien, où P1 dont le stock absorbe les
              délais passe devant P2 et P3, et en C3, à trois techniciens et forte part de la famille
              soumise à l'usure, où P4 passe devant P1.
            </p>
          </Section>
        </div>
      )}
    </Page>
  );
}
