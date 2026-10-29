import { useMemo } from "react";
import { RunControls } from "../components/RunControls";
import { StateLegend } from "../components/StateLegend";
import { KpiList, Loading, Notice, Page, Section } from "../components/ui";
import type { RunPayload } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { fmtHours, fmtInt, fmtPct, fmtUR, rolling } from "../lib/format";
import { useRun } from "../lib/run";
import { COST_LABELS, PALETTE, STATE_HEX, STATES } from "../lib/states";

function days(run: RunPayload) {
  return run.daily.trs.map((_, i) => i + 1);
}

function trsOption(run: RunPayload) {
  const x = days(run);
  return {
    ...baseOption(),
    xAxis: { type: "category", data: x, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 29 } },
    yAxis: {
      type: "value",
      min: (v: { min: number }) => Math.floor(v.min * 50) / 50,
      max: 1,
      ...axisStyle,
      axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => `${Math.round(v * 100)} %` },
    },
    tooltip: { ...baseOption().tooltip, valueFormatter: (v: number) => fmtPct(v) },
    series: [
      { name: "TRS", type: "line", data: rolling(run.daily.trs), showSymbol: false, lineStyle: { width: 2.25, color: PALETTE.plum }, itemStyle: { color: PALETTE.plum } },
      { name: "Disponibilité", type: "line", data: rolling(run.daily.availability), showSymbol: false, lineStyle: { width: 1.5, color: PALETTE.sunDark }, itemStyle: { color: PALETTE.sunDark } },
      { name: "Qualité", type: "line", data: rolling(run.daily.quality), showSymbol: false, lineStyle: { width: 1.5, color: PALETTE.grey, type: "dashed" }, itemStyle: { color: PALETTE.grey } },
    ],
  };
}

const COST_ORDER = ["rebut", "retouche", "arret", "commandes", "corrective", "main_oeuvre", "revision", "urgences", "stockage"];
const COST_COLORS = [PALETTE.plumDark, PALETTE.plum, PALETTE.sun, PALETTE.plumLight, PALETTE.sunDark, "#cbbcec", PALETTE.grey, "#e9d79b", "#ddd7e6"];

function costOption(run: RunPayload) {
  const x = days(run);
  return {
    ...baseOption(),
    xAxis: { type: "category", data: x, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 29 } },
    yAxis: { type: "value", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => fmtInt(v) } },
    legend: { ...baseOption().legend, type: "plain" },
    grid: { left: 64, right: 16, top: 60, bottom: 32 },
    tooltip: { ...baseOption().tooltip, valueFormatter: (v: number) => fmtUR(v) },
    series: COST_ORDER.map((k, i) => ({
      name: COST_LABELS[k],
      type: "line",
      stack: "cout",
      data: run.daily.cost_cumulative[k],
      showSymbol: false,
      lineStyle: { width: 0 },
      areaStyle: { color: COST_COLORS[i], opacity: 1 },
      itemStyle: { color: COST_COLORS[i] },
    })),
  };
}

function statesOption(run: RunPayload) {
  const x = days(run);
  return {
    ...baseOption(),
    legend: { show: false },
    xAxis: { type: "category", data: x, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 29 } },
    yAxis: { type: "value", minInterval: 1, ...axisStyle },
    tooltip: { ...baseOption().tooltip },
    series: STATES.slice(1).map((s, i) => ({
      name: s.label,
      type: "bar",
      stack: "etats",
      barCategoryGap: "10%",
      data: run.daily.state_counts.map((row) => row[i + 1]),
      itemStyle: { color: STATE_HEX[i + 1] },
    })),
  };
}

export function Dashboard() {
  const { run, loading, error } = useRun();
  const options = useMemo(
    () => (run ? { trs: trsOption(run), cost: costOption(run), states: statesOption(run) } : null),
    [run],
  );

  return (
    <Page
      title="Tableau de bord"
      description="Bilan d'une année simulée de la flotte pour un scénario et une politique de gestion. Les courbes sont lissées sur sept jours."
    >
      <RunControls />
      <div className="mt-8">
        {error && <Notice tone="error">La simulation a échoué : {error}</Notice>}
        {!run && loading && <Loading label="Simulation de l'année en cours." />}
      </div>
      {run && options && (
        <div className={loading ? "opacity-60 transition-opacity duration-150" : undefined}>
          <div className="grid gap-12 xl:grid-cols-[minmax(0,1fr)_340px]">
            <div className="min-w-0 space-y-12">
              <Section title="TRS, disponibilité et qualité">
                <Chart option={options.trs} height={280} ariaLabel="Évolution du TRS, de la disponibilité et de la qualité" />
              </Section>
              <Section title="Coût cumulé par poste" description="En unités relatives (UR), 1 UR valant le prix d'une pièce catalogue.">
                <Chart option={options.cost} height={300} ariaLabel="Coût cumulé par poste de dépense" />
              </Section>
              <Section title="Machines à l'arrêt" description="Nombre de machines dans chaque état hors service, jour par jour.">
                <Chart option={options.states} height={220} ariaLabel="Machines à l'arrêt par état" />
                <div className="mt-3">
                  <StateLegend />
                </div>
              </Section>
            </div>
            <aside className="space-y-8">
              <div className="rounded-md bg-mist px-5 py-4">
                <h2 className="text-[18px] font-semibold">Bilan de l'année</h2>
                <p className="mt-0.5 text-[13px] text-ink-soft">
                  {run.scenario}, politique {run.policy}
                </p>
                <div className="mt-3">
                  <KpiList
                    items={[
                      { label: "TRS", value: fmtPct(run.kpis.trs), emphasis: true },
                      { label: "Disponibilité", value: fmtPct(run.kpis.disponibilite) },
                      { label: "Taux de défauts", value: fmtPct(run.kpis.taux_defauts, 2) },
                      { label: "Coût total", value: fmtUR(run.kpis.cout_total) },
                      { label: "Ruptures de stock", value: fmtPct(run.kpis.frequence_ruptures), hint: `${fmtInt(run.kpis.ruptures)} sur ${fmtInt(run.kpis.demandes_pieces)} demandes` },
                      { label: "Attente de pièce", value: fmtHours(run.kpis.attente_piece_moyenne), hint: "moyenne par panne" },
                      { label: "Pannes", value: fmtInt(run.kpis.pannes) },
                      { label: "Maintenances préventives", value: fmtInt(run.kpis.maintenances_preventives) },
                      { label: "MTTR", value: fmtHours(run.kpis.mttr) },
                      { label: "Stock immobilisé", value: fmtUR(run.kpis.stock_immobilise), hint: "valeur moyenne" },
                      { label: "Charge des techniciens", value: fmtPct(run.kpis.charge_techniciens, 0) },
                    ]}
                  />
                </div>
              </div>
              <Section title="Postes de coût">
                <CostTable run={run} />
              </Section>
            </aside>
          </div>
        </div>
      )}
    </Page>
  );
}

function CostTable({ run }: { run: RunPayload }) {
  const total = Object.values(run.costs).reduce((a, b) => a + b, 0);
  const rows = Object.entries(run.costs).sort((a, b) => b[1] - a[1]);
  return (
    <table className="w-full text-[14px]">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-line last:border-0">
            <td className="py-2 text-ink-soft">{COST_LABELS[k] ?? k}</td>
            <td className="py-2 text-right tabular">{fmtInt(v)}</td>
            <td className="w-16 py-2 text-right tabular text-ink-soft">{fmtPct(v / total, 0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
