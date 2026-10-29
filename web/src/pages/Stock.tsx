import { useMemo } from "react";
import { Loading, Page, Section } from "../components/ui";
import type { RunPayload } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { fmtInt, fmtPct } from "../lib/format";
import { useRun } from "../lib/run";
import { PALETTE } from "../lib/states";

function partOption(run: RunPayload, p: number, s: number, S: number) {
  const x = run.daily.stock.map((_, i) => i + 1);
  return {
    ...baseOption(),
    grid: { left: 40, right: 12, top: 30, bottom: 24 },
    xAxis: { type: "category", data: x, ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 59 } },
    yAxis: { type: "value", minInterval: 1, ...axisStyle },
    series: [
      {
        name: "En stock",
        type: "line",
        step: "end",
        data: run.daily.stock.map((r) => r[p]),
        showSymbol: false,
        lineStyle: { width: 1.75, color: PALETTE.plum },
        itemStyle: { color: PALETTE.plum },
        areaStyle: { color: PALETTE.plumWash },
        markLine: {
          symbol: "none",
          silent: true,
          label: { position: "insideEndTop", color: PALETTE.inkSoft, fontSize: 11 },
          data: [
            { yAxis: s, name: "s", lineStyle: { color: PALETTE.sunDark, type: "dashed" }, label: { formatter: `s = ${s}` } },
            { yAxis: S, name: "S", lineStyle: { color: PALETTE.grey, type: "dashed" }, label: { formatter: `S = ${S}` } },
          ],
        },
      },
      {
        name: "En commande",
        type: "line",
        step: "end",
        data: run.daily.on_order.map((r) => r[p]),
        showSymbol: false,
        lineStyle: { width: 1.25, color: PALETTE.plumLight, type: "dashed" },
        itemStyle: { color: PALETTE.plumLight },
      },
      {
        name: "Machines en attente",
        type: "bar",
        data: run.daily.backorders.map((r) => r[p] || null),
        itemStyle: { color: PALETTE.sun },
        barWidth: 2,
      },
    ],
  };
}

export function Stock() {
  const { run } = useRun();
  const options = useMemo(() => {
    if (!run) return [];
    const own = run.meta.policy_params;
    return run.meta.parts.map((part, p) => partOption(run, p, own?.s[p] ?? part.s, own?.S[p] ?? part.S));
  }, [run]);
  if (!run) {
    return (
      <Page title="Stock">
        <Loading label="Simulation de référence en cours." />
      </Page>
    );
  }
  const policyNote = run.meta.policy_params
    ? `Les seuils s et S tracés sont ceux de la politique ${run.policy}.`
    : `La politique ${run.policy} ne fixe pas de seuils ; ceux du scénario sont tracés à titre de repère.`;

  return (
    <Page
      title="Stock"
      description={`Niveau de stock, commandes en cours et machines en attente pour chaque référence, sur l'année simulée. ${policyNote}`}
    >
      <table className="mb-12 w-full text-[14px]">
        <thead>
          <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
            <th className="py-2.5 pr-4 font-medium">Référence</th>
            <th className="py-2.5 pr-4 font-medium">Nature</th>
            <th className="py-2.5 pr-4 text-right font-medium">Prix</th>
            <th className="py-2.5 pr-4 text-right font-medium">Délai normal</th>
            <th className="py-2.5 pr-4 text-right font-medium">Demandes sur panne</th>
            <th className="py-2.5 pr-4 text-right font-medium">Ruptures</th>
            <th className="py-2.5 text-right font-medium">Taux de service</th>
          </tr>
        </thead>
        <tbody>
          {run.meta.parts.map((part, p) => {
            const demands = run.meta.demands_by_part[p];
            const stockouts = run.meta.stockouts_by_part[p];
            return (
              <tr key={part.code} className="border-b border-line">
                <td className="py-2.5 pr-4">
                  <span className="font-medium">{part.code}</span> <span className="text-ink-soft">{part.name}</span>
                </td>
                <td className="py-2.5 pr-4 text-ink-soft">{part.nature}</td>
                <td className="py-2.5 pr-4 text-right tabular">{fmtInt(part.unit_cost)} UR</td>
                <td className="py-2.5 pr-4 text-right tabular">{part.lead_days} j</td>
                <td className="py-2.5 pr-4 text-right tabular">{demands}</td>
                <td className={`py-2.5 pr-4 text-right tabular ${stockouts > 0 ? "text-sun-800" : ""}`}>{stockouts}</td>
                <td className="py-2.5 text-right tabular">{demands ? fmtPct(1 - stockouts / demands) : "n.d."}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="grid gap-x-10 gap-y-10 lg:grid-cols-2">
        {run.meta.parts.map((part, p) => (
          <Section key={part.code} title={`${part.code} · ${part.name}`}>
            <Chart option={options[p]} height={220} ariaLabel={`Stock de ${part.name} sur l'année`} />
          </Section>
        ))}
      </div>
    </Page>
  );
}
