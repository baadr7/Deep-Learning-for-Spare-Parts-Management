import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { useMemo, useState } from "react";
import { useCatalogues } from "../components/RunControls";
import { Button, Field, Input, Notice, Page, Section, Select } from "../components/ui";
import { api, type CompareResponse, type Interval } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { fmtInt, fmtNum, fmtPct, fmtUR } from "../lib/format";
import { PALETTE, POLICY_COLORS } from "../lib/states";

type Row = {
  key: string;
  label: string;
  fmt: (v: number | null) => string;
  better: "low" | "high";
  /** Bornes physiques de l'indicateur, appliquées à l'intervalle de confiance. */
  range: [number, number];
};

const hours = (v: number | null) => (v == null ? "n.d." : `${fmtNum(v, 1)} h`);

const ROWS: Row[] = [
  { key: "cout_total", label: "Coût total", fmt: fmtUR, better: "low", range: [0, Infinity] },
  { key: "trs", label: "TRS", fmt: (v) => fmtPct(v), better: "high", range: [0, 1] },
  { key: "disponibilite", label: "Disponibilité", fmt: (v) => fmtPct(v), better: "high", range: [0, 1] },
  { key: "taux_defauts", label: "Taux de défauts", fmt: (v) => fmtPct(v, 2), better: "low", range: [0, 1] },
  { key: "frequence_ruptures", label: "Fréquence de ruptures", fmt: (v) => fmtPct(v), better: "low", range: [0, 1] },
  { key: "attente_piece_moyenne", label: "Attente de pièce par panne", fmt: hours, better: "low", range: [0, Infinity] },
  { key: "pannes", label: "Pannes", fmt: fmtInt, better: "low", range: [0, Infinity] },
  { key: "stock_immobilise", label: "Stock immobilisé", fmt: fmtUR, better: "low", range: [0, Infinity] },
];

function paretoOption(data: CompareResponse) {
  return {
    ...baseOption(),
    grid: { left: 64, right: 40, top: 36, bottom: 44 },
    tooltip: {
      ...baseOption().tooltip,
      trigger: "item",
      formatter: (p: { seriesName: string; value: [number, number] }) =>
        `${p.seriesName}<br/>Coût ${fmtUR(p.value[0])}<br/>TRS ${fmtPct(p.value[1])}`,
    },
    xAxis: {
      type: "value",
      name: "Coût total (UR)",
      nameLocation: "middle",
      nameGap: 28,
      scale: true,
      ...axisStyle,
      axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => fmtInt(v) },
    },
    yAxis: {
      type: "value",
      scale: true,
      ...axisStyle,
      axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => `${(v * 100).toFixed(1)} %` },
    },
    series: Object.entries(data.results).map(([code, r]) => ({
      name: code,
      type: "scatter",
      symbolSize: 9,
      data: r.runs.map((run) => [run.cout_total, run.trs]),
      itemStyle: { color: POLICY_COLORS[code] ?? PALETTE.plum, borderColor: "#fff", borderWidth: 1 },
    })),
  };
}

function bestCode(data: CompareResponse, key: string, better: "low" | "high") {
  const entries = Object.entries(data.results).filter(([code]) => code !== "Oracle");
  const values = entries.map(([code, r]) => [code, r.summary[key]?.mean ?? NaN] as const);
  const sorted = values.filter(([, v]) => Number.isFinite(v)).sort((a, b) => (better === "low" ? a[1] - b[1] : b[1] - a[1]));
  return sorted[0]?.[0];
}

export function Compare() {
  const { scenarios, policies } = useCatalogues();
  const [scenario, setScenario] = useState("s1_nominal");
  const [selected, setSelected] = useState<string[]>(["P0", "P1", "P2", "Oracle"]);
  const [replications, setReplications] = useState(5);
  const mutation = useMutation({
    // Colonnes dans l'ordre du catalogue, quel que soit l'ordre des clics.
    mutationFn: () =>
      api.compare({ scenario, policies: policies.map((p) => p.code).filter((c) => selected.includes(c)), replications }),
  });
  const data = mutation.data;
  const pareto = useMemo(() => (data ? paretoOption(data) : null), [data]);

  return (
    <Page
      title="Comparaison des politiques"
      description="Chaque politique est simulée sur les mêmes graines (nombres aléatoires communs). Les valeurs sont des moyennes avec leur intervalle de confiance à 95 %."
    >
      <form
        className="flex flex-wrap items-end gap-6"
        onSubmit={(e) => {
          e.preventDefault();
          mutation.mutate();
        }}
      >
        <Field label="Scénario">
          <Select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </Select>
        </Field>
        <fieldset>
          <legend className="mb-1.5 text-[13px] font-medium text-ink-soft">Politiques</legend>
          <div className="flex flex-wrap gap-2">
            {policies.map((p) => {
              const on = selected.includes(p.code);
              return (
                <label
                  key={p.code}
                  title={p.label}
                  className={clsx(
                    "flex h-10 cursor-pointer items-center gap-2 rounded-sm border px-3 text-[14px] transition-colors duration-150 ease-out",
                    on ? "border-plum-600 bg-plum-50 text-plum-900" : "border-line-strong text-ink-soft hover:border-plum-400",
                  )}
                >
                  <input
                    type="checkbox"
                    className="accent-plum-600"
                    checked={on}
                    onChange={() => setSelected(on ? selected.filter((c) => c !== p.code) : [...selected, p.code])}
                  />
                  {p.code}
                </label>
              );
            })}
          </div>
        </fieldset>
        <Field label="Réplications" hint="Une année chacune">
          <Input type="number" min={3} max={30} value={replications} onChange={(e) => setReplications(Number(e.target.value))} className="w-24" />
        </Field>
        <Button type="submit" disabled={mutation.isPending || selected.length === 0}>
          {mutation.isPending ? "Comparaison en cours" : "Comparer"}
        </Button>
      </form>

      <div className="mt-10 space-y-12">
        {mutation.error && <Notice tone="error">{(mutation.error as Error).message}</Notice>}
        {!data && !mutation.isPending && (
          <div className="rounded-md bg-mist px-6 py-10 text-center text-ink-soft">
            Lancez une comparaison pour obtenir le tableau des indicateurs et le front coût, TRS.
          </div>
        )}
        {mutation.isPending && (
          <p className="text-ink-soft">
            Simulation de {selected.length * replications} années en parallèle, comptez quelques secondes par réplication.
          </p>
        )}
        {data && (
          <>
            <Section title="Indicateurs" description="La meilleure politique réaliste (hors oracle) est surlignée pour chaque ligne.">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-[14px]">
                  <thead>
                    <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                      <th className="py-2.5 pr-4 font-medium">Indicateur</th>
                      {Object.entries(data.results).map(([code, r]) => (
                        <th key={code} className="py-2.5 pr-4 text-right font-medium" title={r.label}>{code}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ROWS.map((row) => {
                      const best = bestCode(data, row.key, row.better);
                      return (
                        <tr key={row.key} className="border-b border-line">
                          <td className="py-2.5 pr-4 text-ink-soft">{row.label}</td>
                          {Object.entries(data.results).map(([code, r]) => (
                            <td key={code} className="py-2.5 pr-4 text-right">
                              <Cell interval={r.summary[row.key]} fmt={row.fmt} range={row.range} best={code === best} />
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Section>
            <Section title="Coût et TRS par réplication" description="Un point par année simulée, coût total en abscisse et TRS en ordonnée. En haut à gauche : moins cher et plus productif.">
              {pareto && <Chart option={pareto} height={340} ariaLabel="Nuage coût et TRS par politique" />}
            </Section>
          </>
        )}
      </div>
    </Page>
  );
}

function Cell({ interval, fmt, range, best }: { interval?: Interval; fmt: (v: number | null) => string; range: [number, number]; best: boolean }) {
  if (!interval) return <span className="text-ink-soft">n.d.</span>;
  return (
    <span className={clsx("inline-block rounded-sm px-1.5 py-0.5", best && "bg-sun-200")}>
      <span className={clsx("tabular", best ? "text-sun-800 font-medium" : "text-ink")}>{fmt(interval.mean)}</span>
      <span className="block text-[12px] text-ink-soft tabular">
        {fmt(clamp(interval.low, range))} à {fmt(clamp(interval.high, range))}
      </span>
    </span>
  );
}

function clamp(v: number | null, [lo, hi]: [number, number]): number | null {
  return v == null ? null : Math.min(Math.max(v, lo), hi);
}
