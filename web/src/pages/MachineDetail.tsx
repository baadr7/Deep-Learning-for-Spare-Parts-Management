import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { StateLegend } from "../components/StateLegend";
import { KpiList, Loading, Notice, Page, Section } from "../components/ui";
import { api, type MachineDetail as Detail } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { eventSentence, fmtNum, fmtPct } from "../lib/format";
import { useRun } from "../lib/run";
import { PALETTE, STATES } from "../lib/states";

const SIGNAL_LABELS: Record<string, string> = {
  volt: "Tension (V)",
  rotate: "Vitesse de rotation (tr/min)",
  pressure: "Pression",
  vibration: "Vibration",
};

function signalOption(detail: Detail, index: number) {
  const values = detail.signal_mean.map((row) => row[index]);
  return {
    ...baseOption(),
    legend: { show: false },
    grid: { left: 48, right: 8, top: 12, bottom: 24 },
    xAxis: { type: "category", data: values.map((_, i) => i + 1), ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 59 } },
    yAxis: { type: "value", scale: true, ...axisStyle },
    tooltip: { ...baseOption().tooltip, valueFormatter: (v: number) => fmtNum(v) },
    series: [
      {
        name: SIGNAL_LABELS[detail.signals[index]],
        type: "line",
        data: values,
        showSymbol: false,
        lineStyle: { width: 1.5, color: PALETTE.plum },
        itemStyle: { color: PALETTE.plum },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: PALETTE.danger, type: "dotted", width: 1 },
          label: { show: false },
          data: detail.events.filter((e) => e.type === "panne").map((e) => ({ xAxis: e.day })),
        },
      },
    ],
  };
}

function defectOption(detail: Detail) {
  return {
    ...baseOption(),
    legend: { show: false },
    grid: { left: 48, right: 8, top: 12, bottom: 24 },
    xAxis: { type: "category", data: detail.defect_rate.map((_, i) => i + 1), ...axisStyle, axisLabel: { ...axisStyle.axisLabel, interval: 59 } },
    yAxis: { type: "value", ...axisStyle, axisLabel: { ...axisStyle.axisLabel, formatter: (v: number) => `${Math.round(v * 1000) / 10} %` } },
    tooltip: { ...baseOption().tooltip, valueFormatter: (v: number) => fmtPct(v, 2) },
    series: [
      { name: "Taux de défauts observé", type: "bar", data: detail.defect_rate, barCategoryGap: "20%", itemStyle: { color: PALETTE.sun } },
    ],
  };
}

export function MachineDetail() {
  const { id } = useParams();
  const machine = Number(id);
  const { run } = useRun();
  const query = useQuery({
    queryKey: ["machine", run?.run_id, machine],
    queryFn: () => api.machine(run!.run_id, machine),
    enabled: !!run && Number.isFinite(machine),
  });
  const detail = query.data;
  const options = useMemo(
    () => (detail ? { signals: detail.signals.map((_, i) => signalOption(detail, i)), defects: defectOption(detail) } : null),
    [detail],
  );

  const back = (
    <Link to="/machines" className="inline-flex items-center gap-2 rounded-sm px-2 py-1 text-[14px] font-medium text-plum-700 hover:bg-plum-50">
      <ArrowLeft aria-hidden size={16} /> Toutes les machines
    </Link>
  );

  if (!run || query.isLoading) {
    return (
      <Page title={`Machine ${machine + 1}`} actions={back}>
        <Loading label="Chargement de la fiche." />
      </Page>
    );
  }
  if (query.error || !detail || !options) {
    return (
      <Page title={`Machine ${machine + 1}`} actions={back}>
        <Notice tone="error">Fiche indisponible : {(query.error as Error)?.message ?? "machine inconnue"}</Notice>
      </Page>
    );
  }

  const row = run.machines[machine];
  const parts = run.meta.parts;
  return (
    <Page
      title={`Machine ${machine + 1}`}
      description={`${run.meta.family_labels[detail.family]}, ${detail.shifts} poste${detail.shifts > 1 ? "s" : ""} par jour, ${fmtNum(detail.age_years)} ans de service.`}
      actions={back}
    >
      <div className="grid gap-12 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0 space-y-12">
          <Section title="États sur l'année" description="Une colonne par jour : l'état hors service le plus long de la journée, s'il a duré au moins une heure.">
            <div className="flex h-9 overflow-hidden rounded-sm" role="img" aria-label="Frise des états de la machine">
              {detail.states.map((s, i) => (
                <div key={i} className="h-full flex-1" style={{ background: STATES[s].fill }} title={`Jour ${i + 1} : ${STATES[s].label}`} />
              ))}
            </div>
            <div className="mt-3">
              <StateLegend />
            </div>
          </Section>
          <Section title="Télémétrie" description="Moyennes journalières des heures de marche. Les pointillés rouges marquent les pannes.">
            <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
              {options.signals.map((opt, i) => (
                <div key={detail.signals[i]}>
                  <h3 className="font-sans text-[13px] font-medium text-ink-soft">{SIGNAL_LABELS[detail.signals[i]]}</h3>
                  <Chart option={opt} height={170} ariaLabel={SIGNAL_LABELS[detail.signals[i]]} />
                </div>
              ))}
            </div>
          </Section>
          <Section title="Taux de défauts observé" description="Proportion de pièces défectueuses dans l'échantillon contrôlé chaque jour.">
            <Chart option={options.defects} height={180} ariaLabel="Taux de défauts observé par jour" />
          </Section>
        </div>
        <aside className="space-y-10">
          <div className="rounded-md bg-mist px-5 py-4">
            <h2 className="text-[18px] font-semibold">Sur l'année</h2>
            <KpiList
              items={[
                { label: "Pannes", value: String(row.failures), emphasis: true },
                { label: "Maintenances préventives", value: String(row.preventive) },
                { label: "Heures d'arrêt", value: `${Math.round(detail.down_hours)} h`, hint: `${row.down_days} jours touchés` },
                { label: "Alarmes", value: String(detail.alarms.reduce((a, b) => a + b, 0)) },
              ]}
            />
          </div>
          <Section title="Historique">
            <ol className="max-h-[520px] space-y-2 overflow-y-auto text-[14px]">
              {detail.events.length === 0 && <li className="text-ink-soft">Aucun événement.</li>}
              {detail.events.map((e, i) => (
                <li key={i} className="flex gap-3">
                  <span className="w-12 shrink-0 font-mono text-[13px] text-ink-soft tabular">J{e.day + 1}</span>
                  <span className={e.type === "panne" ? "text-danger" : undefined}>
                    {eventSentence(e, detail.components, parts)}
                  </span>
                </li>
              ))}
            </ol>
          </Section>
        </aside>
      </div>
    </Page>
  );
}
