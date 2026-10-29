import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useCatalogues } from "../components/RunControls";
import { Button, Field, Notice, Page, Section, Select } from "../components/ui";
import { api, type Overrides, type WhatIfResponse } from "../lib/api";
import { fmtHours, fmtInt, fmtNum, fmtPct, fmtUR } from "../lib/format";

type Slider = { key: keyof Overrides; label: string; min: number; max: number; step: number; unit: (v: number) => string };

const SLIDERS: Slider[] = [
  { key: "technicians", label: "Techniciens en poste", min: 1, max: 6, step: 1, unit: (v) => `${v}` },
  { key: "lead_time_factor", label: "Délais fournisseur", min: 0.5, max: 3, step: 0.25, unit: (v) => `× ${fmtNum(v, 2)}` },
  { key: "failure_rate_factor", label: "Taux de panne", min: 0.5, max: 2, step: 0.1, unit: (v) => `× ${fmtNum(v, 1)}` },
];

const ROWS: { key: string; label: string; fmt: (v: number | null) => string; better: "low" | "high" }[] = [
  { key: "cout_total", label: "Coût sur 180 jours", fmt: fmtUR, better: "low" },
  { key: "trs", label: "TRS", fmt: (v) => fmtPct(v), better: "high" },
  { key: "disponibilite", label: "Disponibilité", fmt: (v) => fmtPct(v), better: "high" },
  { key: "frequence_ruptures", label: "Fréquence de ruptures", fmt: (v) => fmtPct(v), better: "low" },
  { key: "attente_piece_moyenne", label: "Attente de pièce par panne", fmt: fmtHours, better: "low" },
  { key: "attente_technicien_moyenne", label: "Attente de technicien par panne", fmt: fmtHours, better: "low" },
  { key: "pannes", label: "Pannes", fmt: fmtInt, better: "low" },
];

export function Scenarios() {
  const { scenarios, policies } = useCatalogues();
  const [scenario, setScenario] = useState("s1_nominal");
  const [policy, setPolicy] = useState("P1");
  const nominalTechs = scenarios.find((s) => s.id === scenario)?.technicians ?? 2;
  const [values, setValues] = useState<Required<Overrides>>({ technicians: 2, lead_time_factor: 1, failure_rate_factor: 1 });

  const reference = useQuery({
    queryKey: ["whatif-reference", scenario, policy],
    queryFn: () => api.whatif({ scenario, policy, overrides: {} }),
    staleTime: Infinity,
  });
  const estimate = useMutation({ mutationFn: () => api.whatif({ scenario, policy, overrides: values }) });

  return (
    <Page
      title="Scénarios"
      description="Faites varier les hypothèses et comparez au scénario de référence. Chaque estimation repose sur trois simulations de 180 jours ; le modèle de substitution rendra la réponse instantanée une fois entraîné."
    >
      <div className="grid gap-12 lg:grid-cols-[340px_minmax(0,1fr)]">
        <form
          className="space-y-6"
          onSubmit={(e) => {
            e.preventDefault();
            estimate.mutate();
          }}
        >
          <Field label="Scénario">
            <Select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </Select>
          </Field>
          <Field label="Politique">
            <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {policies.map((p) => (
                <option key={p.code} value={p.code}>{p.code} · {p.label}</option>
              ))}
            </Select>
          </Field>
          {SLIDERS.map((s) => (
            <div key={s.key}>
              <div className="flex items-baseline justify-between text-[13px]">
                <label htmlFor={s.key} className="font-medium text-ink-soft">{s.label}</label>
                <span className="text-[14px] font-medium text-ink tabular">{s.unit(values[s.key])}</span>
              </div>
              <input
                id={s.key}
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={values[s.key]}
                onChange={(e) => setValues({ ...values, [s.key]: Number(e.target.value) })}
                className="mt-2 w-full accent-plum-600"
              />
              <div className="flex justify-between text-[12px] text-ink-soft tabular">
                <span>{s.unit(s.min)}</span>
                <span>référence : {s.key === "technicians" ? nominalTechs : "× 1"}</span>
                <span>{s.unit(s.max)}</span>
              </div>
            </div>
          ))}
          <Button type="submit" disabled={estimate.isPending} className="w-full">
            {estimate.isPending ? "Estimation en cours" : "Estimer ce scénario"}
          </Button>
        </form>

        <Section title="Effet sur les indicateurs" description="Moyennes sur trois réplications de 180 jours, mêmes graines pour la référence et le scénario.">
          {(estimate.error || reference.error) && (
            <Notice tone="error">{((estimate.error ?? reference.error) as Error).message}</Notice>
          )}
          <ResultTable reference={reference.data} estimate={estimate.data} />
        </Section>
      </div>
    </Page>
  );
}

function ResultTable({ reference, estimate }: { reference?: WhatIfResponse; estimate?: WhatIfResponse }) {
  return (
    <table className="w-full text-[14px]">
      <thead>
        <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
          <th className="py-2.5 pr-4 font-medium">Indicateur</th>
          <th className="py-2.5 pr-4 text-right font-medium">Référence</th>
          <th className="py-2.5 pr-4 text-right font-medium">Scénario</th>
          <th className="py-2.5 text-right font-medium">Écart</th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map((row) => {
          const ref = reference?.summary[row.key]?.mean ?? null;
          const est = estimate?.summary[row.key]?.mean ?? null;
          const delta = ref != null && est != null && ref !== 0 ? (est - ref) / Math.abs(ref) : null;
          const good = delta != null && (row.better === "low" ? delta < 0 : delta > 0);
          return (
            <tr key={row.key} className="border-b border-line">
              <td className="py-2.5 pr-4 text-ink-soft">{row.label}</td>
              <td className="py-2.5 pr-4 text-right tabular">{reference ? row.fmt(ref) : "calcul"}</td>
              <td className="py-2.5 pr-4 text-right tabular">{estimate ? row.fmt(est) : "à estimer"}</td>
              <td className={`py-2.5 text-right tabular ${delta == null || Math.abs(delta) < 0.005 ? "text-ink-soft" : good ? "text-plum-700" : "text-danger"}`}>
                {delta == null ? "" : `${delta > 0 ? "+" : ""}${fmtNum(delta * 100, 1)} %`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
