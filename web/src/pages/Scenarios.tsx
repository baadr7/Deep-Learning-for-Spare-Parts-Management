import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useCatalogues } from "../components/RunControls";
import { Button, Field, Notice, Page, Section, Select } from "../components/ui";
import { api, type SurrogateParams, type WhatIfResponse } from "../lib/api";
import { fmtHours, fmtInt, fmtNum, fmtPct, fmtUR } from "../lib/format";

type Slider = {
  key: keyof SurrogateParams;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: (v: number) => string;
  reference: string;
  policyOnly?: boolean;
};

const SLIDERS: Slider[] = [
  { key: "technicians", label: "Techniciens en poste", min: 1, max: 4, step: 1, unit: (v) => `${v}`, reference: "2" },
  { key: "lead_time_factor", label: "Délais fournisseur", min: 0.5, max: 2.5, step: 0.1, unit: (v) => `× ${fmtNum(v, 1)}`, reference: "× 1" },
  { key: "failure_rate_factor", label: "Taux de panne", min: 0.6, max: 1.8, step: 0.1, unit: (v) => `× ${fmtNum(v, 1)}`, reference: "× 1" },
  { key: "pm_factor", label: "Intervalle de maintenance préventive", min: 0.5, max: 1.6, step: 0.05, unit: (v) => `× ${fmtNum(v, 2)}`, reference: "× 1", policyOnly: true },
  { key: "stock_factor", label: "Niveaux de stock (s et S)", min: 0.5, max: 3, step: 0.1, unit: (v) => `× ${fmtNum(v, 1)}`, reference: "× 1", policyOnly: true },
];

type Row = { key: string; label: string; fmt: (v: number | null) => string; better: "low" | "high" };

const ROWS: Row[] = [
  { key: "cout_total", label: "Coût sur 180 jours", fmt: fmtUR, better: "low" },
  { key: "trs", label: "TRS", fmt: (v) => fmtPct(v), better: "high" },
  { key: "disponibilite", label: "Disponibilité", fmt: (v) => fmtPct(v), better: "high" },
  { key: "frequence_ruptures", label: "Fréquence de ruptures", fmt: (v) => fmtPct(v), better: "low" },
  { key: "attente_piece_moyenne", label: "Attente de pièce par panne", fmt: fmtHours, better: "low" },
  { key: "pannes", label: "Pannes", fmt: fmtInt, better: "low" },
];

const DEFAULTS: SurrogateParams = {
  technicians: 2,
  lead_time_factor: 1,
  failure_rate_factor: 1,
  pm_factor: 1,
  stock_factor: 1,
};

function useDebounced<T>(value: T, delay = 150): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return v;
}

export function Scenarios() {
  const { scenarios, policies } = useCatalogues();
  const [scenario, setScenario] = useState("s1_nominal");
  const [policy, setPolicy] = useState("P1");
  const [values, setValues] = useState<SurrogateParams>(DEFAULTS);
  const debounced = useDebounced(values);

  const status = useQuery({ queryKey: ["surrogate"], queryFn: api.surrogate, staleTime: Infinity });
  const instant = status.data?.available === true;
  const prediction = useQuery({
    queryKey: ["surrogate-predict", debounced],
    queryFn: () => api.surrogatePredict(debounced),
    enabled: instant,
    placeholderData: (prev) => prev,
  });
  const baseline = useQuery({
    queryKey: ["surrogate-predict", DEFAULTS],
    queryFn: () => api.surrogatePredict(DEFAULTS),
    enabled: instant,
    staleTime: Infinity,
  });

  const overrides = {
    technicians: values.technicians,
    lead_time_factor: values.lead_time_factor,
    failure_rate_factor: values.failure_rate_factor,
  };
  const reference = useQuery({
    queryKey: ["whatif-reference", scenario, policy],
    queryFn: () => api.whatif({ scenario, policy, overrides: {} }),
    staleTime: Infinity,
  });
  const estimate = useMutation({ mutationFn: () => api.whatif({ scenario, policy, overrides }) });

  const r2 = status.data?.available ? status.data.metrics.test?.cout_total?.r2 : undefined;

  return (
    <Page
      title="Scénarios"
      description={
        instant
          ? "Faites varier les hypothèses : le modèle de substitution répond aussitôt. Une simulation complète permet ensuite de vérifier le point choisi."
          : "Faites varier les hypothèses et comparez au scénario de référence. Chaque estimation repose sur trois simulations de 180 jours."
      }
    >
      <div className="grid gap-12 lg:grid-cols-[340px_minmax(0,1fr)]">
        <form
          className="space-y-6"
          onSubmit={(e) => {
            e.preventDefault();
            estimate.mutate();
          }}
        >
          {SLIDERS.filter((s) => instant || !s.policyOnly).map((s) => (
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
                <span>référence : {s.reference}</span>
                <span>{s.unit(s.max)}</span>
              </div>
            </div>
          ))}
          <div className="space-y-4 border-t border-line pt-6">
            <Field label="Scénario simulé">
              <Select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </Select>
            </Field>
            <Field
              label="Politique simulée"
              hint={instant ? "Les deux derniers curseurs ne concernent que l'estimation instantanée." : undefined}
            >
              <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {policies.map((p) => (
                  <option key={p.code} value={p.code}>{p.code} · {p.label}</option>
                ))}
              </Select>
            </Field>
            <Button type="submit" variant={instant ? "secondary" : "primary"} disabled={estimate.isPending} className="w-full">
              {estimate.isPending ? "Simulation en cours" : "Vérifier par simulation"}
            </Button>
          </div>
        </form>

        <div className="min-w-0 space-y-12">
          {instant && (
            <Section
              title="Estimation instantanée"
              description={`Politique à seuils fixes, horizon de 180 jours. La marge vaut deux écarts-types entre les cinq réseaux${r2 != null ? ` ; R² du coût sur les scénarios de test : ${fmtNum(r2, 3)}` : ""}.`}
            >
              <table className="w-full text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 pr-4 font-medium">Indicateur</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Référence</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Estimation</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Marge</th>
                    <th className="py-2.5 text-right font-medium">Écart</th>
                  </tr>
                </thead>
                <tbody>
                  {ROWS.map((row) => {
                    const ref = baseline.data?.[row.key]?.mean ?? null;
                    const est = prediction.data?.[row.key];
                    return (
                      <tr key={row.key} className="border-b border-line">
                        <td className="py-2.5 pr-4 text-ink-soft">{row.label}</td>
                        <td className="py-2.5 pr-4 text-right tabular">{row.fmt(ref)}</td>
                        <td className="py-2.5 pr-4 text-right font-medium tabular">{est ? row.fmt(est.mean) : "calcul"}</td>
                        <td className="py-2.5 pr-4 text-right text-ink-soft tabular">{est ? row.fmt(2 * est.std) : ""}</td>
                        <Delta reference={ref} value={est?.mean ?? null} better={row.better} />
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Section>
          )}
          {!instant && status.isFetched && (
            <Notice>
              Le modèle de substitution n'est pas encore entraîné (scripts/train_surrogate.py). En attendant, chaque estimation passe par la simulation.
            </Notice>
          )}
          <Section
            title={instant ? "Vérification par simulation" : "Effet sur les indicateurs"}
            description={`Moyennes sur trois réplications de 180 jours avec la politique ${policy}, mêmes graines pour la référence et le scénario.`}
          >
            {(estimate.error || reference.error) && (
              <Notice tone="error">{((estimate.error ?? reference.error) as Error).message}</Notice>
            )}
            <SimulationTable reference={reference.data} estimate={estimate.data} />
          </Section>
        </div>
      </div>
    </Page>
  );
}

function Delta({ reference, value, better }: { reference: number | null; value: number | null; better: "low" | "high" }) {
  const delta = reference != null && value != null && reference !== 0 ? (value - reference) / Math.abs(reference) : null;
  const good = delta != null && (better === "low" ? delta < 0 : delta > 0);
  const tone = delta == null || Math.abs(delta) < 0.005 ? "text-ink-soft" : good ? "text-plum-700" : "text-danger";
  return (
    <td className={`py-2.5 text-right tabular ${tone}`}>
      {delta == null ? "" : `${delta > 0 ? "+" : ""}${fmtNum(delta * 100, 1)} %`}
    </td>
  );
}

function SimulationTable({ reference, estimate }: { reference?: WhatIfResponse; estimate?: WhatIfResponse }) {
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
          return (
            <tr key={row.key} className="border-b border-line">
              <td className="py-2.5 pr-4 text-ink-soft">{row.label}</td>
              <td className="py-2.5 pr-4 text-right tabular">{reference ? row.fmt(ref) : "calcul"}</td>
              <td className="py-2.5 pr-4 text-right tabular">{estimate ? row.fmt(est) : "à vérifier"}</td>
              <Delta reference={ref} value={est} better={row.better} />
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
