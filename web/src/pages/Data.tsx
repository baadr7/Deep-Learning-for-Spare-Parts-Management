import { useQuery } from "@tanstack/react-query";
import { KpiList, Loading, Notice, Page, Section } from "../components/ui";
import { api } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { fmtInt, fmtNum } from "../lib/format";
import { PALETTE } from "../lib/states";

const FILES: Record<string, string> = {
  telemetry: "PdM_telemetry.csv",
  errors: "PdM_errors.csv",
  maint: "PdM_maint.csv",
  failures: "PdM_failures.csv",
  machines: "PdM_machines.csv",
};

export function Data() {
  const query = useQuery({ queryKey: ["data-summary"], queryFn: api.dataSummary });
  const data = query.data;

  return (
    <Page
      title="Données"
      description="Jeu de données Microsoft Azure Predictive Maintenance : 100 machines suivies heure par heure pendant l'année 2015. Il sert à calibrer le simulateur et à entraîner les modèles de prédiction."
    >
      {query.isLoading && <Loading label="Lecture des fichiers." />}
      {query.error && <Notice tone="error">{(query.error as Error).message}</Notice>}
      {data && !data.complete && (
        <div className="max-w-[70ch] space-y-6">
          <Notice>Les fichiers du jeu de données ne sont pas encore dans le dossier data/raw.</Notice>
          <ol className="list-decimal space-y-2 pl-5 text-[15px]">
            <li>
              Ouvrir la page Kaggle{" "}
              <a
                className="font-medium text-plum-700 underline decoration-plum-200 underline-offset-2 hover:decoration-plum-600"
                href="https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance"
                target="_blank"
                rel="noreferrer"
              >
                Microsoft Azure Predictive Maintenance
              </a>{" "}
              et télécharger l'archive (compte Kaggle requis).
            </li>
            <li>Extraire les cinq fichiers CSV dans le dossier data/raw du dépôt.</li>
            <li>Recharger cette page.</li>
          </ol>
          <table className="w-full text-[14px]">
            <tbody>
              {Object.entries(FILES).map(([key, name]) => (
                <tr key={key} className="border-b border-line">
                  <td className="py-2 font-mono text-[13px]">{name}</td>
                  <td className="py-2 text-right">{data.available[key] ? "présent" : <span className="text-sun-800">manquant</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.complete && (
        <div className="grid gap-12 xl:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-12">
            <Section title="Pannes par mois et par composant">
              <Chart
                height={280}
                ariaLabel="Pannes par mois et par composant"
                option={{
                  ...baseOption(),
                  xAxis: { type: "category", data: data.failures_by_month.months, ...axisStyle },
                  yAxis: { type: "value", minInterval: 1, ...axisStyle },
                  series: Object.entries(data.failures_by_month.series).map(([comp, values], i) => ({
                    name: comp,
                    type: "bar",
                    stack: "pannes",
                    data: values,
                    itemStyle: { color: [PALETTE.plumDark, PALETTE.plum, PALETTE.plumLight, PALETTE.sun][i % 4] },
                  })),
                }}
              />
            </Section>
            <Section title="Âge des machines" description="Nombre de machines par tranche de deux ans.">
              <Chart
                height={220}
                ariaLabel="Histogramme de l'âge des machines"
                option={{
                  ...baseOption(),
                  legend: { show: false },
                  xAxis: {
                    type: "category",
                    data: data.age_histogram.counts.map((_, i) => `${data.age_histogram.edges[i]} à ${data.age_histogram.edges[i + 1]} ans`),
                    ...axisStyle,
                  },
                  yAxis: { type: "value", minInterval: 1, ...axisStyle },
                  series: [{ name: "Machines", type: "bar", data: data.age_histogram.counts, itemStyle: { color: PALETTE.plum } }],
                }}
              />
            </Section>
            <Section title="Signaux de télémétrie">
              <table className="w-full text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2.5 font-medium">Signal</th>
                    <th className="py-2.5 text-right font-medium">Moyenne</th>
                    <th className="py-2.5 text-right font-medium">Écart-type</th>
                    <th className="py-2.5 text-right font-medium">5e centile</th>
                    <th className="py-2.5 text-right font-medium">95e centile</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.signals).map(([name, s]) => (
                    <tr key={name} className="border-b border-line">
                      <td className="py-2.5">{name}</td>
                      <td className="py-2.5 text-right tabular">{fmtNum(s.mean)}</td>
                      <td className="py-2.5 text-right tabular">{fmtNum(s.std)}</td>
                      <td className="py-2.5 text-right tabular">{fmtNum(s.p05)}</td>
                      <td className="py-2.5 text-right tabular">{fmtNum(s.p95)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <aside className="space-y-8">
            <div className="rounded-md bg-mist px-5 py-4">
              <h2 className="text-[18px] font-semibold">Volumes</h2>
              <KpiList
                items={[
                  { label: "Mesures horaires", value: fmtInt(data.rows.telemetry) },
                  { label: "Erreurs", value: fmtInt(data.rows.errors) },
                  { label: "Remplacements", value: fmtInt(data.rows.maint) },
                  { label: "Pannes", value: fmtInt(data.rows.failures), emphasis: true },
                  { label: "Machines", value: fmtInt(data.rows.machines) },
                ]}
              />
            </div>
            <Section title="Pannes par composant">
              <KpiList items={Object.entries(data.failures_by_component).map(([k, v]) => ({ label: k, value: fmtInt(v) }))} />
            </Section>
            <Section title="Machines par modèle">
              <KpiList items={Object.entries(data.machines_by_model).map(([k, v]) => ({ label: k, value: fmtInt(v) }))} />
            </Section>
          </aside>
        </div>
      )}
    </Page>
  );
}
