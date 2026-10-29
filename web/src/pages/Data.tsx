import { useQuery } from "@tanstack/react-query";
import { KpiList, Loading, Notice, Page, Section } from "../components/ui";
import { api, type Calibration } from "../lib/api";
import { axisStyle, baseOption, Chart } from "../lib/chart";
import { fmtInt, fmtNum } from "../lib/format";

const MODEL_LABELS: Record<string, string> = { lightgbm: "LightGBM", gru: "GRU", tcn: "TCN", lstm: "LSTM" };
const SOURCE_LABELS: Record<string, string> = {
  simulation: "Données simulées (cinq années, test sur une année non vue)",
  azure: "Données réelles Azure (test sur novembre et décembre 2015)",
};

function PredictionSection() {
  const query = useQuery({ queryKey: ["prediction-metrics"], queryFn: api.predictionMetrics });
  const sources = Object.entries(query.data ?? {});
  if (!sources.length) return null;
  return (
    <Section
      title="Modèles de prédiction"
      description="Panne d'un composant dans les 24 heures puis dans les 7 jours. Le seuil de décision est choisi sur la période de validation. L'aire sous la courbe précision-rappel se compare à la proportion de pannes, rappelée entre parenthèses."
    >
      <div className="space-y-8">
        {sources.map(([source, models]) => {
          const any = Object.values(models)[0];
          const base24 = any ? any.positifs / any.exemples : 0;
          const base7 = any?.a_7_jours ? any.a_7_jours.positifs / any.a_7_jours.exemples : 0;
          return (
            <div key={source}>
              <h3 className="mb-2 font-sans text-[14px] font-medium text-ink">{SOURCE_LABELS[source] ?? source}</h3>
              <table className="w-full text-[14px]">
                <thead>
                  <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
                    <th className="py-2 font-medium">Modèle</th>
                    <th className="py-2 text-right font-medium">Rappel 24 h</th>
                    <th className="py-2 text-right font-medium">F1 24 h</th>
                    <th className="py-2 text-right font-medium">PR-AUC 24 h ({fmtNum(100 * base24, 2)} %)</th>
                    <th className="py-2 text-right font-medium">F1 7 j</th>
                    <th className="py-2 text-right font-medium">PR-AUC 7 j ({fmtNum(100 * base7, 1)} %)</th>
                    <th className="py-2 text-right font-medium">C-index</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(models).map(([name, m]) => (
                    <tr key={name} className="border-b border-line">
                      <td className="py-2">{MODEL_LABELS[name] ?? name}</td>
                      <td className="py-2 text-right tabular">{fmtNum(m.rappel, 2)}</td>
                      <td className="py-2 text-right tabular">{fmtNum(m.f1, 2)}</td>
                      <td className="py-2 text-right tabular">{fmtNum(m.pr_auc, 3)}</td>
                      <td className="py-2 text-right tabular">{m.a_7_jours ? fmtNum(m.a_7_jours.f1, 2) : "n.d."}</td>
                      <td className="py-2 text-right tabular">{m.a_7_jours ? fmtNum(m.a_7_jours.pr_auc, 3) : "n.d."}</td>
                      <td className="py-2 text-right tabular">{m.c_index_7j != null ? fmtNum(m.c_index_7j, 3) : "n.d."}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function CalibrationSection({ calibration }: { calibration: Calibration }) {
  const v = calibration.validation;
  const year = v?.annee_2015;
  return (
    <Section
      title="Lois de durée de vie"
      description="Lois de Weibull ajustées sur les durées entre remplacements, les remplacements planifiés étant traités comme des observations censurées. Un paramètre de forme supérieur à 1 traduit une usure : la maintenance préventive a alors un intérêt."
    >
      <table className="w-full text-[14px]">
        <thead>
          <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
            <th className="py-2.5 font-medium">Composant</th>
            <th className="py-2.5 text-right font-medium">Forme β</th>
            <th className="py-2.5 text-right font-medium">Échelle ajustée</th>
            {v && <th className="py-2.5 text-right font-medium">Échelle corrigée</th>}
            {year && <th className="py-2.5 text-right font-medium">Pannes 2015, Azure</th>}
            {year && <th className="py-2.5 text-right font-medium">Simulées</th>}
          </tr>
        </thead>
        <tbody>
          {Object.entries(calibration.lois).map(([comp, law]) => {
            const y = year?.[comp];
            return (
              <tr key={comp} className="border-b border-line">
                <td className="py-2.5">{comp}</td>
                <td className="py-2.5 text-right tabular">{fmtNum(law.beta, 2)}</td>
                <td className="py-2.5 text-right tabular">{fmtNum(law.eta / 24, 0)} j</td>
                {v && <td className="py-2.5 text-right tabular">{fmtNum(v.lois_corrigees[comp].eta / 24, 0)} j</td>}
                {y && <td className="py-2.5 text-right tabular">{fmtInt(y.azure)}</td>}
                {y && (
                  <td className="py-2.5 text-right tabular">
                    {fmtInt(y.simulees)}
                    <span className="ml-2 text-ink-soft">
                      {y.ecart_relatif > 0 ? "+" : ""}
                      {fmtNum(100 * y.ecart_relatif, 1)} %
                    </span>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {v && (
        <p className="mt-3 max-w-[70ch] text-[13px] text-ink-soft">
          Rejouée dans le simulateur, la maintenance planifiée d'Azure donne environ 30 % de pannes de moins que la réalité
          avec les lois ajustées directement : les remplacements planifiés visent sans doute des composants déjà usés. L'échelle
          est donc corrigée par simulation sur le premier semestre 2015.{" "}
          {year && (v.critere_atteint ? "Sur l'année, l'écart reste inférieur à 10 % pour chaque composant." : "Sur l'année, l'écart dépasse 10 % pour au moins un composant.")}{" "}
          Sur le seul second semestre, que la correction n'a pas vu, l'écart maximal atteint{" "}
          {fmtNum(100 * Math.max(...Object.values(v.validation_s2).map((r) => Math.abs(r.ecart_relatif))), 0)} %, du même ordre
          que la variation des pannes réelles d'un semestre à l'autre.
        </p>
      )}
    </Section>
  );
}
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
            {data.calibration && <CalibrationSection calibration={data.calibration} />}
            <PredictionSection />
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
