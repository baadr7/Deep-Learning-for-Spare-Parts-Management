import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loading, Page, Select, Field } from "../components/ui";
import { fmtInt, fmtNum } from "../lib/format";
import { useRun } from "../lib/run";
import { STATES } from "../lib/states";

type SortKey = "id" | "failures" | "down_hours" | "age_years";

export function Machines() {
  const { run } = useRun();
  const navigate = useNavigate();
  const [family, setFamily] = useState("all");
  const [sort, setSort] = useState<SortKey>("down_hours");

  const rows = useMemo(() => {
    if (!run) return [];
    const list = run.machines.filter((m) => family === "all" || m.family === family);
    return [...list].sort((a, b) => (sort === "id" ? a.id - b.id : (b[sort] as number) - (a[sort] as number)));
  }, [run, family, sort]);

  if (!run) {
    return (
      <Page title="Machines">
        <Loading label="Simulation de référence en cours." />
      </Page>
    );
  }

  return (
    <Page
      title="Machines"
      description={`Les ${run.machines.length} machines de la simulation de référence (${run.scenario}, politique ${run.policy}). Cliquez sur une ligne pour ouvrir la fiche.`}
    >
      <div className="mb-5 flex flex-wrap gap-4">
        <Field label="Famille">
          <Select value={family} onChange={(e) => setFamily(e.target.value)}>
            <option value="all">Toutes</option>
            {Object.entries(run.meta.family_labels).map(([code, label]) => (
              <option key={code} value={code}>{label}</option>
            ))}
          </Select>
        </Field>
        <Field label="Trier par">
          <Select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="down_hours">Heures d'arrêt</option>
            <option value="failures">Nombre de pannes</option>
            <option value="age_years">Âge</option>
            <option value="id">Numéro</option>
          </Select>
        </Field>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-[14px]">
          <thead>
            <tr className="border-b border-line-strong text-left text-[13px] text-ink-soft">
              <th className="py-2.5 pr-4 font-medium">Machine</th>
              <th className="py-2.5 pr-4 font-medium">Famille</th>
              <th className="py-2.5 pr-4 text-right font-medium">Postes</th>
              <th className="py-2.5 pr-4 text-right font-medium">Âge</th>
              <th className="py-2.5 pr-4 text-right font-medium">Pannes</th>
              <th className="py-2.5 pr-4 text-right font-medium">Préventives</th>
              <th className="py-2.5 pr-4 text-right font-medium">Heures d'arrêt</th>
              <th className="py-2.5 font-medium">État au dernier jour</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const s = STATES[m.final_state];
              return (
                <tr
                  key={m.id}
                  tabIndex={0}
                  onClick={() => navigate(`/machines/${m.id}`)}
                  onKeyDown={(e) => e.key === "Enter" && navigate(`/machines/${m.id}`)}
                  className="cursor-pointer border-b border-line transition-colors duration-150 ease-out hover:bg-plum-50"
                >
                  <td className="py-2.5 pr-4 tabular">{m.id + 1}</td>
                  <td className="py-2.5 pr-4">{run.meta.family_labels[m.family]}</td>
                  <td className="py-2.5 pr-4 text-right tabular">{m.shifts}</td>
                  <td className="py-2.5 pr-4 text-right tabular">{fmtNum(m.age_years)} ans</td>
                  <td className="py-2.5 pr-4 text-right tabular">{m.failures}</td>
                  <td className="py-2.5 pr-4 text-right tabular">{m.preventive}</td>
                  <td className="py-2.5 pr-4 text-right tabular">{fmtInt(m.down_hours)}</td>
                  <td className="py-2.5">
                    <span className="inline-flex items-center gap-2">
                      <span aria-hidden className="size-2.5 rounded-[3px] border" style={{ background: s.fill, borderColor: s.stroke }} />
                      {s.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Page>
  );
}
