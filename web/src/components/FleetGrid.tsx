import { useNavigate } from "react-router-dom";
import { STATES } from "../lib/states";

type Props = {
  states: number[];
  families: string[];
  familyLabels: Record<string, string>;
  /** Si fourni, un clic sur une case ouvre la fiche de la machine. */
  linkable?: boolean;
};

/** Grille de la flotte : une case par machine, regroupées par famille. */
export function FleetGrid({ states, families, familyLabels, linkable }: Props) {
  const navigate = useNavigate();
  const groups = Array.from(new Set(families));
  return (
    <div className="space-y-6">
      {groups.map((code) => {
        const ids = families.flatMap((f, i) => (f === code ? [i] : []));
        const stopped = ids.filter((i) => states[i] !== 0).length;
        return (
          <div key={code}>
            <div className="mb-2 flex items-baseline justify-between text-[13px]">
              <span className="font-medium text-ink">
                {familyLabels[code] ?? code}
                <span className="ml-2 font-normal text-ink-soft">{ids.length} machines</span>
              </span>
              <span className="tabular text-ink-soft">
                {stopped === 0 ? "toutes en service" : `${stopped} à l'arrêt`}
              </span>
            </div>
            <ol className="grid grid-cols-[repeat(auto-fill,minmax(26px,1fr))] gap-1.5">
              {ids.map((i) => {
                const s = STATES[states[i] ?? 0];
                const label = `Machine ${i + 1}, ${s.label.toLowerCase()}`;
                return (
                  <li key={i}>
                    <button
                      type="button"
                      title={label}
                      aria-label={label}
                      disabled={!linkable}
                      onClick={() => navigate(`/machines/${i}`)}
                      className="flex aspect-square w-full items-center justify-center rounded-sm border font-mono text-[10.5px] leading-none transition-[background-color,border-color] duration-150 ease-out enabled:hover:outline enabled:hover:outline-2 enabled:hover:outline-plum-600 disabled:cursor-default"
                      style={{ background: s.fill, borderColor: s.stroke, color: s.text }}
                    >
                      {i + 1}
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>
        );
      })}
    </div>
  );
}
