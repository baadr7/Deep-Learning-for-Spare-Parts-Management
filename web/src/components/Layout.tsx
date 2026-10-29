import clsx from "clsx";
import {
  ChartColumnBig,
  Cog,
  Database,
  Gauge,
  Package,
  Radio,
  SlidersHorizontal,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useRun } from "../lib/run";

const NAV = [
  { to: "/", label: "Tableau de bord", icon: Gauge, end: true },
  { to: "/direct", label: "Flotte en direct", icon: Radio },
  { to: "/machines", label: "Machines", icon: Cog },
  { to: "/stock", label: "Stock", icon: Package },
  { to: "/comparaison", label: "Comparaison", icon: ChartColumnBig },
  { to: "/scenarios", label: "Scénarios", icon: SlidersHorizontal },
  { to: "/donnees", label: "Données", icon: Database },
];

export function Layout() {
  const { run, loading } = useRun();
  return (
    <div className="flex min-h-screen">
      <div className="w-[244px] shrink-0 border-r border-line bg-mist">
      <aside className="sticky top-0 flex h-screen flex-col">
        <div className="px-6 pt-7 pb-8">
          <p className="font-display text-[19px] leading-tight font-semibold text-plum-900">
            Pilotage flotte
          </p>
          <p className="mt-1 text-[13px] text-ink-soft">Maintenance et pièces de rechange</p>
        </div>
        <nav aria-label="Navigation principale" className="flex-1 px-3">
          <ul className="space-y-0.5">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    clsx(
                      "flex items-center gap-3 rounded-sm px-3 py-2 text-[14px] transition-colors duration-150 ease-out",
                      isActive
                        ? "bg-plum-100 font-medium text-plum-900"
                        : "text-ink-soft hover:bg-plum-50 hover:text-ink",
                    )
                  }
                >
                  <Icon aria-hidden size={17} strokeWidth={1.75} />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t border-line px-6 py-5 text-[13px] text-ink-soft">
          <p className="font-medium text-ink">Simulation de référence</p>
          {loading ? (
            <p className="mt-1">Calcul en cours</p>
          ) : run ? (
            <p className="mt-1 tabular">
              {run.scenario}, politique {run.policy}, graine {run.seed}
            </p>
          ) : (
            <p className="mt-1">Aucune</p>
          )}
        </div>
      </aside>
      </div>
      <main className="min-w-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}
