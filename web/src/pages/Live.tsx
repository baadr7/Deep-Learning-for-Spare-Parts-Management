import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { FleetGrid } from "../components/FleetGrid";
import { useCatalogues } from "../components/RunControls";
import { StateLegend } from "../components/StateLegend";
import { Button, Field, KpiList, Notice, Page, Section, Select } from "../components/ui";
import { liveSocketUrl, type Part, type SimEvent } from "../lib/api";
import { eventSentence, fmtInt, fmtPct, fmtUR } from "../lib/format";
import { useRun } from "../lib/run";

type Frame = {
  type: "day";
  day: number;
  states: number[];
  stock: number[];
  on_order: number[];
  backorders: number[];
  techs_busy: number;
  techs_queue: number;
  trs: number;
  availability: number;
  cost_total: number;
  failures: number;
  stockouts: number;
  events: SimEvent[];
};

type Start = { type: "start"; days: number; machines: number; families: string[]; parts: Part[]; technicians: number };
type Status = "idle" | "running" | "paused" | "done" | "error";

const COMPONENTS = ["Roulement", "Joints", "Pièce usinée", "Moteur"];
const SPEEDS = [
  { value: 2, label: "2 jours par seconde" },
  { value: 8, label: "8 jours par seconde" },
  { value: 20, label: "20 jours par seconde" },
  { value: 50, label: "50 jours par seconde" },
];

export function Live() {
  const { scenarios, policies } = useCatalogues();
  const { open } = useRun();
  const [scenario, setScenario] = useState("s1_nominal");
  const [policy, setPolicy] = useState("P1");
  const [speed, setSpeed] = useState(8);
  const [status, setStatus] = useState<Status>("idle");
  const [start, setStart] = useState<Start>();
  const [frame, setFrame] = useState<Frame>();
  const [feed, setFeed] = useState<(SimEvent & { key: string })[]>([]);
  const [runId, setRunId] = useState<string>();
  const ws = useRef<WebSocket | null>(null);

  // Un lien du type /direct?lancer=1 démarre la simulation à l'ouverture (démonstration).
  const [search] = useSearchParams();
  useEffect(() => {
    if (search.get("lancer") === "1") launch();
    return () => {
      const socket = ws.current;
      ws.current = null;
      socket?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const familyLabels = Object.fromEntries(
    Object.entries(scenarios.find((s) => s.id === scenario)?.families ?? {}).map(([k, v]) => [k, v.label]),
  );

  function launch() {
    ws.current?.close();
    setFrame(undefined);
    setFeed([]);
    setRunId(undefined);
    const socket = new WebSocket(liveSocketUrl());
    ws.current = socket;
    // Une connexion remplacée ou fermée ne doit plus modifier l'affichage.
    const current = () => ws.current === socket;
    socket.onopen = () => {
      if (!current()) return;
      socket.send(JSON.stringify({ scenario, policy, seed: 1, speed }));
      setStatus("running");
    };
    socket.onmessage = (msg) => {
      if (!current()) return;
      const data = JSON.parse(msg.data);
      if (data.type === "start") setStart(data);
      else if (data.type === "day") {
        setFrame(data);
        if (data.events.length) {
          setFeed((prev) =>
            [...data.events.map((e: SimEvent, i: number) => ({ ...e, key: `${data.day}-${i}-${e.t}` })).reverse(), ...prev].slice(0, 60),
          );
        }
      } else if (data.type === "done") {
        setRunId(data.run_id);
        setStatus("done");
      }
    };
    socket.onerror = () => {
      if (current()) setStatus("error");
    };
    socket.onclose = () => {
      if (current()) setStatus((s) => (s === "running" || s === "paused" ? "idle" : s));
    };
  }

  function send(payload: object) {
    ws.current?.send(JSON.stringify(payload));
  }

  const running = status === "running" || status === "paused";
  const parts = start?.parts ?? [];

  return (
    <Page
      title="Flotte en direct"
      description="La simulation avance jour après jour. Chaque case est une machine, colorée selon son état hors service le plus long de la journée ; un clic ouvre sa fiche une fois l'année terminée."
      actions={
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Scénario">
            <Select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={running}>
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </Select>
          </Field>
          <Field label="Politique">
            <Select value={policy} onChange={(e) => setPolicy(e.target.value)} disabled={running}>
              {policies.map((p) => (
                <option key={p.code} value={p.code}>{p.code} · {p.label}</option>
              ))}
            </Select>
          </Field>
          <Field label="Vitesse">
            <Select
              value={speed}
              onChange={(e) => {
                const v = Number(e.target.value);
                setSpeed(v);
                if (running) send({ action: "speed", value: v });
              }}
            >
              {SPEEDS.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </Select>
          </Field>
          {status === "running" && (
            <Button variant="secondary" onClick={() => { send({ action: "pause" }); setStatus("paused"); }}>
              Mettre en pause
            </Button>
          )}
          {status === "paused" && (
            <Button variant="secondary" onClick={() => { send({ action: "resume" }); setStatus("running"); }}>
              Reprendre
            </Button>
          )}
          <Button onClick={launch}>{running ? "Recommencer" : "Lancer la simulation"}</Button>
        </div>
      }
    >
      {status === "error" && (
        <Notice tone="error">Connexion à l'API impossible. Vérifiez que le serveur FastAPI tourne sur le port 8000.</Notice>
      )}
      {!frame && status !== "error" && (
        <div className="rounded-md bg-mist px-6 py-10 text-center text-ink-soft">
          Choisissez un scénario et une politique, puis lancez la simulation pour voir la flotte évoluer.
        </div>
      )}
      {frame && start && (
        <div className="grid gap-12 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="min-w-0">
            <div className="mb-6">
              <div className="flex items-baseline justify-between text-[14px]">
                <span className="font-display text-[18px] font-semibold">
                  Jour <span className="tabular">{frame.day + 1}</span>
                  <span className="text-ink-soft"> sur {start.days}</span>
                </span>
                {status === "done" && runId && (
                  <Link
                    to="/"
                    onClick={() => open(runId)}
                    className="rounded-sm px-2 py-1 font-medium text-plum-700 hover:bg-plum-50"
                  >
                    Ouvrir le bilan de cette année
                  </Link>
                )}
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-plum-100" aria-hidden>
                <div className="h-full bg-plum-600" style={{ width: `${((frame.day + 1) / start.days) * 100}%` }} />
              </div>
            </div>
            <FleetGrid
              states={frame.states}
              families={start.families}
              familyLabels={familyLabels}
              linkable={status === "done"}
            />
            <div className="mt-5">
              <StateLegend />
            </div>
          </div>
          <aside className="space-y-10">
            <Section title="Depuis le premier jour">
              <KpiList
                items={[
                  { label: "TRS", value: fmtPct(frame.trs), emphasis: true },
                  { label: "Disponibilité", value: fmtPct(frame.availability) },
                  { label: "Coût cumulé", value: fmtUR(frame.cost_total) },
                  { label: "Pannes", value: fmtInt(frame.failures) },
                  { label: "Ruptures de stock", value: fmtInt(frame.stockouts) },
                ]}
              />
            </Section>
            <Section title="Magasin">
              <StockBars parts={parts} frame={frame} />
            </Section>
            <Section title="Techniciens">
              <p className="text-[14px]">
                <span className="tabular">{frame.techs_busy}</span> sur {start.technicians} en intervention
                {frame.techs_queue > 0 && (
                  <span className="text-sun-800">
                    , <span className="tabular">{frame.techs_queue}</span> machine{frame.techs_queue > 1 ? "s" : ""} en attente
                  </span>
                )}
              </p>
            </Section>
          </aside>
          <Section title="Journal" description="Derniers événements, du plus récent au plus ancien." className="xl:col-span-2">
            <ol className="max-h-[320px] divide-y divide-line overflow-y-auto">
              {feed.map((e) => (
                <li key={e.key} className="flex gap-4 py-2 text-[14px]">
                  <span className="w-20 shrink-0 font-mono text-[13px] text-ink-soft tabular">
                    J{e.day + 1} {String(Math.floor(e.t % 24)).padStart(2, "0")} h
                  </span>
                  <span className={e.type === "panne" ? "text-danger" : e.type === "rupture" ? "text-sun-800" : undefined}>
                    {eventSentence(e, COMPONENTS, parts)}
                  </span>
                </li>
              ))}
            </ol>
          </Section>
        </div>
      )}
    </Page>
  );
}

function StockBars({ parts, frame }: { parts: Part[]; frame: Frame }) {
  return (
    <ul className="space-y-4">
      {parts.map((p, i) => {
        const max = Math.max(p.S, frame.stock[i] + frame.on_order[i], 1);
        const waiting = frame.backorders[i];
        return (
          <li key={p.code}>
            <div className="flex items-baseline justify-between text-[14px]">
              <span>
                <span className="font-medium">{p.code}</span> <span className="text-ink-soft">{p.name}</span>
              </span>
              <span className="tabular">
                {frame.stock[i]}
                <span className="text-ink-soft"> en stock</span>
              </span>
            </div>
            <div className="relative mt-1.5 h-2 rounded-full bg-plum-50" aria-hidden>
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-plum-600"
                style={{ width: `${(frame.stock[i] / max) * 100}%` }}
              />
              <div
                className="absolute inset-y-0 rounded-full bg-plum-200"
                style={{ left: `${(frame.stock[i] / max) * 100}%`, width: `${(frame.on_order[i] / max) * 100}%` }}
              />
              <div className="absolute -top-1 h-4 w-px bg-sun-600" style={{ left: `${(p.s / max) * 100}%` }} title={`Seuil s = ${p.s}`} />
            </div>
            <p className="mt-1 text-[12px] text-ink-soft">
              {frame.on_order[i] > 0 ? `${frame.on_order[i]} en commande` : "aucune commande en cours"}
              {waiting > 0 && <span className="font-medium text-sun-800">, {waiting} machine{waiting > 1 ? "s" : ""} en attente</span>}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
