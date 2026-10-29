import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api, type RunParams, type RunPayload } from "./api";

type RunContextValue = {
  params: RunParams;
  run?: RunPayload;
  loading: boolean;
  error?: string;
  simulate: (params: RunParams) => void;
  open: (runId: string) => void;
};

const RunContext = createContext<RunContextValue | null>(null);

export const DEFAULT_PARAMS: RunParams = { scenario: "s1_nominal", policy: "P1", seed: 1 };

/** Simulation de référence partagée par le tableau de bord, les machines et le stock. */
export function RunProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useState<RunParams>(DEFAULT_PARAMS);
  const [run, setRun] = useState<RunPayload>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const latest = useRef(0);

  const load = useCallback((promise: Promise<RunPayload>) => {
    const id = ++latest.current;
    setLoading(true);
    setError(undefined);
    promise
      .then((payload) => {
        if (id !== latest.current) return;
        setRun(payload);
        setParams({ scenario: payload.meta.scenario_id, policy: payload.policy, seed: payload.seed });
      })
      .catch((err: Error) => {
        if (id === latest.current) setError(err.message);
      })
      .finally(() => {
        if (id === latest.current) setLoading(false);
      });
  }, []);

  const simulate = useCallback(
    (p: RunParams) => {
      setParams(p);
      load(api.createRun(p));
    },
    [load],
  );

  const open = useCallback((runId: string) => load(api.getRun(runId)), [load]);

  useEffect(() => {
    simulate(DEFAULT_PARAMS);
  }, [simulate]);

  return (
    <RunContext.Provider value={{ params, run, loading, error, simulate, open }}>
      {children}
    </RunContext.Provider>
  );
}

export function useRun(): RunContextValue {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun doit être utilisé dans RunProvider");
  return ctx;
}
