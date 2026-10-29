export type Part = {
  code: string;
  name: string;
  nature: string;
  unit_cost: number;
  lead_days: number;
  urgent_lead_days: number;
  s: number;
  S: number;
  initial_stock: number;
};

export type Scenario = {
  id: string;
  name: string;
  description: string;
  machines: number;
  technicians: number;
  horizon_days: number;
  families: Record<string, { label: string; share: number }>;
  parts: Part[];
};

export type PolicyInfo = { code: string; label: string };

export type Kpis = Record<string, number | null>;

export type RunMeta = {
  scenario_id: string;
  families: string[];
  family_labels: Record<string, string>;
  shifts: number[];
  age_years: number[];
  parts: Part[];
  components: string[];
  signals: string[];
  states: Record<string, string>;
  technicians: number;
  failures_by_component: number[];
  stockouts_by_part: number[];
  demands_by_part: number[];
  state_hours: number[];
  down_hours: number[];
  policy_params?: { s: number[]; S: number[] };
};

export type RunPayload = {
  run_id: string;
  scenario: string;
  policy: string;
  seed: number;
  days: number;
  kpis: Kpis;
  costs: Record<string, number>;
  meta: RunMeta;
  machines: MachineRow[];
  daily: {
    trs: (number | null)[];
    availability: (number | null)[];
    quality: (number | null)[];
    stock: number[][];
    on_order: number[][];
    backorders: number[][];
    state_counts: number[][];
    cost_total: number[];
    cost_cumulative: Record<string, number[]>;
    produced: number[];
    defects: number[];
  };
};

export type MachineRow = {
  id: number;
  family: string;
  shifts: number;
  age_years: number;
  final_state: number;
  failures: number;
  preventive: number;
  down_days: number;
  down_hours: number;
};

export type SimEvent = {
  t: number;
  day: number;
  type: string;
  machine: number | null;
  component: number | null;
  part?: number;
  qty?: number;
  urgent?: boolean;
};

export type MachineDetail = {
  machine: number;
  family: string;
  shifts: number;
  age_years: number;
  states: number[];
  down_hours: number;
  signal_mean: (number | null)[][];
  defect_rate: (number | null)[];
  alarms: number[];
  events: SimEvent[];
  signals: string[];
  components: string[];
};

export type Interval = { mean: number | null; low: number | null; high: number | null };

export type CompareResponse = {
  scenario: string;
  replications: number;
  results: Record<
    string,
    {
      label: string;
      summary: Record<string, Interval>;
      runs: { seed: number; cout_total: number; trs: number }[];
    }
  >;
};

export type WhatIfResponse = { source: string; summary: Record<string, Interval> };

export type SurrogateParams = {
  technicians: number;
  lead_time_factor: number;
  failure_rate_factor: number;
  pm_factor: number;
  stock_factor: number;
};

export type SurrogateStatus =
  | { available: false }
  | {
      available: true;
      params: Record<keyof SurrogateParams, { low: number; high: number; integer: boolean }>;
      outputs: string[];
      metrics: { test?: Record<string, { r2: number; mape: number }>; scenarios?: number; days?: number };
    };

export type SurrogatePrediction = Record<string, { mean: number; std: number }>;

export type Overrides = {
  technicians?: number;
  lead_time_factor?: number;
  failure_rate_factor?: number;
};

export type RunParams = { scenario: string; policy: string; seed: number; days?: number };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* corps non JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  scenarios: () => request<Scenario[]>("/api/scenarios"),
  policies: () => request<PolicyInfo[]>("/api/policies"),
  createRun: (params: RunParams) =>
    request<RunPayload>("/api/runs", { method: "POST", body: JSON.stringify(params) }),
  getRun: (runId: string) => request<RunPayload>(`/api/runs/${runId}`),
  machine: (runId: string, machine: number) =>
    request<MachineDetail>(`/api/runs/${runId}/machines/${machine}`),
  events: (runId: string, kind?: string) =>
    request<SimEvent[]>(`/api/runs/${runId}/events${kind ? `?kind=${kind}` : ""}`),
  compare: (body: {
    scenario: string;
    policies: string[];
    replications: number;
    overrides?: Overrides;
  }) => request<CompareResponse>("/api/compare", { method: "POST", body: JSON.stringify(body) }),
  whatif: (body: { scenario: string; policy: string; overrides: Overrides }) =>
    request<WhatIfResponse>("/api/whatif", { method: "POST", body: JSON.stringify(body) }),
  dataSummary: () => request<DataSummary>("/api/data/summary"),
  surrogate: () => request<SurrogateStatus>("/api/surrogate"),
  predictionMetrics: () => request<Record<string, Record<string, PredictionMetrics>>>("/api/models/prediction"),
  surrogatePredict: (params: SurrogateParams) =>
    request<SurrogatePrediction>("/api/surrogate/predict", { method: "POST", body: JSON.stringify(params) }),
};

export type DataSummary =
  | { complete: false; available: Record<string, boolean> }
  | {
      complete: true;
      available: Record<string, boolean>;
      rows: Record<string, number>;
      period: Record<string, [string, string]>;
      signals: Record<string, { mean: number; std: number; p05: number; p95: number }>;
      failures_by_component: Record<string, number>;
      failures_by_model: Record<string, number>;
      failures_by_month: { months: string[]; series: Record<string, number[]> };
      machines_by_model: Record<string, number>;
      age_histogram: { edges: number[]; counts: number[] };
      errors_by_type: Record<string, number>;
      maint_by_component: Record<string, number>;
      calibration: Calibration | null;
    };

export type ClassMetrics = {
  rappel: number;
  precision: number;
  f1: number;
  pr_auc: number;
  positifs: number;
  exemples: number;
};

export type PredictionMetrics = ClassMetrics & { a_7_jours?: ClassMetrics; c_index_7j?: number };

export type ComparedCount = { azure: number; simulees: number; ecart_type: number; ecart_relatif: number };

export type Calibration = {
  lois: Record<string, { beta: number; eta: number; n: number; pannes: number }>;
  validation?: {
    lois_corrigees: Record<string, { beta: number; eta: number }>;
    validation_s2: Record<string, ComparedCount>;
    annee_2015?: Record<string, ComparedCount>;
    critere_atteint: boolean;
    critere_atteint_s2?: boolean;
  };
};

export function liveSocketUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/api/ws/live`;
}
