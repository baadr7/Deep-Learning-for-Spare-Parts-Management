import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, type RunParams } from "../lib/api";
import { useRun } from "../lib/run";
import { Button, Field, Input, Select } from "./ui";

export function useCatalogues() {
  const scenarios = useQuery({ queryKey: ["scenarios"], queryFn: api.scenarios, staleTime: Infinity });
  const policies = useQuery({ queryKey: ["policies"], queryFn: api.policies, staleTime: Infinity });
  return { scenarios: scenarios.data ?? [], policies: policies.data ?? [] };
}

/** Choix du scénario, de la politique et de la graine de la simulation de référence. */
export function RunControls() {
  const { params, simulate, loading } = useRun();
  const { scenarios, policies } = useCatalogues();
  const [draft, setDraft] = useState<RunParams>(params);
  useEffect(() => setDraft(params), [params]);

  return (
    <form
      className="flex flex-wrap items-end gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        simulate(draft);
      }}
    >
      <Field label="Scénario">
        <Select
          value={draft.scenario}
          onChange={(e) => setDraft({ ...draft, scenario: e.target.value })}
          className="min-w-[200px]"
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Politique">
        <Select
          value={draft.policy}
          onChange={(e) => setDraft({ ...draft, policy: e.target.value })}
          className="min-w-[240px]"
        >
          {policies.map((p) => (
            <option key={p.code} value={p.code}>
              {p.code} · {p.label}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Graine">
        <Input
          type="number"
          min={0}
          value={draft.seed}
          onChange={(e) => setDraft({ ...draft, seed: Number(e.target.value) })}
          className="w-24"
        />
      </Field>
      <Button type="submit" disabled={loading}>
        {loading ? "Simulation en cours" : "Simuler 365 jours"}
      </Button>
    </form>
  );
}
