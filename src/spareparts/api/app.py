"""API de l'application web (cahier des charges, § 6.4).

Lancement : uv run uvicorn spareparts.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from spareparts import __version__
from spareparts.api.serialize import jsonable, live_frame, machine_payload, run_payload
from spareparts.config import list_scenarios, load_scenario
from spareparts.data import azure
from spareparts.evaluation import summarize
from spareparts.policies import POLICY_LABELS, make_policy
from spareparts.sim import FleetSimulation, SimulationResult
from spareparts.sim.results import build_result

app = FastAPI(title="Spare Parts DL", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_RUNS = 20
_runs: OrderedDict[str, SimulationResult] = OrderedDict()
_pool: ProcessPoolExecutor | None = None


def _store(result: SimulationResult) -> str:
    run_id = uuid.uuid4().hex[:10]
    _runs[run_id] = result
    while len(_runs) > MAX_RUNS:
        _runs.popitem(last=False)
    return run_id


def _get_run(run_id: str) -> SimulationResult:
    if run_id not in _runs:
        raise HTTPException(404, "simulation inconnue ou expirée")
    return _runs[run_id]


def _executor() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor()
    return _pool


class Overrides(BaseModel):
    technicians: int | None = Field(default=None, ge=1, le=10)
    lead_time_factor: float | None = Field(default=None, gt=0, le=5)
    failure_rate_factor: float | None = Field(default=None, gt=0, le=5)


class RunRequest(BaseModel):
    scenario: str = "s1_nominal"
    policy: str = "P0"
    seed: int = 1
    days: int | None = Field(default=None, ge=1, le=730)
    overrides: Overrides = Overrides()


class CompareRequest(BaseModel):
    scenario: str = "s1_nominal"
    policies: list[str] = ["P0", "P1", "P2", "Oracle"]
    replications: int = Field(default=5, ge=1, le=30)
    days: int | None = Field(default=None, ge=30, le=730)
    overrides: Overrides = Overrides()


class WhatIfRequest(BaseModel):
    scenario: str = "s1_nominal"
    policy: str = "P1"
    replications: int = Field(default=3, ge=1, le=10)
    days: int = Field(default=180, ge=30, le=365)
    overrides: Overrides = Overrides()


def _simulate_kpis(scenario: str, policy: str, seed: int, days, overrides: dict) -> dict:
    cfg = load_scenario(scenario, **overrides)
    sim = FleetSimulation(cfg, seed=seed, log_events=False)
    return sim.run(make_policy(policy, scenario), days=days).kpis


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    out = []
    for key in list_scenarios():
        cfg = load_scenario(key)
        out.append(
            {
                "id": key,
                "name": cfg.name,
                "description": cfg.description,
                "machines": cfg.fleet.n_machines,
                "technicians": cfg.technicians,
                "horizon_days": cfg.horizon_days,
                "families": {
                    code: {"label": fam.label, "share": cfg.fleet.family_share.get(code, 0)}
                    for code, fam in cfg.families.items()
                },
                "parts": [p.model_dump() for p in cfg.parts],
            }
        )
    return out


@app.get("/api/policies")
def policies() -> list[dict]:
    return [{"code": code, "label": label} for code, label in POLICY_LABELS.items()]


@app.post("/api/runs")
async def create_run(req: RunRequest) -> dict:
    overrides = req.overrides.model_dump(exclude_none=True)
    try:
        cfg = load_scenario(req.scenario, **overrides)
        policy = make_policy(req.policy, req.scenario)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    sim = FleetSimulation(cfg, seed=req.seed)
    result = await asyncio.to_thread(sim.run, policy, req.days)
    result.meta["scenario_id"] = req.scenario
    return run_payload(_store(result), result)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return run_payload(run_id, _get_run(run_id))


@app.get("/api/runs/{run_id}/machines/{machine}")
def get_machine(run_id: str, machine: int) -> dict:
    result = _get_run(run_id)
    if not 0 <= machine < len(result.meta["families"]):
        raise HTTPException(404, "machine inconnue")
    return machine_payload(result, machine)


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, kind: str | None = None, limit: int = 500) -> list[dict]:
    events = _get_run(run_id).events
    if kind:
        events = [e for e in events if e["type"] == kind]
    return jsonable(events[-limit:])


async def _replications(scenario, policy, seeds, days, overrides) -> list[dict]:
    loop = asyncio.get_running_loop()
    pool = _executor()
    jobs = [
        loop.run_in_executor(pool, _simulate_kpis, scenario, policy, s, days, overrides)
        for s in seeds
    ]
    return await asyncio.gather(*jobs)


@app.post("/api/compare")
async def compare(req: CompareRequest) -> dict:
    overrides = req.overrides.model_dump(exclude_none=True)
    seeds = list(range(1000, 1000 + req.replications))
    out = {}
    for code in req.policies:
        runs = await _replications(req.scenario, code, seeds, req.days, overrides)
        out[code] = {
            "label": POLICY_LABELS.get(code, code),
            "summary": summarize(runs),
            "runs": [
                {"seed": s, "cout_total": r["cout_total"], "trs": r["trs"]}
                for s, r in zip(seeds, runs, strict=True)
            ],
        }
    return jsonable({"scenario": req.scenario, "replications": req.replications, "results": out})


@app.post("/api/whatif")
async def whatif(req: WhatIfRequest) -> dict:
    overrides = req.overrides.model_dump(exclude_none=True)
    seeds = list(range(2000, 2000 + req.replications))
    runs = await _replications(req.scenario, req.policy, seeds, req.days, overrides)
    return jsonable({"source": "simulation", "summary": summarize(runs)})


@app.get("/api/data/summary")
def data_summary() -> dict:
    return jsonable(azure.summary())


@app.websocket("/api/ws/live")
async def live(ws: WebSocket) -> None:
    """Diffuse une simulation jour après jour ; le client peut la mettre en pause ou l'accélérer.

    Premier message attendu : {"scenario", "policy", "seed", "days", "speed"}. Messages
    suivants : {"action": "pause" | "resume" | "stop"} ou {"action": "speed", "value": n}.
    """
    await ws.accept()
    try:
        params = await ws.receive_json()
        scenario = params.get("scenario", "s1_nominal")
        cfg = load_scenario(scenario, **(params.get("overrides") or {}))
        policy = make_policy(params.get("policy", "P0"), scenario)
    except (WebSocketDisconnect, FileNotFoundError, ValueError) as exc:
        await ws.close(code=1003, reason=str(exc)[:120])
        return
    days = int(params.get("days") or cfg.horizon_days)
    control = {"paused": False, "speed": float(params.get("speed", 8)), "stop": False}

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "pause":
                    control["paused"] = True
                elif action == "resume":
                    control["paused"] = False
                elif action == "speed":
                    control["speed"] = max(0.5, min(float(msg.get("value", 8)), 60.0))
                elif action == "stop":
                    control["stop"] = True
        except (WebSocketDisconnect, RuntimeError):
            control["stop"] = True

    reader_task = asyncio.create_task(reader())
    sim = FleetSimulation(cfg, seed=int(params.get("seed", 1)))
    policy.reset(sim)
    await ws.send_json(
        jsonable(
            {
                "type": "start",
                "days": days,
                "machines": sim.M,
                "families": [sim.family_codes[f] for f in sim.family],
                "parts": [p.model_dump() for p in cfg.parts],
                "technicians": cfg.technicians,
            }
        )
    )
    try:
        while sim.day < days and not control["stop"]:
            if control["paused"]:
                await asyncio.sleep(0.1)
                continue
            sim.apply(policy.decide(sim.observe()))
            record = sim.advance_day()
            await ws.send_json(live_frame(sim, record))
            await asyncio.sleep(1.0 / control["speed"])
        if not control["stop"]:
            result = build_result(sim, policy)
            result.meta["scenario_id"] = scenario
            run_id = _store(result)
            await ws.send_json(jsonable({"type": "done", "run_id": run_id, "kpis": result.kpis}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()
