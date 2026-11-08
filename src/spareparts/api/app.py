"""API de l'application web (cahier des charges, § 6.4).

Lancement : uv run uvicorn spareparts.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from spareparts import __version__
from spareparts.api.serialize import jsonable, live_frame, machine_payload, run_payload
from spareparts.config import list_scenarios, load_scenario
from spareparts.data import azure
from spareparts.evaluation import summarize
from spareparts.policies import POLICY_LABELS, available_policies, make_policy
from spareparts.sim import FleetSimulation, SimulationResult
from spareparts.sim.results import build_result

app = FastAPI(title="Spare Parts DL", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_RUNS = 20
WS_INIT_TIMEOUT_SECONDS = 30.0
# Chaque processus enfant réimporte numpy, scipy et torch : en ouvrir un par cœur épuise la
# mémoire virtuelle de la machine, les enfants meurent au chargement des DLL natives et le pool
# devient inutilisable. Le nombre de travailleurs est donc borné.
MAX_WORKERS = min(4, os.cpu_count() or 1)
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
        _pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    return _pool


def _abandon_pool() -> None:
    """Écarte un pool dont un enfant est mort.

    Un pool cassé ne redevient jamais utilisable : le conserver condamne toutes les requêtes
    ultérieures à un 500. On le remplace donc par un pool neuf.
    """
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


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


class LiveRequest(BaseModel):
    scenario: str = "s1_nominal"
    policy: str = "P0"
    seed: int = 1
    days: int | None = Field(default=None, ge=1, le=730)
    speed: float = Field(default=8, ge=0.5, le=60)
    overrides: Overrides = Field(default_factory=Overrides)


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
def policies(scenario: str = "s1_nominal") -> list[dict]:
    return [{"code": code, "label": POLICY_LABELS[code]} for code in available_policies(scenario)]


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


async def _dispatch_replications(scenario, policy, seeds, days, overrides) -> list[dict]:
    loop = asyncio.get_running_loop()
    jobs = [
        loop.run_in_executor(_executor(), _simulate_kpis, scenario, policy, s, days, overrides)
        for s in seeds
    ]
    return await asyncio.gather(*jobs)


async def _replications(scenario, policy, seeds, days, overrides) -> list[dict]:
    try:
        return await _dispatch_replications(scenario, policy, seeds, days, overrides)
    except BrokenProcessPool:
        # Un enfant est mort : le pool est définitivement cassé. On en ouvre un neuf et on
        # rejoue une fois ; si la panne se reproduit, elle est signalée sans masquer la cause.
        _abandon_pool()
        try:
            return await _dispatch_replications(scenario, policy, seeds, days, overrides)
        except BrokenProcessPool as exc:
            _abandon_pool()
            raise HTTPException(
                503, "mémoire insuffisante pour exécuter les processus de simulation"
            ) from exc


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
    return jsonable({**azure.summary(), "calibration": azure.calibration()})


@app.get("/api/models/prediction")
def prediction_metrics() -> dict:
    """Métriques des modèles de prédiction, par source de données (§ 5.2)."""
    import json

    from spareparts.config import ROOT

    out = {}
    for source, name in (("simulation", "s1_nominal"), ("azure", "azure")):
        path = ROOT / "reports" / "metrics" / f"prediction_{name}.json"
        if path.exists():
            out[source] = json.loads(path.read_text(encoding="utf-8"))
    return jsonable(out)


@app.get("/api/evaluation")
def evaluation_report() -> dict:
    """Évaluation comparative enregistrée (§ 7.3) : coût, TRS, TOST et robustesse.

    Le fichier est produit par scripts/evaluate_policies.py --robustness. Les tirages bruts
    (clé « runs ») sont écartés : l'interface n'affiche que les synthèses par politique.
    """
    import json

    from spareparts.config import ROOT

    path = ROOT / "reports" / "evaluation_s1_nominal.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("runs", None)
    return jsonable({"available": True, **data})


@app.get("/api/generalisation")
def generalisation_report() -> dict:
    """Stabilité du classement selon la composition de la flotte (item 12, § 7.4).

    Le fichier est produit par scripts/generalize_ranking.py. Les tirages bruts
    (clé « runs » de chaque composition) sont écartés : l'interface n'affiche que les
    synthèses par composition.
    """
    import json

    from spareparts.config import ROOT

    path = ROOT / "reports" / "metrics" / "generalisation_flotte.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        case.pop("runs", None)
    return jsonable({"available": True, **data})


@app.get("/api/plancher")
def floor_report() -> dict:
    """Plancher de représentation des espaces d'action (décomposition de l'objectif O4).

    Le fichier est produit par scripts/action_space_floor.py. Le classement exhaustif des
    36 actions de l'espace réduit (clé « classement ») est écarté : l'interface n'affiche que
    les planchers, les bornes publiées et la décomposition de l'écart à P1.
    """
    import json

    from spareparts.config import ROOT

    path = ROOT / "reports" / "metrics" / "plancher_representation.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    for space in data.get("espaces", {}).values():
        space.pop("classement", None)
    return jsonable({"available": True, **data})


_surrogate: dict = {}


def _load_surrogate():
    """Charge le modèle de substitution une seule fois ; None s'il n'est pas entraîné."""
    from spareparts.surrogate import MODEL_PATH

    if "model" not in _surrogate:
        if not MODEL_PATH.exists():
            return None
        from spareparts.surrogate.model import DeepEnsemble

        _surrogate["model"], _surrogate["metrics"] = DeepEnsemble.load(MODEL_PATH)
    return _surrogate["model"]


class SurrogateRequest(BaseModel):
    technicians: float = Field(ge=1, le=4)
    lead_time_factor: float = Field(ge=0.5, le=2.5)
    failure_rate_factor: float = Field(ge=0.6, le=1.8)
    pm_factor: float = Field(ge=0.5, le=1.6)
    stock_factor: float = Field(ge=0.5, le=3.0)


@app.get("/api/surrogate")
def surrogate_status() -> dict:
    try:
        model = _load_surrogate()
    except ImportError:
        model = None
    if model is None:
        return {"available": False}
    from spareparts.surrogate.doe import OUTPUTS, PARAMS

    return jsonable(
        {
            "available": True,
            "params": {
                k: {"low": lo, "high": hi, "integer": i} for k, (lo, hi, i) in PARAMS.items()
            },
            "outputs": OUTPUTS,
            "metrics": _surrogate.get("metrics", {}),
        }
    )


@app.post("/api/surrogate/predict")
def surrogate_predict(req: SurrogateRequest) -> dict:
    import pandas as pd

    model = _load_surrogate()
    if model is None:
        raise HTTPException(404, "modèle de substitution non entraîné : scripts/train_surrogate.py")
    mean, std = model.predict(pd.DataFrame([req.model_dump()]))
    return jsonable({k: {"mean": mean[k].iloc[0], "std": std[k].iloc[0]} for k in mean.columns})


@app.websocket("/api/ws/live")
async def live(ws: WebSocket) -> None:
    """Diffuse une simulation jour après jour ; le client peut la mettre en pause ou l'accélérer.

    Premier message attendu : {"scenario", "policy", "seed", "days", "speed"}. Messages
    suivants : {"action": "pause" | "resume" | "stop"} ou {"action": "speed", "value": n}.
    """
    await ws.accept()
    try:
        payload = await asyncio.wait_for(
            ws.receive_json(), timeout=WS_INIT_TIMEOUT_SECONDS
        )
    except TimeoutError:
        await ws.close(code=1008, reason="délai d'attente des paramètres dépassé")
        return
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await ws.close(code=1008, reason=str(exc)[:120])
        return

    try:
        params = LiveRequest.model_validate(payload)
        scenario = params.scenario
        cfg = load_scenario(
            scenario, **params.overrides.model_dump(exclude_none=True)
        )
        policy = make_policy(params.policy, scenario)
    except (ValidationError, FileNotFoundError, ValueError, HTTPException) as exc:
        await ws.close(code=1008, reason=str(exc)[:120])
        return
    days = params.days or cfg.horizon_days
    control = {"paused": False, "speed": params.speed, "stop": False}

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    control["stop"] = True
                    return
                action = msg.get("action")
                if action == "pause":
                    control["paused"] = True
                elif action == "resume":
                    control["paused"] = False
                elif action == "speed":
                    control["speed"] = max(0.5, min(float(msg.get("value", 8)), 60.0))
                elif action == "stop":
                    control["stop"] = True
        except (WebSocketDisconnect, RuntimeError, TypeError, ValueError):
            control["stop"] = True

    sim = FleetSimulation(cfg, seed=params.seed)
    policy.reset(sim)
    reader_task = asyncio.create_task(reader())
    try:
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
        with contextlib.suppress(asyncio.CancelledError):
            await reader_task
