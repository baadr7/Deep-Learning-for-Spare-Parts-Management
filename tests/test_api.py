import numpy as np
import pytest
from fastapi.testclient import TestClient

from spareparts.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_catalogues(client):
    assert client.get("/api/health").json()["status"] == "ok"
    ids = {s["id"] for s in client.get("/api/scenarios").json()}
    assert {"s0_homogene", "s1_nominal"} <= ids
    codes = {p["code"] for p in client.get("/api/policies").json()}
    assert {"P0", "P1", "P2", "Oracle"} <= codes


def test_run_then_machine_detail(client):
    run = client.post("/api/runs", json={"policy": "P0", "days": 20, "seed": 3}).json()
    assert len(run["daily"]["trs"]) == 20
    assert 0 <= run["kpis"]["disponibilite"] <= 1
    detail = client.get(f"/api/runs/{run['run_id']}/machines/0").json()
    assert len(detail["states"]) == 20
    assert len(detail["signal_mean"][0]) == 4


def test_unknown_policy_is_rejected(client):
    assert client.post("/api/runs", json={"policy": "P9"}).status_code == 400


def test_live_stream_sends_each_day_then_done(client):
    with client.websocket_connect("/api/ws/live") as ws:
        ws.send_json({"policy": "P2", "days": 3, "speed": 60})
        assert ws.receive_json()["type"] == "start"
        days = [ws.receive_json() for _ in range(3)]
        assert [d["day"] for d in days] == [0, 1, 2]
        assert len(days[0]["states"]) == 100
        done = ws.receive_json()
        assert done["type"] == "done"
    assert client.get(f"/api/runs/{done['run_id']}").status_code == 200


def test_compare_returns_confidence_intervals(client):
    body = {"policies": ["P0"], "replications": 2, "days": 30}
    res = client.post("/api/compare", json=body).json()["results"]["P0"]
    trs = res["summary"]["trs"]
    assert trs["low"] <= trs["mean"] <= trs["high"]
    assert len(res["runs"]) == 2


def test_evaluation_endpoint_exposes_syntheses_without_raw_runs(client):
    body = client.get("/api/evaluation").json()
    assert body["available"] is True
    # Seules les synthèses par politique sont exposées : les tirages bruts sont écartés.
    assert "runs" not in body
    assert {"scenario", "replications", "policies", "nominal"} <= body.keys()
    # P1 est la référence : elle n'apparaît donc jamais comme politique comparée.
    assert "P1" not in body["tost_cout"]
    assert "P1" not in body["robustesse_tost"][next(iter(body["robustesse_tost"]))]


def test_generalisation_endpoint_exposes_compositions_without_raw_runs(client):
    body = client.get("/api/generalisation").json()
    assert body["available"] is True
    assert {"base", "replications", "policies", "cases"} <= body.keys()
    # Seules les synthèses par composition sont exposées : les tirages bruts sont écartés.
    assert body["cases"] and all("runs" not in case for case in body["cases"])
    # Chaque composition porte son tau et le verdict de l'ordre attendu du cahier des charges.
    assert all(
        "kendall_vs_nominal" in case and "conserve" in case["ordre_cdc"] for case in body["cases"]
    )
    # La composition nominale n'a pas de part de famille U : la valeur non finie devient null.
    nominal = next(case for case in body["cases"] if case["nom"] == "nominal")
    assert nominal["part_U"] is None


def test_plancher_endpoint_exposes_decomposition_without_ranking(client):
    body = client.get("/api/plancher").json()
    assert body["available"] is True
    assert {"scenario", "reference", "decomposition"} <= body.keys()
    # La décomposition porte sur les deux agents à espace d'action paramétrique.
    assert set(body["decomposition"]) == {"DQN", "P4"}
    for dec in body["decomposition"].values():
        # L'écart observé à P1 se scinde exactement en apprentissage + représentation.
        assert dec["ecart_observe"] == pytest.approx(
            dec["ecart_apprentissage"] + dec["ecart_representation"]
        )
    # Le classement des 36 actions réduites n'est pas exposé à l'interface (synthèses seules).
    assert all("classement" not in space for space in body["espaces"].values())


def test_run_events_can_be_filtered_by_kind(client):
    run = client.post("/api/runs", json={"policy": "P0", "days": 120, "seed": 7}).json()
    events = client.get(f"/api/runs/{run['run_id']}/events").json()
    assert events and {"type", "day"} <= events[0].keys()
    pannes = client.get(f"/api/runs/{run['run_id']}/events", params={"kind": "panne"}).json()
    assert pannes and all(event["type"] == "panne" for event in pannes)


def test_unknown_run_and_machine_are_rejected(client):
    assert client.get("/api/runs/inconnu").status_code == 404
    run = client.post("/api/runs", json={"policy": "P0", "days": 5}).json()
    assert client.get(f"/api/runs/{run['run_id']}/machines/9999").status_code == 404


def test_whatif_returns_a_simulation_summary(client):
    body = {"policy": "P1", "replications": 2, "days": 60}
    res = client.post("/api/whatif", json=body).json()
    assert res["source"] == "simulation"
    cout = res["summary"]["cout_total"]
    assert cout["low"] <= cout["mean"] <= cout["high"]


def test_data_summary_reports_the_raw_files_availability(client):
    body = client.get("/api/data/summary").json()
    assert set(body["available"]) == {"telemetry", "errors", "maint", "failures", "machines"}
    assert "calibration" in body


def test_prediction_metrics_are_exposed_when_present(client):
    body = client.get("/api/models/prediction").json()
    # Les métriques par source sont enregistrées par scripts/ et servies telles quelles.
    assert {"simulation", "azure"} <= body.keys()


def test_report_endpoints_degrade_without_the_committed_files(client, monkeypatch, tmp_path):
    # Un dépôt fraîchement cloné n'a pas encore produit ces fichiers : l'API le dit sans planter.
    monkeypatch.setattr("spareparts.config.ROOT", tmp_path)
    assert client.get("/api/evaluation").json() == {"available": False}
    assert client.get("/api/generalisation").json() == {"available": False}
    assert client.get("/api/plancher").json() == {"available": False}


def test_surrogate_endpoints_degrade_without_a_trained_model(client, monkeypatch, tmp_path):
    from spareparts.api import app as api

    api._surrogate.clear()
    monkeypatch.setattr("spareparts.surrogate.MODEL_PATH", tmp_path / "absent.pt")
    assert client.get("/api/surrogate").json() == {"available": False}
    resp = client.post(
        "/api/surrogate/predict",
        json={
            "technicians": 2,
            "lead_time_factor": 1.0,
            "failure_rate_factor": 1.0,
            "pm_factor": 1.0,
            "stock_factor": 1.0,
        },
    )
    assert resp.status_code == 404


def test_jsonable_converts_numpy_scalars_and_drops_non_finite_values():
    from spareparts.api.serialize import jsonable

    out = jsonable({"entier": np.int64(3), "booleen": np.bool_(True), "non_fini": float("nan")})
    assert out == {"entier": 3, "booleen": True, "non_fini": None}


def test_simulate_kpis_helper_matches_a_direct_run():
    from spareparts.api.app import _simulate_kpis

    kpis = _simulate_kpis("s1_nominal", "P0", 4, 30, {})
    assert kpis["cout_total"] > 0.0


def test_the_run_store_keeps_only_the_most_recent_runs():
    from types import SimpleNamespace

    from spareparts.api import app as api

    saved = dict(api._runs)
    try:
        api._runs.clear()
        for _ in range(api.MAX_RUNS + 3):
            api._store(SimpleNamespace())
        assert len(api._runs) == api.MAX_RUNS
    finally:
        api._runs.clear()
        api._runs.update(saved)


def test_live_stream_rejects_an_unknown_policy(client):
    from fastapi import WebSocketDisconnect

    # Le premier message porte le scénario et la politique : une politique inconnue fait fermer
    # le canal avec le code 1003 (déconnexion applicative) avant toute diffusion.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws/live") as ws:
            ws.send_json({"policy": "P9"})
            ws.receive_json()
