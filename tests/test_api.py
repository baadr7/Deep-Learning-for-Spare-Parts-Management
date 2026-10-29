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
