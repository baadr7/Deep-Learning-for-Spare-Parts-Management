"""Tests des modules d'apprentissage ; ignorés si l'extra ``ml`` n'est pas installé."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from spareparts.config import load_scenario  # noqa: E402
from spareparts.envs.fleet_env import FleetEnv  # noqa: E402
from spareparts.predict.dataset import HORIZON, simulate_dataset, tabular_features  # noqa: E402
from spareparts.predict.models import ARCHITECTURES, failure_probability, survival_loss  # noqa: E402
from spareparts.sim import FleetSimulation  # noqa: E402


@pytest.mark.parametrize("level", ["reduced", "n1", "n2"])
def test_env_follows_gymnasium_api(level):
    from gymnasium.utils.env_checker import check_env

    env = FleetEnv(level=level, horizon_days=5)
    check_env(env, skip_render_check=True)


def test_n2_masks_never_target_a_stopped_machine():
    # Les masques portent sur chaque machine séparément : ils ne peuvent pas limiter le nombre
    # total d'interventions au nombre de techniciens libres, le simulateur refuse l'excédent.
    env = FleetEnv(level="n2", horizon_days=10)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        masks = env.action_masks()
        nvec = env.action_space.nvec
        action, start = [], 0
        for n in nvec:
            allowed = np.flatnonzero(masks[start : start + n])
            action.append(int(rng.choice(allowed)))
            start += n
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        assert np.isfinite(reward) and obs.shape == env.observation_space.shape
        assert "machine indisponible" not in info["rejected"]
    assert truncated


def test_survival_loss_and_probabilities():
    logits = torch.zeros(3, 2, HORIZON, requires_grad=True)
    event = torch.tensor([[0, -1], [3, -1], [-1, 2]])
    observed = torch.tensor([[0, HORIZON], [3, 5], [HORIZON, 2]])
    loss = survival_loss(logits, event, observed)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    p = failure_probability(logits.detach(), 7)
    assert torch.all((p >= 0) & (p <= 1))


def test_dataset_and_networks_have_consistent_shapes():
    data = simulate_dataset(seeds=(1,), days=8)
    assert data.seq.shape[1:] == (48, 5)
    assert data.event_bin.shape == data.observed.shape
    assert (data.observed <= HORIZON).all()
    assert tabular_features(data).shape[0] == len(data)
    seq = torch.from_numpy(data.seq[:16])
    static = torch.from_numpy(data.static[:16])
    for arch in ARCHITECTURES.values():
        out = arch(5, static.shape[1], data.event_bin.shape[1], HORIZON)(seq, static)
        assert out.shape == (16, data.event_bin.shape[1], HORIZON)


def test_surrogate_fits_and_predicts():
    from spareparts.surrogate.doe import OUTPUTS, PARAMS, sample
    from spareparts.surrogate.model import DeepEnsemble

    X = sample(30, seed=1)
    Y = X.assign(**{k: X["pm_factor"] * (i + 1) for i, k in enumerate(OUTPUTS)})
    ens = DeepEnsemble(n_models=2, hidden=16)
    ens.fit(X[list(PARAMS)], Y, epochs=20)
    mean, std = ens.predict(X.head(3)[list(PARAMS)])
    assert list(mean.columns) == OUTPUTS and (std.to_numpy() >= 0).all()


def test_weibull_fit_recovers_parameters_under_censoring():
    pytest.importorskip("lifelines")
    from spareparts.reliability.weibull import fit_weibull

    rng = np.random.default_rng(0)
    life = 3000 * rng.weibull(2.5, 3000)
    planned = rng.uniform(500, 4000, 3000)  # remplacements préventifs : censure à droite
    observed = np.minimum(life, planned)
    fit = fit_weibull(observed, life <= planned)
    assert abs(fit["beta"] - 2.5) < 0.2
    assert abs(fit["eta"] - 3000) < 150


def test_azure_lifetimes_mark_failures_and_censoring():
    import pandas as pd

    from spareparts.reliability.weibull import lifetimes_from_azure

    t = pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"])
    maint = pd.DataFrame({"datetime": t, "machineID": 1, "comp": "comp1"})
    failures = pd.DataFrame({"datetime": [t[1]], "machineID": [1], "failure": ["comp1"]})
    lt = lifetimes_from_azure(maint, failures)
    assert lt["event"].tolist() == [True, False]
    assert lt["duration"].iloc[0] == 31 * 24


def test_azure_dataset_does_not_leak_the_target():
    from spareparts.data import azure

    if not azure.is_complete():
        pytest.skip("jeu Azure absent de data/raw")
    from spareparts.predict.azure_dataset import build_azure_dataset

    data = build_azure_dataset()
    n_models = 4
    since = data.static[:, n_models + 1 :]  # temps depuis le dernier remplacement, par composant
    failing_now = data.event_bin == 0
    # Une panne datée de l'instant de prédiction ne doit pas apparaître comme un remplacement.
    assert (since[failing_now] > 0).all()
    assert set(np.unique(data.group)) == set(range(1, 13))


def test_telemetry_window_matches_training_layout():
    sim = FleetSimulation(load_scenario("s1_nominal"), seed=3, log_events=False)
    sim.enable_window(48)
    for _ in range(3):
        sim.advance_day()
    window = sim.telemetry_window()
    assert window.shape == (sim.M, 48, 5)
    assert set(np.unique(window[..., 4])) <= {0.0, 1.0}


def test_predictive_policy_runs_with_a_trained_network(tmp_path):
    from spareparts.policies.predictive import PredictivePolicy
    from spareparts.predict.training import split_by_group, train_network

    data = simulate_dataset(seeds=(1, 2, 3), days=6)
    train, val, test = split_by_group(data, val_groups=[2], test_groups=[3])
    model, metrics = train_network("gru", train, val, test, epochs=1, log=lambda *_: None)
    assert 0.0 <= metrics["rappel"] <= 1.0
    path = tmp_path / "gru.pt"
    torch.save(model.state_dict(), path)
    result = FleetSimulation(load_scenario("s1_nominal"), seed=4).run(
        PredictivePolicy(path, "gru"), days=5
    )
    assert result.days == 5


def test_branching_dqn_learns_and_plays_valid_actions():
    from spareparts.rl.bdq import BDQConfig, BDQPolicy, train_bdq

    env = FleetEnv(level="n2", horizon_days=6)
    cfg = BDQConfig(total_steps=24, learning_starts=8, batch_size=8, target_every=5)
    net = train_bdq(env, cfg, log=lambda *_: None)
    assert len(net.sizes) == len(env.action_space.nvec)
    result = FleetSimulation(load_scenario("s1_nominal"), seed=5).run(BDQPolicy(net), days=4)
    assert result.days == 4


def test_trained_agent_runs_inside_the_simulator(tmp_path):
    from sb3_contrib import MaskablePPO

    from spareparts.policies.learned import LearnedPolicy

    env = FleetEnv(level="n1", horizon_days=20)
    model = MaskablePPO("MlpPolicy", env, n_steps=20, batch_size=20, seed=0, device="cpu")
    model.learn(total_timesteps=40)
    path = tmp_path / "agent.zip"
    model.save(path)
    policy = LearnedPolicy(path, "n1", name="P4", label="test")
    result = FleetSimulation(load_scenario("s1_nominal"), seed=2).run(policy, days=10)
    assert result.days == 10 and result.kpis["cout_total"] > 0
