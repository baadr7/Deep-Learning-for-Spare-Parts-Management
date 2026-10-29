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
