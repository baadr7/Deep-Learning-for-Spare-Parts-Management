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


def test_calibration_curve_measures_reliability():
    from spareparts.predict.training import classification_curve, classification_metrics

    # Probabilités constantes par tranche : la fréquence observée doit suivre la probabilité.
    proba = np.repeat([0.1, 0.5, 0.9], 100)
    rng = np.random.default_rng(0)
    y = (rng.random(proba.size) < proba).astype(int)
    points, ece = classification_curve(y, proba)
    assert sum(effectif for *_, effectif in points) == proba.size
    assert ece < 0.15  # un tirage conforme aux probabilités annoncées reste bien calibré
    assert 0.0 <= classification_metrics(y, proba, 0.5)["ece"] <= 1.0


def test_azure_lifetimes_mark_failures_and_censoring():
    import pandas as pd

    from spareparts.reliability.weibull import lifetimes_from_azure

    t = pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"])
    maint = pd.DataFrame({"datetime": t, "machineID": 1, "comp": "comp1"})
    failures = pd.DataFrame({"datetime": [t[1]], "machineID": [1], "failure": ["comp1"]})
    lt = lifetimes_from_azure(maint, failures)
    assert lt["event"].tolist() == [True, False]
    assert lt["duration"].iloc[0] == 31 * 24


def write_azure_fixture(raw):
    """Écrit un jeu Azure minimal mais conforme au schéma des cinq CSV officiels.

    Le test ne peut pas dépendre du jeu complet — plusieurs gigaoctets, absent du dépôt — sans
    quoi il serait ignoré la plupart du temps. On reconstitue donc l'historique dont
    ``build_azure_dataset`` a besoin : quatre machines de modèles distincts, un historique
    quotidien couvrant les douze mois, et sur la première machine le cas limite qui porte le
    garde-fou — une panne et le remplacement qu'elle déclenche, horodatés au même instant.
    """
    import pandas as pd

    t0 = pd.Timestamp("2015-01-01 06:00:00")
    days = pd.date_range(t0, periods=351, freq="D")
    step = np.arange(len(days))
    pd.DataFrame(
        {
            "datetime": days,
            "machineID": 1,
            "volt": 100.0 + step % 7,
            "rotate": 400.0 + step % 5,
            "pressure": 10.0 + step % 3,
            "vibration": 40.0 + step % 4,
        }
    ).to_csv(raw / "PdM_telemetry.csv", index=False)
    pd.DataFrame(
        {
            "machineID": [1, 2, 3, 4],
            "model": ["model1", "model2", "model3", "model4"],
            "age": [5, 10, 15, 20],
        }
    ).to_csv(raw / "PdM_machines.csv", index=False)
    pd.DataFrame(
        {
            "datetime": [t0 + pd.Timedelta(days=10), t0 + pd.Timedelta(days=200)],
            "machineID": [1, 3],
            "errorID": ["error1", "error2"],
        }
    ).to_csv(raw / "PdM_errors.csv", index=False)
    failure_day = t0 + pd.Timedelta(days=100)
    pd.DataFrame(
        {
            "datetime": [failure_day],
            "machineID": [1],
            "failure": ["comp1"],
        }
    ).to_csv(raw / "PdM_failures.csv", index=False)
    # Comp1 est remplacé une fois de façon planifiée (jour 50), puis à la suite de sa panne
    # (jour 100) : ce second remplacement porte le même horodatage que la panne.
    pd.DataFrame(
        {
            "datetime": [t0 + pd.Timedelta(days=50 + k) for k in range(4)] + [failure_day],
            "machineID": [1] * 5,
            "comp": ["comp1", "comp2", "comp3", "comp4", "comp1"],
        }
    ).to_csv(raw / "PdM_maint.csv", index=False)


def test_azure_dataset_does_not_leak_the_target(tmp_path, monkeypatch):
    from spareparts.data import azure

    monkeypatch.setattr(azure, "RAW_DIR", tmp_path)
    write_azure_fixture(tmp_path)
    assert azure.is_complete()
    from spareparts.predict.azure_dataset import build_azure_dataset

    data = build_azure_dataset()
    n_models = 4
    since = data.static[:, n_models + 1 :]  # temps depuis le dernier remplacement, par composant
    failing_now = data.event_bin == 0
    # Le jeu doit contenir le cas limite, sinon le garde-fou ne prouverait rien.
    assert failing_now.any()
    # Une panne datée de l'instant de prédiction ne doit pas apparaître comme un remplacement :
    # le remplacement qu'elle déclenche est daté du même instant et doit être écarté. Le dernier
    # remplacement vu est alors celui, planifié, du jour 50, soit (100 − 50) jours / 1 000 = 1,2 ;
    # la valeur 0 trahirait la fuite, la valeur 3,0 l'absence de tout remplacement antérieur.
    assert (since[failing_now] > 0).all()
    assert np.allclose(since[failing_now], 1.2)
    assert set(np.unique(data.group)) == set(range(1, 13))


def test_azure_normalization_uses_training_period_only(tmp_path, monkeypatch):
    import pandas as pd

    from spareparts.data import azure

    monkeypatch.setattr(azure, "RAW_DIR", tmp_path)
    write_azure_fixture(tmp_path)
    from spareparts.predict.azure_dataset import build_azure_dataset

    original = build_azure_dataset()
    telemetry_path = tmp_path / "PdM_telemetry.csv"
    telemetry = pd.read_csv(telemetry_path, parse_dates=["datetime"])
    test_period = telemetry["datetime"] >= pd.Timestamp("2015-09-01")
    telemetry.loc[test_period, ["volt", "rotate", "pressure", "vibration"]] += 10_000
    telemetry.to_csv(telemetry_path, index=False)
    changed_holdout = build_azure_dataset()

    train = original.group <= 8
    holdout = original.group >= 9
    np.testing.assert_array_equal(original.seq[train], changed_holdout.seq[train])
    assert not np.allclose(original.seq[holdout], changed_holdout.seq[holdout])


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


def test_branch_reward_sums_to_the_global_reward():
    # La récompense par branche sert à entraîner chaque tête du BDQ sur sa propre décision ; pour
    # que la cible de valeur reste cohérente, sa somme doit reproduire exactement la récompense
    # globale (moyenne des cibles par branche).
    env = FleetEnv(level="n2", horizon_days=4, branch_reward=True)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    branch = info["reward_branch"]
    assert branch.shape == (int(env.action_space.nvec.size),)
    assert branch.dtype == np.float32
    assert np.isclose(float(branch.sum()), reward, atol=1e-4)
    assert obs.shape == env.observation_space.shape


def test_branch_reward_is_ignored_outside_the_n2_level():
    env = FleetEnv(level="n1", horizon_days=3, branch_reward=True)
    env.reset(seed=0)
    *_, info = env.step(env.action_space.sample())
    assert "reward_branch" not in info


def test_branching_dqn_trains_with_per_branch_rewards():
    from spareparts.rl.bdq import BDQConfig, train_bdq

    env = FleetEnv(level="n2", horizon_days=6, branch_reward=True)
    cfg = BDQConfig(
        total_steps=24, learning_starts=8, batch_size=8, target_every=5, branch_reward=True
    )
    net = train_bdq(env, cfg, log=lambda *_: None)
    assert len(net.sizes) == len(env.action_space.nvec)


def test_risk_guided_masking_hides_low_risk_components():
    from spareparts.envs.codecs import Context, Level2Codec
    from spareparts.sim.states import MachineState

    sim = FleetSimulation(load_scenario("s1_nominal"), seed=0, log_events=False)
    sim.advance_day()
    obs = sim.observe()
    # Tous les composants deviennent éligibles : machine en marche, pièce en stock, composant
    # usé au-delà de la moitié de son intervalle, technicien libre.
    obs.machine_state[:] = MachineState.OPERATIONAL
    obs.stock[:] = 10
    obs.age_unit[:] = obs.pm_interval * 2.0
    obs.techs_free = sim.M
    ctx = Context.from_sim(sim, 365, use_risk=True, risk_min=0.2)
    codec = Level2Codec(ctx)
    start = int(codec.nvec[: 2 * ctx.n_parts].sum())
    width = ctx.n_components + 1

    obs.risk = np.zeros_like(obs.age_unit)
    low = codec.masks(obs)[start:].reshape(ctx.n_machines, width)
    # Sous le seuil, la maintenance préventive est interdite : seule l'option « ne rien faire »
    # reste ouverte pour chaque machine.
    assert not low[:, 1:].any()
    assert low[:, 0].all()

    obs.risk = np.ones_like(obs.age_unit)
    high = codec.masks(obs)[start:].reshape(ctx.n_machines, width)
    assert high[:, 1:].all()


def test_randomized_env_samples_scenario_parameters():
    from spareparts.envs.fleet_env import RANDOMIZATION_RANGES

    env = FleetEnv(level="n1", horizon_days=3, randomize=RANDOMIZATION_RANGES)
    _, info = env.reset(seed=0)
    assert set(info["overrides"]) == set(RANDOMIZATION_RANGES)
    technicians = info["overrides"]["technicians"]
    lo, hi = RANDOMIZATION_RANGES["technicians"]
    assert isinstance(technicians, int) and lo <= technicians <= hi
    # Le tirage suit la graine : une autre graine change les paramètres de l'épisode.
    _, other = env.reset(seed=1)
    assert other["overrides"] != info["overrides"]
    # Le codec est reconstruit sur les nouvelles bornes et l'épisode reste jouable.
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and np.isfinite(reward)


def test_recurrent_ppo_trains_and_predicts():
    # Variante à mémoire du § 5.3 : elle éprouve l'observabilité partielle sans changer le codec.
    from sb3_contrib import RecurrentPPO

    env = FleetEnv(level="n1", horizon_days=5)
    model = RecurrentPPO(
        "MlpLstmPolicy", env, n_steps=10, batch_size=10, gamma=0.995, seed=0, device="cpu"
    )
    model.learn(total_timesteps=20)
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert np.asarray(action).shape == (int(env.action_space.nvec.size),)


def test_recurrent_policy_runs_inside_the_simulator(tmp_path):
    # La variante à mémoire (§ 6, amélioration 13) doit se rejouer dans le simulateur comme les
    # autres agents : même codec N1, état récurrent remis à zéro au début de chaque épisode.
    from sb3_contrib import RecurrentPPO

    from spareparts.policies.learned import LearnedPolicy

    env = FleetEnv(level="n1", horizon_days=20)
    model = RecurrentPPO("MlpLstmPolicy", env, n_steps=20, batch_size=20, seed=0, device="cpu")
    model.learn(total_timesteps=40)
    path = tmp_path / "rppo.zip"
    model.save(path)
    policy = LearnedPolicy(path, "n1", name="RPPO", label="test", algo="recurrent_ppo")
    result = FleetSimulation(load_scenario("s1_nominal"), seed=2).run(policy, days=10)
    assert result.days == 10 and result.kpis["cout_total"] > 0


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


def test_padded_distribution_matches_sb3_contrib():
    from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution

    from spareparts.rl.padded import PaddedMaskableMultiCategorical

    env = FleetEnv(level="n2", horizon_days=5)
    env.reset(seed=1)
    dims = [int(n) for n in env.action_space.nvec]
    torch.manual_seed(0)
    logits = torch.randn(6, sum(dims))
    masks = np.tile(env.action_masks(), (6, 1))
    reference = MaskableMultiCategoricalDistribution(dims).proba_distribution(logits)
    reference.apply_masking(masks)
    padded = PaddedMaskableMultiCategorical(dims).proba_distribution(logits)
    padded.apply_masking(masks)

    actions = reference.sample()
    assert torch.allclose(padded.log_prob(actions), reference.log_prob(actions), atol=1e-4)
    assert torch.allclose(padded.entropy(), reference.entropy(), atol=1e-4)
    assert torch.equal(padded.mode(), reference.mode())
    # Les tirages ne choisissent jamais une option masquée.
    sampled = padded.sample().numpy()
    starts = np.concatenate([[0], np.cumsum(dims)[:-1]])
    assert masks[np.arange(6)[:, None], starts[None, :] + sampled].all()


def test_best_model_saver_annualises_curriculum_episodes(tmp_path):
    """Le gardien du meilleur modèle compare les tirages à durée égale.

    Sous curriculum, les épisodes commencent à 90 jours : leur retour cumulé est bien moins
    négatif que celui d'une année complète, si bien qu'un gardien naïf retiendrait le tout
    premier tirage, avant tout apprentissage. On vérifie qu'un tirage long convergé l'emporte,
    puis qu'un tirage ultérieur moins bon ne l'écrase pas.
    """
    import importlib.util

    from spareparts.config import ROOT

    spec = importlib.util.spec_from_file_location("train_rl", ROOT / "scripts" / "train_rl.py")
    train_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_rl)

    class _FakeModel:
        def __init__(self):
            self.ep_info_buffer = []
            self.saved = []

        def save(self, path):
            self.saved.append(list(self.ep_info_buffer))

    saver = train_rl.best_model_saver(tmp_path / "agent.zip", horizon=365)
    saver.model = _FakeModel()

    saver.model.ep_info_buffer = [{"r": -100.0, "l": 90}] * 40
    saver._on_rollout_end()
    assert len(saver.model.saved) == 1
    assert saver.best == pytest.approx(-100.0 / 90 * 365)

    # Un tirage d'un an à −360 est meilleur à l'année (−360 > −405,6) : il est retenu, alors que
    # comparer les récompenses brutes (−360 > −100 est faux) l'aurait écarté.
    saver.model.ep_info_buffer = [{"r": -360.0, "l": 365}] * 40
    saver._on_rollout_end()
    assert len(saver.model.saved) == 2
    assert saver.best == pytest.approx(-360.0)

    saver.model.ep_info_buffer = [{"r": -450.0, "l": 365}] * 40
    saver._on_rollout_end()
    assert len(saver.model.saved) == 2
    assert saver.best == pytest.approx(-360.0)


def test_weibull_per_component_and_accelerated_fit():
    import pandas as pd

    from spareparts.reliability.weibull import fit_aft, fit_by_component

    # Durées de vie engendrées par deux lois distinctes : l'ajustement par composant doit
    # retrouver deux échelles différentes, l'ajustement accéléré doit produire des coefficients.
    rng = np.random.default_rng(0)
    rows = []
    for machine in range(1, 41):
        for comp, eta in (("comp1", 3000.0), ("comp2", 5000.0)):
            for _ in range(4):
                rows.append(
                    {
                        "machineID": machine,
                        "comp": comp,
                        "duration": float(eta * rng.weibull(2.0)),
                        "event": True,
                    }
                )
    lifetimes = pd.DataFrame(rows)
    by_component = fit_by_component(lifetimes)
    assert set(by_component) == {"comp1", "comp2"}
    assert abs(by_component["comp1"]["eta"] - 3000.0) < 600.0
    assert by_component["comp2"]["eta"] > by_component["comp1"]["eta"]

    machines = pd.DataFrame(
        {
            "machineID": list(range(1, 41)),
            "model": [f"model{m % 4 + 1}" for m in range(1, 41)],
            "age": [float(m % 20) for m in range(1, 41)],
        }
    )
    summary = fit_aft(lifetimes, machines)
    assert "coef" in summary.columns and not summary.empty


def test_surrogate_saves_loads_and_scores(tmp_path):
    from spareparts.surrogate.doe import OUTPUTS, PARAMS, sample
    from spareparts.surrogate.model import DeepEnsemble, score

    X = sample(24, seed=2)
    Y = X.assign(**{k: X["pm_factor"] * (i + 1) for i, k in enumerate(OUTPUTS)})
    ensemble = DeepEnsemble(n_models=2, hidden=8)
    ensemble.fit(X[list(PARAMS)], Y, epochs=5)

    path = tmp_path / "ensemble.pt"
    ensemble.save(path, metrics={"r2": 0.9})
    reloaded, metrics = DeepEnsemble.load(path)
    assert metrics == {"r2": 0.9}
    assert len(reloaded.models) == 2

    mean, std = reloaded.predict(X.head(3)[list(PARAMS)])
    assert list(mean.columns) == OUTPUTS and (std.to_numpy() >= 0).all()

    stats = score(Y.head(3), mean)
    assert set(stats) == set(OUTPUTS)
    assert all(isinstance(value["r2"], float) for value in stats.values())


def test_surrogate_api_serves_the_trained_ensemble(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from spareparts import surrogate
    from spareparts.api import app as api
    from spareparts.surrogate.doe import OUTPUTS, PARAMS, sample
    from spareparts.surrogate.model import DeepEnsemble

    X = sample(40, seed=5)
    Y = X.assign(**{k: X["pm_factor"] * (i + 1) for i, k in enumerate(OUTPUTS)})
    ensemble = DeepEnsemble(n_models=2, hidden=8)
    ensemble.fit(X[list(PARAMS)], Y, epochs=10)
    path = tmp_path / "ensemble.pt"
    ensemble.save(path, metrics={"r2": 0.87})

    api._surrogate.clear()
    monkeypatch.setattr(surrogate, "MODEL_PATH", path)
    try:
        with TestClient(api.app) as client:
            status = client.get("/api/surrogate").json()
            assert status["available"] is True
            assert status["metrics"] == {"r2": 0.87}
            assert set(status["outputs"]) == set(OUTPUTS)
            assert set(status["params"]) == set(PARAMS)

            payload = {name: (lo + hi) / 2 for name, (lo, hi, _) in PARAMS.items()}
            prediction = client.post("/api/surrogate/predict", json=payload).json()
            assert set(prediction) == set(OUTPUTS)
            assert all({"mean", "std"} <= value.keys() for value in prediction.values())
    finally:
        api._surrogate.clear()
